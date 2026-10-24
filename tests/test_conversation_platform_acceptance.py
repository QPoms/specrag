"""End-to-end acceptance for the Conversation-centered research lifecycle."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from specrag.access import ServiceDelegationSigner
from specrag.agents import ConversationAgent
from specrag.interfaces.workspace_services import WorkspaceServices
from specrag.models import (
    CorpusVersionRef,
    MCPResearchResultResponse,
    NativeFunctionCall,
)
from specrag.orchestration import (
    ConversationTurnWorker,
    ResearchLifecycleWorker,
    ResearchToolService,
    RunHarness,
    RunWorker,
)
from specrag.prompts import (
    CONVERSATION_AGENT_INSTRUCTIONS,
    CONVERSATION_AGENT_PROMPT_VERSION,
)


SECRET = b"conversation-platform-acceptance-secret"
VERSION = CorpusVersionRef(
    specification="24.501", version="17.9.0", release=17
)


class _Workflow:
    def resume(self, *, thread_id):
        return {"stop_reason": "insufficient_evidence"}

    def checkpoint(self, *, thread_id):
        return SimpleNamespace(values={"thread_id": thread_id})


class _Application:
    workflow = _Workflow()

    def run(self, **kwargs):
        return {"stop_reason": "insufficient_evidence"}


def _factory(release, control, event_sink, research_context):
    return _Application()


def _call(name: str, arguments: dict[str, object], suffix: str):
    return NativeFunctionCall(
        call_id=f"provider-{suffix}", name=name, arguments=arguments
    )


class _AcceptanceModel:
    """Deterministic model double exercising all Conversation Agent branches."""

    def __init__(self) -> None:
        self.step = 0
        self.active_scopes = []

    def call_tools(self, **kwargs):
        self.step += 1
        payload = kwargs["input_payload"]
        self.active_scopes.append(payload.get("active_research_scope"))
        prior_outputs = kwargs["prior_outputs"]
        available = payload.get("available_research_results", [])

        if self.step == 1:
            calls = (_call("respond", {
                "text": "你好，我们可以一起梳理研究问题。",
                "source_kind": "direct_conversation",
                "research_result_ids": [],
                "run_ids": [],
                "citations": [],
            }, "direct"),)
        elif self.step == 2:
            calls = (_call("start_research", {
                "question": "When shall the UE initiate registration?",
                "conditions": ["UE is not registered"],
                "requested_scope": {
                    "policy": "highest_version_within_release",
                    "release": 17,
                    "versions": [],
                },
            }, "research-1"),)
        elif self.step in {3, 6}:
            result = prior_outputs[0].output["result"]
            calls = (_call("respond", {
                "text": result["answer"],
                "source_kind": "research_result",
                "research_result_ids": [result["result_id"]],
                "run_ids": [result["run_id"]],
                "citations": result["evidence_ids"],
            }, f"answer-{self.step}"),)
        elif self.step == 4:
            result = available[-1]
            calls = (_call("respond", {
                "text": result["answer"],
                "source_kind": "historical_result_readback",
                "research_result_ids": [result["result_id"]],
                "run_ids": [result["run_id"]],
                "citations": result["evidence_ids"],
            }, "readback"),)
        elif self.step == 5:
            result = available[-1]
            calls = (_call("start_research", {
                "question": "Re-evaluate registration under the changed condition.",
                "conditions": ["UE is deregistered after a network-triggered event"],
                "requested_scope": result["requested_scope"],
                "prior_results": [{
                    "result_id": result["result_id"],
                    "content_sha256": result["content_sha256"],
                    "relationship": "condition_change",
                }],
            }, "research-2"),)
        else:  # pragma: no cover - protects the scripted acceptance contract
            raise AssertionError(f"unexpected model step {self.step}")
        return SimpleNamespace(calls=calls, replay_items=())


class _InProcessMCPClient:
    """Exercises the same signed delegation contract without a network socket."""

    def __init__(self, services, research, signer) -> None:
        self.services = services
        self.research = research
        self.signer = signer

    def _trusted(self, token):
        claim = self.signer.verify(token)
        actor = self.services.identity.delegated_context(
            user_id=claim.user_id,
            workspace_id=claim.workspace_id,
            delegation_id=claim.nonce,
            authenticated_at=claim.issued_at,
            expires_at=claim.expires_at,
        )
        return actor, claim

    async def start_research(self, token, request):
        actor, claim = self._trusted(token)
        return await self.research.start_research(actor, claim, request)

    async def get_research_status(self, token, run_id):
        actor, _ = self._trusted(token)
        return self.research.get_research_status(actor, run_id)

    async def get_research_result(
        self, token, *, result_id=None, run_id=None
    ):
        actor, _ = self._trusted(token)
        return MCPResearchResultResponse(
            result=self.research.get_research_result(
                actor, result_id=result_id, run_id=run_id
            ).result
        )


class _AsyncModel:
    def __init__(self) -> None:
        self.step = 0

    def call_tools(self, **kwargs):
        self.step += 1
        if self.step == 1:
            calls = (_call("start_research", {
                "question": "What does the specification require?",
                "requested_scope": {
                    "policy": "highest_version_within_release",
                    "release": 17,
                    "versions": [],
                },
            }, "async-start"),)
        elif self.step == 2:
            result = kwargs["input_payload"]["available_research_results"][-1]
            calls = (_call("respond", {
                "text": result["answer"],
                "source_kind": "research_result",
                "research_result_ids": [result["result_id"]],
                "run_ids": [result["run_id"]],
                "citations": result["evidence_ids"],
            }, "async-result"),)
        else:  # pragma: no cover
            raise AssertionError(f"unexpected model step {self.step}")
        return SimpleNamespace(calls=calls, replay_items=())


def _append_and_process(services, worker, actor, conversation_id, text, key):
    accepted = services.conversations.append_user_message(
        actor, conversation_id, text=text, operation_key=key
    )
    assert worker.process_one()
    stored = services.conversation_agent_store.turn(accepted.turn.turn_id)
    assert stored is not None and stored.status == "completed"


def test_persistent_conversation_research_readback_and_condition_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "platform.sqlite3"
    services = WorkspaceServices.create(path)
    services.identity.bootstrap("alice", "correct horse battery")
    token, _ = services.identity.login("alice", "correct horse battery")
    workspace = services.identity.create_workspace(token, "Research Workspace")
    actor = services.identity.context(token, workspace.workspace_id)
    conversation = services.conversations.create(actor).conversation

    harness = RunHarness(
        services.runs,
        _factory,
        corpus_snapshot=lambda _scope: (VERSION,),
        result_lookup=lambda run_id: (
            item.result_id
            if (item := services.result_store.latest_for_run(run_id))
            else None
        ),
    )
    run_worker = RunWorker(
        harness, services.run_queue, poll_seconds=0.01, retry_delay_seconds=0.01
    )
    lifecycle = ResearchLifecycleWorker(
        services.runs,
        services.research_results,
        services.conversation_lifecycle_store,
        services.completion_queue,
        poll_seconds=0.01,
        retry_delay_seconds=0.01,
    )
    research = ResearchToolService(
        harness,
        run_worker,
        services.research_results,
        services.conversations,
        services.conversation_lifecycle_store,
        services.conversation_scope_store,
        wait_seconds=2,
        poll_seconds=0.01,
    )
    signer = ServiceDelegationSigner(SECRET)
    model = _AcceptanceModel()
    agent = ConversationAgent(
        model,
        instructions=CONVERSATION_AGENT_INSTRUCTIONS,
        prompt_version=CONVERSATION_AGENT_PROMPT_VERSION,
    )
    turn_worker = ConversationTurnWorker(
        queue=services.conversation_turn_queue,
        store=services.conversation_agent_store,
        conversations=services.conversation_store,
        lifecycle=services.conversation_lifecycle_store,
        identity=services.identity,
        context_harness=services.context_harness,
        memory_recall=services.memory_recall,
        memory=services.memory,
        results=services.research_results,
        agent=agent,
        mcp_client=_InProcessMCPClient(services, research, signer),
        delegation_signer=signer,
    )
    run_worker.start()
    lifecycle.start()
    try:
        conversation_id = conversation.conversation_id
        _append_and_process(
            services, turn_worker, actor, conversation_id,
            "你好", "ordinary-message",
        )
        assert services.runs.list_runs(
            actor.workspace_id, owner_user_id=actor.user_id, limit=10
        ) == ()

        _append_and_process(
            services, turn_worker, actor, conversation_id,
            "When shall the UE initiate registration?", "research-message",
        )
        first_results = services.research_results.list_for_conversation(
            actor, conversation_id
        )
        assert len(first_results) == 1

        _append_and_process(
            services, turn_worker, actor, conversation_id,
            "请回读刚才结论的依据。", "readback-message",
        )
        assert len(services.runs.list_runs(
            actor.workspace_id, owner_user_id=actor.user_id, limit=10
        )) == 1

        _append_and_process(
            services, turn_worker, actor, conversation_id,
            "把条件改成网络事件后 UE 已注销，再重新研究。", "changed-condition",
        )
        results = services.research_results.list_for_conversation(
            actor, conversation_id
        )
        assert len(results) == 2
        oldest, newest = results
        assert newest.result_id != oldest.result_id
        assert newest.conditions != oldest.conditions
        assert newest.prior_results[0].result_id == oldest.result_id
        assert newest.prior_results[0].relationship == "condition_change"
        scoped_conversation = services.conversation_store.get(conversation_id)
        assert scoped_conversation is not None
        assert scoped_conversation.active_scope_revision_id is not None
        active_scope = services.conversation_store.get_scope_revision(
            scoped_conversation.active_scope_revision_id
        )
        assert active_scope is not None and active_scope.revision == 2
        assert active_scope.conditions == newest.conditions
        assert active_scope.predecessor_id is not None
        inherited = [scope for scope in model.active_scopes if scope is not None]
        assert inherited
        assert inherited[0]["conditions"] == ["UE is not registered"]

        messages = services.conversation_store.messages(conversation_id)
        assert [item.role for item in messages] == [
            "user", "assistant", "user", "assistant",
            "user", "assistant", "user", "assistant",
        ]
        assistant_sources = [
            item.content["source_kind"] for item in messages
            if item.role == "assistant"
        ]
        assert assistant_sources == [
            "direct_conversation",
            "research_result",
            "historical_result_readback",
            "research_result",
        ]
        assert all(
            link.presentation_status == "presented"
            for link in services.conversation_lifecycle_store.list_for_conversation(
                conversation_id
            )
        )
    finally:
        run_worker.stop()
        lifecycle.stop()
        services.close()

    restored = WorkspaceServices.create(path)
    try:
        restored_actor = restored.identity.context(token, workspace.workspace_id)
        detail = restored.conversations.get(restored_actor, conversation_id)
        assert len(detail.messages) == 8
        assert len(restored.research_results.list_for_conversation(
            restored_actor, conversation_id
        )) == 2
    finally:
        restored.close()


def test_async_result_queues_exactly_one_internal_presentation_turn(
    tmp_path: Path,
) -> None:
    services = WorkspaceServices.create(tmp_path / "async-platform.sqlite3")
    services.identity.bootstrap("alice", "correct horse battery")
    token, _ = services.identity.login("alice", "correct horse battery")
    workspace = services.identity.create_workspace(token, "Async Workspace")
    actor = services.identity.context(token, workspace.workspace_id)
    conversation = services.conversations.create(actor).conversation
    accepted = services.conversations.append_user_message(
        actor,
        conversation.conversation_id,
        text="What does the specification require?",
        operation_key="async-request",
    )
    harness = RunHarness(
        services.runs,
        _factory,
        corpus_snapshot=lambda _scope: (VERSION,),
        result_lookup=lambda run_id: (
            item.result_id
            if (item := services.result_store.latest_for_run(run_id))
            else None
        ),
    )
    run_worker = RunWorker(harness, services.run_queue)
    lifecycle = ResearchLifecycleWorker(
        services.runs,
        services.research_results,
        services.conversation_lifecycle_store,
        services.completion_queue,
    )
    research = ResearchToolService(
        harness,
        run_worker,
        services.research_results,
        services.conversations,
        services.conversation_lifecycle_store,
        services.conversation_scope_store,
        wait_seconds=0,
    )
    signer = ServiceDelegationSigner(SECRET)
    turn_worker = ConversationTurnWorker(
        queue=services.conversation_turn_queue,
        store=services.conversation_agent_store,
        conversations=services.conversation_store,
        lifecycle=services.conversation_lifecycle_store,
        identity=services.identity,
        context_harness=services.context_harness,
        memory_recall=services.memory_recall,
        memory=services.memory,
        results=services.research_results,
        agent=ConversationAgent(
            _AsyncModel(),
            instructions=CONVERSATION_AGENT_INSTRUCTIONS,
            prompt_version=CONVERSATION_AGENT_PROMPT_VERSION,
        ),
        mcp_client=_InProcessMCPClient(services, research, signer),
        delegation_signer=signer,
    )
    try:
        assert turn_worker.process_one()
        origin = services.conversation_agent_store.turn(accepted.turn.turn_id)
        assert origin is not None and origin.status == "completed"
        assert services.run_queue.pending_count() == 1

        assert run_worker.process_one()
        assert lifecycle.process_one()
        assert not lifecycle.process_one()
        assert services.conversation_turn_queue.pending_count() == 1
        assert turn_worker.process_one()
        assert services.conversation_turn_queue.pending_count() == 0

        messages = services.conversation_store.messages(
            conversation.conversation_id
        )
        assert [item.role for item in messages] == [
            "user", "assistant", "assistant"
        ]
        assert messages[1].content["source_kind"] == "direct_conversation"
        assert messages[2].content["source_kind"] == "research_result"
        links = services.conversation_lifecycle_store.list_for_conversation(
            conversation.conversation_id
        )
        assert len(links) == 1
        assert links[0].presentation_status == "presented"
        assert sum(
            event.event_type == "auto_reply_presented"
            for event in services.conversation_store.events(
                conversation.conversation_id
            )
        ) == 1
    finally:
        services.close()

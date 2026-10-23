from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from specrag.access import ServiceDelegationSigner
from specrag.agents import (
    ConversationAgent,
    ConversationToolExecution,
)
from specrag.interfaces.workspace_services import WorkspaceServices
from specrag.models import (
    ConversationTurnBudget,
    MCPResearchStartResponse,
    NativeFunctionCall,
    ResearchResult,
    SpecificationScope,
)
from specrag.orchestration import ConversationTurnWorker, MCPConversationToolExecutor
from specrag.prompts import (
    CONVERSATION_AGENT_INSTRUCTIONS,
    CONVERSATION_AGENT_PROMPT_VERSION,
)


SECRET = b"conversation-worker-test-secret-32-bytes"
SCOPE = SpecificationScope(policy="highest_version_within_release", release=17)


class _Model:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.requests = []

    def call_tools(self, **kwargs):
        self.requests.append(kwargs)
        calls = self.rounds.pop(0)
        return SimpleNamespace(calls=tuple(calls), replay_items=())


class _Executor:
    def __init__(self, executions=()):
        self.executions = list(executions)
        self.calls = []

    def execute(self, call, *, operation_key):
        self.calls.append((call, operation_key))
        return self.executions.pop(0)


class _UnusedMCP:
    async def start_research(self, token, request):
        raise AssertionError("ordinary conversation must not call MCP")

    async def get_research_status(self, token, run_id):
        raise AssertionError("ordinary conversation must not call MCP")

    async def get_research_result(self, token, *, result_id=None, run_id=None):
        raise AssertionError("ordinary conversation must not call MCP")


def _agent(model):
    return ConversationAgent(
        model,
        instructions=CONVERSATION_AGENT_INSTRUCTIONS,
        prompt_version=CONVERSATION_AGENT_PROMPT_VERSION,
    )


def _call(name: str, arguments: dict, suffix: str = "1") -> NativeFunctionCall:
    return NativeFunctionCall(call_id=f"provider-{suffix}", name=name, arguments=arguments)


def _result() -> ResearchResult:
    return ResearchResult(
        result_id="RESULT-" + "A" * 32,
        run_id="RUN-" + "B" * 32,
        workspace_id="WS-1",
        owner_user_id="USR-1",
        conversation_id="CONV-" + "C" * 32,
        evidence_status="insufficient",
        answer="The available sources do not establish the requested rule.",
        requested_scope=SCOPE,
        corpus_versions=(),
        source_outcome_sha256="1" * 64,
        content_sha256="2" * 64,
        created_at=datetime.now(UTC),
    )


def test_agent_answers_ordinary_conversation_without_research() -> None:
    model = _Model([[ _call("respond", {
        "text": "你好，我们可以先明确研究目标。",
        "source_kind": "direct_conversation",
        "research_result_ids": [],
        "run_ids": [],
        "citations": [],
    }) ]])
    executor = _Executor()
    outcome = _agent(model).run(
        {"recent_messages": []},
        budget=ConversationTurnBudget(),
        executor=executor,
        operation_prefix="test-turn",
    )
    assert outcome.message.source_kind == "direct_conversation"
    assert outcome.tool_calls == 0
    assert executor.calls == []


def test_direct_response_is_available_with_zero_external_tool_budget() -> None:
    model = _Model([[ _call("respond", {
        "text": "请补充你的研究条件。",
        "source_kind": "direct_conversation",
        "research_result_ids": [],
        "run_ids": [],
        "citations": [],
    }) ]])
    outcome = _agent(model).run(
        {},
        budget=ConversationTurnBudget(max_tool_calls=0),
        executor=_Executor(),
        operation_prefix="no-tools",
    )
    assert outcome.message.text == "请补充你的研究条件。"


def test_agent_uses_one_research_run_and_validates_current_result() -> None:
    result = _result()
    model = _Model([
        [_call("start_research", {"question": "What is required?"})],
        [_call("respond", {
            "text": result.answer,
            "source_kind": "research_result",
            "research_result_ids": [result.result_id],
            "run_ids": [result.run_id],
            "citations": [],
        }, "2")],
    ])
    executor = _Executor([ConversationToolExecution(
        output=MCPResearchStartResponse(
            disposition="completed",
            run_id=result.run_id,
            status="completed",
            result_id=result.result_id,
            result=result,
        ).model_dump(mode="json"),
        results=(result,),
    )])
    outcome = _agent(model).run(
        {"recent_messages": []},
        budget=ConversationTurnBudget(),
        executor=executor,
        operation_prefix="test-turn",
    )
    assert outcome.research_submissions == 1
    assert outcome.message.research_result_ids == (result.result_id,)
    assert len(executor.calls) == 1


def test_agent_rejects_direct_normative_claim_and_unavailable_result() -> None:
    normative = _Model([[ _call("respond", {
        "text": "3GPP TS 24.501 shall require this behavior.",
        "source_kind": "direct_conversation",
        "research_result_ids": [],
        "run_ids": [],
        "citations": [],
    }) ]])
    with pytest.raises(ValueError, match="requires a Research Result"):
        _agent(normative).run(
            {}, budget=ConversationTurnBudget(), executor=_Executor(),
            operation_prefix="normative",
        )

    invented = _Model([[ _call("respond", {
        "text": "A claimed result.",
        "source_kind": "historical_result_readback",
        "research_result_ids": ["RESULT-" + "F" * 32],
        "run_ids": [],
        "citations": [],
    }) ]])
    with pytest.raises(ValueError, match="unavailable"):
        _agent(invented).run(
            {}, budget=ConversationTurnBudget(), executor=_Executor(),
            operation_prefix="invented",
        )


def test_turn_worker_persists_direct_reply_and_releases_conversation(
    tmp_path: Path,
) -> None:
    services = WorkspaceServices.create(tmp_path / "turn.sqlite3")
    services.identity.bootstrap("alice", "correct horse battery")
    token, _ = services.identity.login("alice", "correct horse battery")
    workspace = services.identity.create_workspace(token, "Turn Workspace")
    actor = services.identity.context(token, workspace.workspace_id)
    conversation = services.conversations.create(actor).conversation
    accepted = services.conversations.append_user_message(
        actor,
        conversation.conversation_id,
        text="你好",
        operation_key="message-1",
        expected_revision=0,
    )
    model = _Model([[ _call("respond", {
        "text": "你好，需要研究哪一项规范？",
        "source_kind": "direct_conversation",
        "research_result_ids": [],
        "run_ids": [],
        "citations": [],
    }) ]])
    worker = ConversationTurnWorker(
        queue=services.conversation_turn_queue,
        store=services.conversation_agent_store,
        conversations=services.conversation_store,
        lifecycle=services.conversation_lifecycle_store,
        identity=services.identity,
        context_harness=services.context_harness,
        memory_recall=services.memory_recall,
        memory=services.memory,
        results=services.research_results,
        agent=_agent(model),
        mcp_client=_UnusedMCP(),
        delegation_signer=ServiceDelegationSigner(SECRET),
    )
    try:
        assert worker.process_one()
        turn = services.conversation_agent_store.turn(accepted.turn.turn_id)
        assert turn is not None and turn.status == "completed"
        stored = services.conversation_store.get(conversation.conversation_id)
        assert stored is not None and stored.active_turn_id is None
        messages = services.conversation_store.messages(conversation.conversation_id)
        assert [item.role for item in messages] == ["user", "assistant"]
        assert messages[-1].content["source_kind"] == "direct_conversation"
        assert services.conversation_turn_queue.pending_count() == 0
        assert {event.event_type for event in services.conversation_store.events(
            conversation.conversation_id
        )} >= {"turn_started", "message_appended", "turn_stopped"}
    finally:
        services.close()


def test_tool_audit_persists_arguments_full_result_and_idempotent_replay(
    tmp_path: Path,
) -> None:
    services = WorkspaceServices.create(tmp_path / "audit.sqlite3")
    services.identity.bootstrap("alice", "correct horse battery")
    token, _ = services.identity.login("alice", "correct horse battery")
    workspace = services.identity.create_workspace(token, "Audit Workspace")
    actor = services.identity.context(token, workspace.workspace_id)
    conversation = services.conversations.create(actor).conversation
    accepted = services.conversations.append_user_message(
        actor,
        conversation.conversation_id,
        text="Research registration requirements.",
        operation_key="message-1",
        expected_revision=0,
    )
    call_id = "CALL-" + "D" * 32
    arguments = {"question": "When shall the UE register?"}
    try:
        services.conversation_agent_store.start_turn(accepted.turn.turn_id)
        started = services.conversation_agent_store.start_tool_call(
            accepted.turn.turn_id,
            call_id=call_id,
            tool_name="start_research",
            arguments=arguments,
            operation_key="turn-1:step:1:call:1",
        )
        page = services.context_harness.record_tool_result(
            conversation.context_id,
            call_id,
            {"disposition": "accepted", "run_id": "RUN-" + "E" * 32},
        )
        completed = services.conversation_agent_store.finish_tool_call(
            call_id,
            tool_result_id=page.tool_result_id,
            run_id=None,
            result_id=None,
        )
        replay = services.conversation_agent_store.start_tool_call(
            accepted.turn.turn_id,
            call_id=call_id,
            tool_name="start_research",
            arguments=arguments,
            operation_key="turn-1:step:1:call:1",
        )
        assert started.status == "running"
        assert completed.status == replay.status == "completed"
        assert completed.tool_result_id == page.tool_result_id
        assert len(services.conversation_agent_store.tool_calls(
            accepted.turn.turn_id
        )) == 1
        stored = services.context_harness.store.get_tool_result(page.tool_result_id)
        assert stored is not None
        assert stored.content["disposition"] == "accepted"
        event_types = [event.event_type for event in services.conversation_store.events(
            conversation.conversation_id
        )]
        assert event_types.count("tool_call_started") == 1
        assert event_types.count("tool_call_completed") == 1
    finally:
        services.close()


def test_tool_arguments_are_validated_before_private_audit_persistence(
    tmp_path: Path,
) -> None:
    services = WorkspaceServices.create(tmp_path / "audit-validation.sqlite3")
    services.identity.bootstrap("alice", "correct horse battery")
    token, _ = services.identity.login("alice", "correct horse battery")
    workspace = services.identity.create_workspace(token, "Audit Workspace")
    actor = services.identity.context(token, workspace.workspace_id)
    conversation = services.conversations.create(actor).conversation
    accepted = services.conversations.append_user_message(
        actor,
        conversation.conversation_id,
        text="Check status",
        operation_key="message-1",
    )
    services.conversation_agent_store.start_turn(accepted.turn.turn_id)
    executor = MCPConversationToolExecutor(
        client=_UnusedMCP(),
        signer=ServiceDelegationSigner(SECRET),
        actor=actor,
        conversation=conversation,
        turn=accepted.turn,
        origin_message_id=accepted.message.message_id,
        origin_turn_id=accepted.turn.turn_id,
        submitted_user_sequence=accepted.message.sequence,
        audit=services.conversation_agent_store,
        context_harness=services.context_harness,
    )
    try:
        with pytest.raises(ValueError):
            executor.execute(
                _call("get_research_status", {
                    "run_id": "RUN-" + "A" * 32,
                    "user_id": "spoofed-user",
                }),
                operation_key="invalid-status-call",
            )
        assert services.conversation_agent_store.tool_calls(
            accepted.turn.turn_id
        ) == ()
    finally:
        services.close()

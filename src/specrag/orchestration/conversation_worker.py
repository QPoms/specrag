"""Durable Worker for serialized Conversation Agent Turns."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Event, Thread
from typing import Protocol
from uuid import uuid4

from specrag.access import ServiceDelegationSigner
from specrag.agents import ConversationAgent
from specrag.context import ContextHarness
from specrag.memory import MemoryRecallService, MemoryService
from specrag.models import (
    ActorContext,
    Conversation,
    ConversationScopeRevision,
    ConversationTurn,
    NativeToolCapability,
    ResearchResult,
)

from .conversation_tools import MCPConversationToolExecutor, ResearchMCPClient
from .results import ResearchResultService


_SAFE_AGENT_REJECTIONS = {
    "specification-backed claim requires a Research Result": (
        "The model attempted to publish a specification-backed claim as direct "
        "conversation without an applicable Research Result."
    ),
    "respond must be the only call in its model step": (
        "The model returned a final response together with another tool call; "
        "the Conversation Agent rejected the ambiguous output."
    ),
    "conversation model returned no native call": (
        "The model returned no required Conversation Agent tool call."
    ),
}


def _turn_failure(exc: Exception) -> tuple[str, str]:
    public_message = getattr(exc, "public_message", None)
    if isinstance(public_message, str) and public_message.strip():
        return getattr(exc, "failure_kind", "internal_error"), public_message[:1000]
    rejection = _SAFE_AGENT_REJECTIONS.get(str(exc))
    if rejection is not None:
        return "structured_output", rejection
    return "internal_error", "Conversation Agent execution failed."


class ConversationQueue(Protocol):
    def claim(self, worker_id: str, *, lease_seconds: float) -> tuple[str, int] | None: ...
    def renew(self, turn_id: str, worker_id: str, *, lease_seconds: float) -> bool: ...
    def complete(self, turn_id: str, worker_id: str) -> bool: ...
    def release(
        self, turn_id: str, worker_id: str, *, delay_seconds: float
    ) -> bool: ...


class ConversationReadStore(Protocol):
    def get(self, conversation_id: str) -> Conversation | None: ...
    def messages(self, conversation_id: str) -> tuple[object, ...]: ...
    def get_scope_revision(
        self, scope_revision_id: str
    ) -> ConversationScopeRevision | None: ...


class ConversationAgentStore(Protocol):
    def turn(self, turn_id: str) -> ConversationTurn | None: ...
    def start_turn(self, turn_id: str, *, now=None) -> ConversationTurn: ...
    def tool_calls(self, turn_id: str) -> tuple[object, ...]: ...
    def finish_turn(
        self, turn_id: str, message, *, stop_reason: str,
        token_estimate: int, message_id: str, now=None,
    ): ...
    def fail_turn(
        self, turn_id: str, *, failure_kind: str, safe_message: str, now=None,
    ) -> ConversationTurn: ...


class ConversationIdentity(Protocol):
    def delegated_context(
        self, *, user_id: str, workspace_id: str, delegation_id: str,
        authenticated_at: datetime, expires_at: datetime,
    ) -> ActorContext: ...


class ConversationLifecycle(Protocol):
    def get_by_run(self, run_id: str): ...
    def schedule_pending(self, conversation_id: str, *, now=None): ...


class ConversationTurnWorker:
    def __init__(
        self,
        *,
        queue: ConversationQueue,
        store: ConversationAgentStore,
        conversations: ConversationReadStore,
        lifecycle: ConversationLifecycle,
        identity: ConversationIdentity,
        context_harness: ContextHarness,
        memory_recall: MemoryRecallService,
        memory: MemoryService,
        results: ResearchResultService,
        agent: ConversationAgent,
        mcp_client: ResearchMCPClient,
        delegation_signer: ServiceDelegationSigner,
        model_capability: NativeToolCapability | None = None,
        poll_seconds: float = 0.1,
        lease_seconds: float = 120.0,
    ) -> None:
        if poll_seconds <= 0 or lease_seconds <= 0:
            raise ValueError("Conversation Worker timing values must be positive")
        self.queue = queue
        self.store = store
        self.conversations = conversations
        self.lifecycle = lifecycle
        self.identity = identity
        self.context_harness = context_harness
        self.memory_recall = memory_recall
        self.memory = memory
        self.results = results
        self.agent = agent
        self.mcp_client = mcp_client
        self.delegation_signer = delegation_signer
        self.model_capability = model_capability
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.worker_id = "CONVERSATION-" + uuid4().hex.upper()
        self._stopping = Event()
        self._wake = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = Thread(
            target=self._run, name="specrag-conversation-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def run_forever(self) -> None:
        self._stopping.clear()
        self._run()

    def process_one(self) -> bool:
        claimed = self.queue.claim(self.worker_id, lease_seconds=self.lease_seconds)
        if claimed is None:
            return False
        turn_id, _attempt = claimed
        heartbeat_stop = Event()
        heartbeat = Thread(
            target=self._heartbeat,
            args=(turn_id, heartbeat_stop),
            name=f"specrag-conversation-lease-{turn_id[-8:]}",
            daemon=True,
        )
        heartbeat.start()
        conversation_id: str | None = None
        try:
            try:
                turn = self.store.start_turn(turn_id)
                if (
                    self.model_capability is not None
                    and self.model_capability.status != "ready"
                ):
                    self.store.fail_turn(
                        turn_id,
                        failure_kind=(
                            self.model_capability.failure_kind or "internal_error"
                        ),
                        safe_message=self.model_capability.message,
                    )
                    self.queue.complete(turn_id, self.worker_id)
                    return True
                conversation = self.conversations.get(turn.conversation_id)
                if conversation is None:
                    raise KeyError(turn.conversation_id)
                conversation_id = conversation.conversation_id
                actor = self._actor(conversation, turn)
                historical = {
                    result.result_id: result
                    for result in self.results.list_for_conversation(
                        actor, conversation.conversation_id, limit=20
                    )
                }
                query, origin_message_id, origin_turn_id, origin_sequence = self._origin(
                    turn, historical
                )
                recalled = self.memory_recall.recall(actor, query)
                active = self.context_harness.reconstruct(conversation.context_id)
                active_scope = (
                    self.conversations.get_scope_revision(
                        conversation.active_scope_revision_id
                    )
                    if conversation.active_scope_revision_id is not None
                    else None
                )
                payload = self._input_payload(
                    active, recalled.entries, tuple(historical.values()), turn,
                    active_scope,
                )
                executor = MCPConversationToolExecutor(
                    client=self.mcp_client,
                    signer=self.delegation_signer,
                    actor=actor,
                    conversation=conversation,
                    turn=turn,
                    origin_message_id=origin_message_id,
                    origin_turn_id=origin_turn_id,
                    submitted_user_sequence=origin_sequence,
                    audit=self.store,  # type: ignore[arg-type]
                    context_harness=self.context_harness,
                )
                outcome = self.agent.run(
                    payload,
                    budget=turn.budget,
                    executor=executor,
                    historical_results=historical,
                    trigger_result_id=turn.trigger_result_id,
                    stop_requested=lambda: self._cancelled(turn_id),
                    operation_prefix=f"conversation:{turn.conversation_id}:turn:{turn_id}",
                )
                self.store.finish_turn(
                    turn_id,
                    outcome.message,
                    stop_reason=outcome.stop_reason,
                    token_estimate=self.context_harness.estimator.estimate(
                        outcome.message.model_dump(mode="json")
                    ),
                    message_id="MSG-" + uuid4().hex.upper(),
                )
            except Exception as exc:
                try:
                    failure_kind, safe_message = _turn_failure(exc)
                    self.store.fail_turn(
                        turn_id,
                        failure_kind=failure_kind,
                        safe_message=safe_message,
                    )
                except Exception:
                    self.queue.release(
                        turn_id, self.worker_id, delay_seconds=self.poll_seconds
                    )
                    return True
            self.queue.complete(turn_id, self.worker_id)
            if conversation_id is not None:
                self.lifecycle.schedule_pending(conversation_id)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=self.poll_seconds + 0.5)
        return True

    def _run(self) -> None:
        while not self._stopping.is_set():
            if not self.process_one():
                self._wake.wait(self.poll_seconds)
                self._wake.clear()

    def _heartbeat(self, turn_id: str, stopping: Event) -> None:
        interval = max(0.05, self.lease_seconds / 3)
        while not stopping.wait(interval):
            if not self.queue.renew(
                turn_id, self.worker_id, lease_seconds=self.lease_seconds
            ):
                return

    def _actor(self, conversation: Conversation, turn: ConversationTurn) -> ActorContext:
        now = datetime.now(timezone.utc)
        return self.identity.delegated_context(
            user_id=conversation.owner_user_id,
            workspace_id=conversation.workspace_id,
            delegation_id=turn.turn_id,
            authenticated_at=now,
            expires_at=now + timedelta(minutes=5),
        )

    def _origin(
        self,
        turn: ConversationTurn,
        historical: dict[str, ResearchResult],
    ) -> tuple[str, str, str, int]:
        if turn.trigger == "user_message":
            if turn.trigger_message_id is None:
                raise RuntimeError("user Turn has no origin message")
            messages = self.conversations.messages(turn.conversation_id)
            message = next(
                (item for item in messages
                 if getattr(item, "message_id", None) == turn.trigger_message_id),
                None,
            )
            if message is None:
                raise RuntimeError("origin message is unavailable")
            content = getattr(message, "content", {})
            query = (
                content.text if hasattr(content, "text")
                else str(content.get("text", "")) if isinstance(content, dict)
                else str(content)
            )
            return (
                query, turn.trigger_message_id, turn.turn_id,
                turn.submitted_user_sequence,
            )
        if turn.trigger_run_id is None or turn.trigger_result_id is None:
            raise RuntimeError("Result Turn has no trigger binding")
        link = self.lifecycle.get_by_run(turn.trigger_run_id)
        result = historical.get(turn.trigger_result_id)
        if link is None or result is None:
            raise RuntimeError("automatic Result origin is unavailable")
        return (
            result.answer, link.origin_message_id, link.origin_turn_id,
            link.submitted_user_sequence,
        )

    @staticmethod
    def _input_payload(
        active, memories, results, turn: ConversationTurn,
        active_scope: ConversationScopeRevision | None,
    ) -> dict[str, object]:
        return {
            "security": {
                "conversation_history": "context_only_not_evidence",
                "memory": "untrusted_context_not_evidence",
                "research_results": "only_valid_source_for_specification_claims",
            },
            "trigger": turn.model_dump(mode="json"),
            "active_research_scope": (
                active_scope.model_dump(mode="json") if active_scope else None
            ),
            "summary": (
                active.summary.model_dump(mode="json") if active.summary else None
            ),
            "recent_messages": [
                item.model_dump(mode="json") for item in active.messages
            ],
            "memory_context": [
                {
                    "memory_id": item.memory_id,
                    "scope": item.scope,
                    "key": item.key,
                    "content": item.content,
                    "instruction": "Background only; never cite as specification evidence.",
                }
                for item in memories
            ],
            "available_research_results": [
                {
                    "result_id": item.result_id,
                    "run_id": item.run_id,
                    "answer": item.answer,
                    "evidence_status": item.evidence_status,
                    "evidence_ids": list(item.evidence_ids),
                    "conditions": list(item.conditions),
                    "content_sha256": item.content_sha256,
                    "requested_scope": item.requested_scope.model_dump(mode="json"),
                    "corpus_versions": [
                        version.model_dump(mode="json")
                        for version in item.corpus_versions
                    ],
                    "unresolved_questions": list(item.unresolved_questions),
                    "prior_results": [
                        dependency.model_dump(mode="json")
                        for dependency in item.prior_results
                    ],
                }
                for item in results
            ],
        }

    def _cancelled(self, turn_id: str) -> bool:
        turn = self.store.turn(turn_id)
        return turn is None or turn.status == "cancelled" or self._stopping.is_set()


__all__ = ["ConversationTurnWorker"]

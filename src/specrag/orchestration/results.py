"""Result materialization and owner-private readback application service."""

from __future__ import annotations

from typing import Protocol

from specrag.access import AccessPolicy
from specrag.evidence.results import ResearchResultMaterializer
from specrag.models import ActorContext, Conversation, ResearchOutcome, ResearchResult, ResearchRun


class OutcomeRunStore(Protocol):
    def get(self, run_id: str) -> ResearchRun | None: ...
    def get_outcome(
        self, run_id: str, run_attempt: int
    ) -> ResearchOutcome | None: ...


class ResultStore(Protocol):
    def put(self, result: ResearchResult) -> ResearchResult: ...
    def get(self, result_id: str) -> ResearchResult | None: ...
    def get_by_run_attempt(
        self, run_id: str, run_attempt: int
    ) -> ResearchResult | None: ...
    def latest_for_run(self, run_id: str) -> ResearchResult | None: ...
    def latest_before_attempt(
        self, run_id: str, run_attempt: int
    ) -> ResearchResult | None: ...
    def list_for_conversation(
        self, conversation_id: str, *, limit: int = 100
    ) -> tuple[ResearchResult, ...]: ...


class ConversationLookup(Protocol):
    def get(self, conversation_id: str) -> Conversation | None: ...


class ResearchResultService:
    def __init__(
        self,
        runs: OutcomeRunStore,
        results: ResultStore,
        conversations: ConversationLookup,
        *,
        materializer: ResearchResultMaterializer | None = None,
        policy: AccessPolicy | None = None,
    ) -> None:
        self.runs = runs
        self.results = results
        self.conversations = conversations
        self.materializer = materializer or ResearchResultMaterializer()
        self.policy = policy or AccessPolicy()

    def materialize(self, run_id: str, run_attempt: int) -> ResearchResult | None:
        existing = self.results.get_by_run_attempt(run_id, run_attempt)
        if existing is not None:
            return existing
        result = self.build(run_id, run_attempt)
        return self.results.put(result) if result is not None else None

    def build(self, run_id: str, run_attempt: int) -> ResearchResult | None:
        """Build a deterministic Result for a caller-owned commit transaction."""

        run = self.runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        if run_attempt > run.run_attempt:
            raise ValueError("Result attempt is newer than the current Run")
        outcome = self.runs.get_outcome(run_id, run_attempt)
        if outcome is None:
            return None
        completed_attempt = run.model_copy(update={
            "run_attempt": run_attempt,
            "status": "completed",
            "stop_reason": outcome.stop_reason,
            "finished_at": outcome.created_at,
            "failure_kind": None,
            "failure_retryable": None,
            "failure_attempt_count": None,
            "failure_status_code": None,
        })
        previous = self.results.latest_before_attempt(run_id, run_attempt)
        result = self.materializer.materialize(
            completed_attempt,
            outcome,
            supersedes_result_id=previous.result_id if previous else None,
        )
        return result

    def get(self, context: ActorContext, result_id: str) -> ResearchResult:
        result = self.results.get(result_id)
        if result is None:
            raise KeyError(result_id)
        self.policy.require(
            context,
            "result.read",
            workspace_id=result.workspace_id,
            owner_user_id=result.owner_user_id,
        )
        return result

    def for_run(self, context: ActorContext, run_id: str) -> ResearchResult | None:
        run = self.runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        self.policy.require(
            context,
            "run.content.read",
            workspace_id=run.workspace_id,
            owner_user_id=run.owner_user_id,
        )
        return self.results.latest_for_run(run_id)

    def list_for_conversation(
        self,
        context: ActorContext,
        conversation_id: str,
        *,
        limit: int = 100,
    ) -> tuple[ResearchResult, ...]:
        conversation = self.conversations.get(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        self.policy.require(
            context,
            "result.read",
            workspace_id=conversation.workspace_id,
            owner_user_id=conversation.owner_user_id,
        )
        return self.results.list_for_conversation(conversation_id, limit=limit)


__all__ = [
    "ConversationLookup",
    "OutcomeRunStore",
    "ResearchResultService",
    "ResultStore",
]

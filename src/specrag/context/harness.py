"""Provider-independent Context Harness service and storage boundary."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from specrag.models import (
    CompactionCheckpoint,
    CompactionSummary,
    CompactionTrigger,
    ContextBudget,
    ContextMessage,
    ContextMessageRole,
    ContextRecord,
    ReconstructedContext,
    StoredToolResult,
    ToolResultPage,
)

from .budget import ConservativeTokenEstimator, TokenBudgetManager, canonical_json

T = TypeVar("T")


class ContextBudgetExceededError(RuntimeError):
    failure_kind = "context_overflow"
    retryable = False
    status_code = None

    def __init__(self, budget: ContextBudget) -> None:
        super().__init__(
            f"estimated input {budget.estimated_input_tokens} exceeds "
            f"{budget.model_id} ceiling {budget.input_ceiling_tokens}"
        )
        self.budget = budget


class CompactionRejectedError(RuntimeError):
    pass


class SummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prior_summary: CompactionSummary | None = None
    messages: tuple[ContextMessage, ...]
    recent_start_message_id: str


class SummaryGenerator(Protocol):
    def summarize(self, request: SummaryRequest) -> CompactionSummary: ...


class ContextStore(Protocol):
    def put_context(self, context: ContextRecord) -> ContextRecord: ...
    def get_context(self, context_id: str) -> ContextRecord | None: ...
    def append_message(self, message: ContextMessage) -> ContextMessage: ...
    def messages(self, context_id: str) -> tuple[ContextMessage, ...]: ...
    def put_tool_result(self, result: StoredToolResult) -> StoredToolResult: ...
    def get_tool_result(self, tool_result_id: str) -> StoredToolResult | None: ...
    def get_tool_result_by_call(
        self, context_id: str, tool_call_id: str,
    ) -> StoredToolResult | None: ...
    def activate_checkpoint(
        self, checkpoint: CompactionCheckpoint,
    ) -> CompactionCheckpoint: ...
    def get_checkpoint(self, checkpoint_id: str) -> CompactionCheckpoint | None: ...


class InMemoryContextStore:
    """Small test/local adapter with the same idempotency as SQLite."""

    def __init__(self) -> None:
        self._contexts: dict[str, ContextRecord] = {}
        self._messages: dict[str, list[ContextMessage]] = {}
        self._results: dict[str, StoredToolResult] = {}
        self._calls: dict[tuple[str, str], str] = {}
        self._checkpoints: dict[str, CompactionCheckpoint] = {}
        self._lock = Lock()

    def put_context(self, context: ContextRecord) -> ContextRecord:
        with self._lock:
            existing = self._contexts.get(context.context_id)
            if existing is not None:
                return existing
            self._contexts[context.context_id] = context
            self._messages[context.context_id] = []
            return context

    def get_context(self, context_id: str) -> ContextRecord | None:
        return self._contexts.get(context_id)

    def append_message(self, message: ContextMessage) -> ContextMessage:
        with self._lock:
            items = self._messages.setdefault(message.context_id, [])
            if any(item.message_id == message.message_id for item in items):
                return next(item for item in items if item.message_id == message.message_id)
            if items and message.sequence != items[-1].sequence + 1:
                raise ValueError("context message sequence is not contiguous")
            if not items and message.sequence != 1:
                raise ValueError("first context message sequence must be one")
            items.append(message)
            return message

    def messages(self, context_id: str) -> tuple[ContextMessage, ...]:
        return tuple(self._messages.get(context_id, ()))

    def put_tool_result(self, result: StoredToolResult) -> StoredToolResult:
        with self._lock:
            key = (result.context_id, result.tool_call_id)
            if key in self._calls:
                existing = self._results[self._calls[key]]
                if existing.content_sha256 != result.content_sha256:
                    raise ValueError("tool call ID was reused with different content")
                return existing
            self._results[result.tool_result_id] = result
            self._calls[key] = result.tool_result_id
            return result

    def get_tool_result(self, tool_result_id: str) -> StoredToolResult | None:
        return self._results.get(tool_result_id)

    def get_tool_result_by_call(
        self, context_id: str, tool_call_id: str,
    ) -> StoredToolResult | None:
        result_id = self._calls.get((context_id, tool_call_id))
        return self._results.get(result_id) if result_id is not None else None

    def activate_checkpoint(
        self, checkpoint: CompactionCheckpoint,
    ) -> CompactionCheckpoint:
        with self._lock:
            context = self._contexts.get(checkpoint.context_id)
            if context is None:
                raise KeyError(f"unknown context: {checkpoint.context_id}")
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
            self._contexts[context.context_id] = context.model_copy(update={
                "active_checkpoint_id": checkpoint.checkpoint_id,
                "updated_at": checkpoint.created_at,
            })
            return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> CompactionCheckpoint | None:
        return self._checkpoints.get(checkpoint_id)


class ContextHarness:
    def __init__(
        self,
        store: ContextStore,
        budgets: TokenBudgetManager,
        *,
        tool_result_soft_limit_tokens: int = 4_000,
        recent_history_tokens: int = 15_000,
        summary_validation_attempts: int = 2,
    ) -> None:
        if (
            tool_result_soft_limit_tokens < 1
            or recent_history_tokens < 1
            or summary_validation_attempts < 1
        ):
            raise ValueError("context limits must be positive")
        self.store = store
        self.budgets = budgets
        self.estimator: ConservativeTokenEstimator = budgets.estimator
        self.tool_result_soft_limit_tokens = tool_result_soft_limit_tokens
        self.recent_history_tokens = recent_history_tokens
        self.summary_validation_attempts = summary_validation_attempts

    def create_context(
        self,
        context_id: str,
        *,
        workspace_id: str | None = None,
        owner_user_id: str | None = None,
        run_id: str | None = None,
    ) -> ContextRecord:
        now = datetime.now(timezone.utc)
        return self.store.put_context(ContextRecord(
            context_id=context_id,
            workspace_id=workspace_id,
            owner_user_id=owner_user_id,
            run_id=run_id,
            created_at=now,
            updated_at=now,
        ))

    def preflight(
        self,
        model_id: str,
        components: dict[str, Any],
        *,
        expected_output_tokens: int | None = None,
    ) -> ContextBudget:
        budget = self.budgets.inspect(
            model_id, components, expected_output_tokens=expected_output_tokens,
        )
        if not budget.fits:
            raise ContextBudgetExceededError(budget)
        return budget

    def append_message(
        self,
        context_id: str,
        role: ContextMessageRole,
        content: Any,
        *,
        turn_id: str | None = None,
        message_id: str | None = None,
    ) -> ContextMessage:
        if self.store.get_context(context_id) is None:
            self.create_context(context_id)
        sequence = len(self.store.messages(context_id)) + 1
        message = ContextMessage(
            message_id=message_id or "MSG-" + uuid4().hex.upper(),
            context_id=context_id,
            sequence=sequence,
            role=role,
            content=content,
            token_estimate=self.estimator.estimate(content),
            created_at=datetime.now(timezone.utc),
            turn_id=turn_id,
        )
        return self.store.append_message(message)

    def record_tool_result(
        self,
        context_id: str,
        tool_call_id: str,
        content: Any,
        *,
        max_tokens: int | None = None,
    ) -> ToolResultPage:
        if self.store.get_context(context_id) is None:
            self.create_context(context_id)
        serialized = canonical_json(content)
        import hashlib

        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        existing = self.store.get_tool_result_by_call(context_id, tool_call_id)
        if existing is not None:
            if existing.content_sha256 != digest:
                raise ValueError("tool call ID was reused with different content")
            return self._page(existing, None, max_tokens or self.tool_result_soft_limit_tokens)
        result = self.store.put_tool_result(StoredToolResult(
            tool_result_id="TOOLRESULT-" + uuid4().hex.upper(),
            context_id=context_id,
            tool_call_id=tool_call_id,
            content_sha256=digest,
            content=content,
            token_estimate=self.estimator.estimate_text(serialized),
            created_at=datetime.now(timezone.utc),
        ))
        return self._page(result, None, max_tokens or self.tool_result_soft_limit_tokens)

    def read_tool_result(
        self,
        tool_result_id: str,
        *,
        cursor: str | None = None,
        max_tokens: int | None = None,
    ) -> ToolResultPage:
        result = self.store.get_tool_result(tool_result_id)
        if result is None:
            raise KeyError(f"unknown tool result: {tool_result_id}")
        return self._page(result, cursor, max_tokens or self.tool_result_soft_limit_tokens)

    def _page(
        self, result: StoredToolResult, cursor: str | None, max_tokens: int,
    ) -> ToolResultPage:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        serialized = canonical_json(result.content)
        start = 0
        if cursor is not None:
            if not cursor.startswith("char:"):
                raise ValueError("invalid tool-result cursor")
            try:
                start = int(cursor[5:])
            except ValueError as exc:
                raise ValueError("invalid tool-result cursor") from exc
            if not 0 <= start < len(serialized):
                raise ValueError("tool-result cursor is out of range")
        if start == 0 and result.token_estimate <= max_tokens:
            return ToolResultPage(
                tool_result_id=result.tool_result_id,
                content_sha256=result.content_sha256,
                content=result.content,
                truncated=False,
                token_estimate=result.token_estimate,
            )
        low, high = start + 1, len(serialized)
        best = low
        while low <= high:
            middle = (low + high) // 2
            if self.estimator.estimate_text(serialized[start:middle]) <= max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        fragment = serialized[start:best]
        next_cursor = f"char:{best}" if best < len(serialized) else None
        return ToolResultPage(
            tool_result_id=result.tool_result_id,
            content_sha256=result.content_sha256,
            content={"json_fragment": fragment, "character_offset": start},
            truncated=next_cursor is not None or start > 0,
            next_cursor=next_cursor,
            token_estimate=self.estimator.estimate_text(fragment),
        )

    def reconstruct(self, context_id: str) -> ReconstructedContext:
        context = self.store.get_context(context_id)
        if context is None:
            raise KeyError(f"unknown context: {context_id}")
        messages = self.store.messages(context_id)
        if context.active_checkpoint_id is None:
            return ReconstructedContext(context_id=context_id, messages=messages)
        checkpoint = self.store.get_checkpoint(context.active_checkpoint_id)
        if checkpoint is None:
            raise RuntimeError("active context checkpoint is missing")
        start = next((
            index for index, item in enumerate(messages)
            if item.message_id == checkpoint.recent_start_message_id
        ), None)
        if start is None:
            raise RuntimeError("checkpoint recent message boundary is missing")
        return ReconstructedContext(
            context_id=context_id,
            summary=checkpoint.summary,
            messages=messages[start:],
        )

    def compact(
        self,
        context_id: str,
        generator: SummaryGenerator,
        *,
        trigger: CompactionTrigger,
        model_id: str | None = None,
        recent_history_tokens: int | None = None,
    ) -> CompactionCheckpoint:
        active = self.reconstruct(context_id)
        messages = active.messages
        keep_budget = recent_history_tokens or self.recent_history_tokens
        start = self._recent_start(messages, keep_budget)
        if start <= 0:
            raise CompactionRejectedError("no older message history is available")
        recent = messages[start:]
        source_text = canonical_json({
            "prior_summary": active.summary,
            "messages": [item.model_dump(mode="json") for item in messages],
        })
        request = SummaryRequest(
            prior_summary=active.summary,
            messages=messages,
            recent_start_message_id=recent[0].message_id,
        )
        missing: tuple[str, ...] = ()
        summary: CompactionSummary | None = None
        generation_error: Exception | None = None
        for _ in range(self.summary_validation_attempts):
            try:
                summary = generator.summarize(request)
            except Exception as exc:
                if (
                    not isinstance(exc, ValueError)
                    and getattr(exc, "failure_kind", None) != "structured_output"
                ):
                    raise
                generation_error = exc
                continue
            missing = tuple(ref for ref in summary.references if ref not in source_text)
            if not missing:
                break
        else:
            if missing:
                reason = "summary contains references absent from source history: " + ", ".join(missing)
            else:
                reason = "summary generation failed structural validation"
            raise CompactionRejectedError(reason) from generation_error
        assert summary is not None
        before = self.estimator.estimate({
            "summary": active.summary,
            "messages": [item.content for item in messages],
        })
        after = self.estimator.estimate({
            "summary": summary,
            "messages": [item.content for item in recent],
        })
        if after >= before:
            raise CompactionRejectedError("summary does not reduce active context")
        checkpoint = CompactionCheckpoint(
            checkpoint_id="CHECKPOINT-" + uuid4().hex.upper(),
            context_id=context_id,
            summary=summary,
            covers_through_message_id=messages[start - 1].message_id,
            recent_start_message_id=recent[0].message_id,
            trigger=trigger,
            model_id=model_id,
            before_tokens=before,
            after_tokens=after,
            created_at=datetime.now(timezone.utc),
        )
        return self.store.activate_checkpoint(checkpoint)

    @staticmethod
    def _recent_start(messages: tuple[ContextMessage, ...], budget: int) -> int:
        used = 0
        start = len(messages)
        while start > 0:
            turn_id = messages[start - 1].turn_id
            group_start = start - 1
            if turn_id is not None:
                while group_start > 0 and messages[group_start - 1].turn_id == turn_id:
                    group_start -= 1
            group_tokens = sum(item.token_estimate for item in messages[group_start:start])
            if used and used + group_tokens > budget:
                break
            if not used and group_tokens > budget:
                # An oversized newest turn remains atomic even when over budget.
                return group_start
            used += group_tokens
            start = group_start
        return start

    def prepare(
        self,
        context_id: str,
        model_id: str,
        components: dict[str, Any],
        *,
        generator: SummaryGenerator | None = None,
        manual: bool = False,
        expected_output_tokens: int | None = None,
    ) -> tuple[ReconstructedContext, ContextBudget]:
        active = self.reconstruct(context_id)
        request_components = {**components, "message_history": active.model_dump(mode="json")}
        budget = self.budgets.inspect(
            model_id, request_components, expected_output_tokens=expected_output_tokens,
        )
        if (manual or not budget.fits) and generator is not None:
            self.compact(
                context_id, generator, trigger="manual" if manual else "auto",
                model_id=model_id,
            )
            active = self.reconstruct(context_id)
            request_components["message_history"] = active.model_dump(mode="json")
            budget = self.budgets.inspect(
                model_id, request_components,
                expected_output_tokens=expected_output_tokens,
            )
        if not budget.fits:
            raise ContextBudgetExceededError(budget)
        return active, budget

    def call_with_overflow_recovery(
        self,
        context_id: str,
        call: Callable[[], T],
        generator: SummaryGenerator,
        *,
        model_id: str | None = None,
    ) -> T:
        """Retry the same LLM-only call once; tool execution stays outside."""

        try:
            return call()
        except Exception as exc:
            if getattr(exc, "failure_kind", None) != "context_overflow":
                raise
        self.compact(context_id, generator, trigger="overflow", model_id=model_id)
        return call()


__all__ = [
    "CompactionRejectedError", "ContextBudgetExceededError", "ContextHarness",
    "ContextStore", "InMemoryContextStore", "SummaryGenerator", "SummaryRequest",
]

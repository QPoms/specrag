"""LangGraph runtime for the bounded Single Researcher workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from specrag.evidence import partial_synthesis_eligible
from specrag.models import (
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchBudget,
    ResearchGap,
    ResearchPlan,
    ResearchRouteDecision,
    ResearchTask,
    ResearchTaskExecution,
    ResearcherEvidenceSelectionRecord,
    ResearchWorkerResult,
    SourceValidationResult,
    TaskSufficiencyAssessment,
    ToolCallRecord,
    ValidatedEvidence,
)

from .state import ResearchState


StateUpdate = Mapping[str, object]
NodeHandler = Callable[[ResearchState], StateUpdate]
RuntimeGuard = Callable[[ResearchState], None]
EventSink = Callable[[str, ResearchState, StateUpdate, float], None]
ReceiptCapture = Callable[[], AbstractContextManager[list[Any]]]


def _receipt_payload(receipt: Any, stage: str) -> dict[str, object]:
    if hasattr(receipt, "model_dump"):
        value = receipt.model_dump(mode="json")
    elif isinstance(receipt, Mapping):
        value = dict(receipt)
    else:
        raise TypeError("model receipt must be a mapping or Pydantic model")
    return {**value, "stage": stage}


def _merge_model_receipts(
    state: ResearchState,
    update: dict[str, object],
    captured: list[Any],
    *,
    stage: str,
) -> None:
    existing = list(update.get("model_receipts", state["model_receipts"]))
    handler_added = existing[len(state["model_receipts"]) :]
    duplicate_counts: dict[tuple[object, object, object], int] = {}
    for item in handler_added:
        if not isinstance(item, Mapping):
            continue
        key = (
            item.get("request_sha256"),
            item.get("response_sha256"),
            item.get("purpose"),
        )
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
    for receipt in captured:
        payload = _receipt_payload(receipt, stage)
        key = (
            payload.get("request_sha256"),
            payload.get("response_sha256"),
            payload.get("purpose"),
        )
        duplicates = duplicate_counts.get(key, 0)
        if duplicates:
            duplicate_counts[key] = duplicates - 1
            continue
        existing.append(payload)
    if existing != list(state["model_receipts"]):
        update["model_receipts"] = tuple(existing)


_CHECKPOINT_MODEL_ALLOWLIST = (
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchBudget,
    ResearchGap,
    ResearchPlan,
    ResearchRouteDecision,
    ResearchTask,
    ResearchTaskExecution,
    ResearcherEvidenceSelectionRecord,
    ResearchWorkerResult,
    SourceValidationResult,
    TaskSufficiencyAssessment,
    ToolCallRecord,
    ValidatedEvidence,
)

_PARTIAL_SYNTHESIS_STOP_REASONS = frozenset(
    {
        "iteration_budget",
        "tool_budget",
        "replan_budget",
        "research_time_budget",
        "insufficient_evidence",
    }
)


def _in_memory_checkpointer() -> InMemorySaver:
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=_CHECKPOINT_MODEL_ALLOWLIST,
    )
    return InMemorySaver(serde=serde)


@dataclass(frozen=True)
class ResearchNodeHandlers:
    """Framework-independent business handlers called by graph nodes."""

    router: NodeHandler
    planner: NodeHandler
    researcher: NodeHandler
    source_validator: NodeHandler
    sufficiency_judge: NodeHandler
    replanner: NodeHandler
    synthesizer: NodeHandler


def _node(
    name: str,
    handler: NodeHandler,
    guard: RuntimeGuard | None = None,
    event_sink: EventSink | None = None,
    receipt_capture: ReceiptCapture | None = None,
) -> Callable[[ResearchState], dict[str, object]]:
    def invoke(state: ResearchState) -> dict[str, object]:
        from time import perf_counter

        if guard is not None:
            guard(state)
        started = perf_counter()
        capture = receipt_capture() if receipt_capture is not None else nullcontext([])
        try:
            with capture as receipts:
                update = dict(handler(state))
        except Exception:
            update = {}
            _merge_model_receipts(state, update, receipts, stage=name)
            if event_sink is not None:
                event_sink(name, state, update, (perf_counter() - started) * 1000)
            raise
        _merge_model_receipts(state, update, receipts, stage=name)
        if event_sink is not None:
            event_sink(name, state, update, (perf_counter() - started) * 1000)
        return update

    return invoke


def _researcher_node(
    handler: NodeHandler,
    guard: RuntimeGuard | None = None,
    event_sink: EventSink | None = None,
    receipt_capture: ReceiptCapture | None = None,
) -> Callable[[ResearchState], dict[str, object]]:
    def invoke(state: ResearchState) -> dict[str, object]:
        from time import perf_counter

        if guard is not None:
            guard(state)
        started = perf_counter()
        capture = receipt_capture() if receipt_capture is not None else nullcontext([])
        with capture as receipts:
            update = dict(handler(state))
        _merge_model_receipts(state, update, receipts, stage="researcher")
        update.setdefault("iteration", state["iteration"] + 1)
        if event_sink is not None:
            event_sink("researcher", state, update, (perf_counter() - started) * 1000)
        return update

    return invoke


def _replanner_node(
    handler: NodeHandler,
    guard: RuntimeGuard | None = None,
    event_sink: EventSink | None = None,
    receipt_capture: ReceiptCapture | None = None,
) -> Callable[[ResearchState], dict[str, object]]:
    def invoke(state: ResearchState) -> dict[str, object]:
        from time import perf_counter

        if guard is not None:
            guard(state)
        started = perf_counter()
        capture = receipt_capture() if receipt_capture is not None else nullcontext([])
        with capture as receipts:
            update = dict(handler(state))
        _merge_model_receipts(state, update, receipts, stage="replanner")
        budget = cast(ResearchBudget, update.get("budget", state["budget"]))
        if "budget" not in update:
            update["budget"] = budget.model_copy(
                update={
                    "replans_used": budget.replans_used + 1,
                }
            )
        update.setdefault("evidence_sufficient", None)
        if event_sink is not None:
            event_sink("replanner", state, update, (perf_counter() - started) * 1000)
        return update

    return invoke


def _has_incomplete_attempted_task(state: ResearchState) -> bool:
    return any(
        task.status in {"running", "blocked"} for task in state["research_tasks"]
    )


def _has_ready_pending_task(state: ResearchState) -> bool:
    completed = {
        task.task_id for task in state["research_tasks"] if task.status == "completed"
    }
    return any(
        task.status == "pending" and set(task.dependencies).issubset(completed)
        for task in state["research_tasks"]
    )


def _can_advance_without_replan(state: ResearchState) -> bool:
    return not _has_incomplete_attempted_task(state) and _has_ready_pending_task(state)


def _has_judge_gap(state: ResearchState, task_id: str) -> bool:
    """Return whether a task has an unresolved downstream Judge gap."""

    return any(gap.task_id == task_id and not gap.resolved for gap in state["gaps"])


def _budget_guard(
    state: ResearchState,
    *,
    research_deadline_at: datetime | None = None,
) -> dict[str, object]:
    if state["evidence_sufficient"] is True or state["stop_reason"] is not None:
        return {}
    if (
        research_deadline_at is not None
        and datetime.now(timezone.utc) >= research_deadline_at
    ):
        return {"stop_reason": "research_time_budget"}
    budget = state["budget"]
    if state["iteration"] >= budget.max_iterations:
        return {"stop_reason": "iteration_budget"}
    completed = {
        task.task_id for task in state["research_tasks"] if task.status == "completed"
    }
    ready = (
        task
        for task in state["research_tasks"]
        if task.status in {"pending", "running"}
        and set(task.dependencies).issubset(completed)
    )
    if not any(
        budget.remaining_tool_calls_for(
            task.task_id,
            corrective=_has_judge_gap(state, task.task_id),
        )
        > 0
        for task in ready
    ):
        return {"stop_reason": "tool_budget"}
    if (
        not _can_advance_without_replan(state)
        and budget.replans_used >= budget.max_replans
    ):
        return {"stop_reason": "replan_budget"}
    return {}


def _route_after_guard(
    state: ResearchState,
) -> Literal["researcher", "replanner", "synthesizer", "__end__"]:
    if state["evidence_sufficient"] is True:
        return "synthesizer"
    if state["stop_reason"] is not None:
        if state[
            "stop_reason"
        ] in _PARTIAL_SYNTHESIS_STOP_REASONS and partial_synthesis_eligible(
            state["validated_evidence"],
            state["sufficiency_assessments"],
        ):
            return "synthesizer"
        return END
    if _can_advance_without_replan(state):
        return "researcher"
    return "replanner"


@dataclass(frozen=True)
class SingleResearchWorkflow:
    """Compiled graph plus thread-scoped checkpoint and recovery helpers."""

    graph: Any

    @staticmethod
    def _config(thread_id: str) -> dict[str, dict[str, str]]:
        if not thread_id.strip():
            raise ValueError("thread_id must not be blank")
        return {"configurable": {"thread_id": thread_id}}

    def invoke(self, state: ResearchState, *, thread_id: str) -> ResearchState:
        result = self.graph.invoke(state, config=self._config(thread_id))
        return cast(ResearchState, result)

    def resume(
        self,
        *,
        thread_id: str,
        update: StateUpdate | None = None,
    ) -> ResearchState:
        config = self._config(thread_id)
        if update:
            self.graph.update_state(config, dict(update))
        result = self.graph.invoke(None, config=config)
        return cast(ResearchState, result)

    def checkpoint(self, *, thread_id: str) -> Any:
        return self.graph.get_state(self._config(thread_id))

    def update(self, *, thread_id: str, update: StateUpdate) -> None:
        self.graph.update_state(self._config(thread_id), dict(update))


def build_single_research_workflow(
    handlers: ResearchNodeHandlers,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    runtime_guard: RuntimeGuard | None = None,
    event_sink: EventSink | None = None,
    receipt_capture: ReceiptCapture | None = None,
    research_deadline_at: datetime | None = None,
) -> SingleResearchWorkflow:
    """Compile the explicit Single Researcher graph with checkpointing."""

    builder = StateGraph(ResearchState)
    builder.add_node(
        "router",
        _node("router", handlers.router, runtime_guard, event_sink, receipt_capture),
    )
    builder.add_node(
        "planner",
        _node("planner", handlers.planner, runtime_guard, event_sink, receipt_capture),
    )
    builder.add_node(
        "researcher",
        _researcher_node(
            handlers.researcher, runtime_guard, event_sink, receipt_capture
        ),
    )
    builder.add_node(
        "source_validator",
        _node(
            "source_validator",
            handlers.source_validator,
            runtime_guard,
            event_sink,
            receipt_capture,
        ),
    )
    builder.add_node(
        "sufficiency_judge",
        _node(
            "sufficiency_judge",
            handlers.sufficiency_judge,
            runtime_guard,
            event_sink,
            receipt_capture,
        ),
    )
    builder.add_node(
        "budget_guard",
        lambda state: _budget_guard(
            state,
            research_deadline_at=research_deadline_at,
        ),
    )
    builder.add_node(
        "replanner",
        _replanner_node(handlers.replanner, runtime_guard, event_sink, receipt_capture),
    )
    builder.add_node(
        "synthesizer",
        _node(
            "synthesizer",
            handlers.synthesizer,
            runtime_guard,
            event_sink,
            receipt_capture,
        ),
    )

    builder.add_edge(START, "router")
    builder.add_edge("router", "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "source_validator")
    builder.add_edge("source_validator", "sufficiency_judge")
    builder.add_edge("sufficiency_judge", "budget_guard")
    builder.add_conditional_edges("budget_guard", _route_after_guard)
    builder.add_edge("replanner", "researcher")
    builder.add_edge("synthesizer", END)

    compiled = builder.compile(checkpointer=checkpointer or _in_memory_checkpointer())
    return SingleResearchWorkflow(graph=compiled)


__all__ = [
    "NodeHandler",
    "ResearchNodeHandlers",
    "SingleResearchWorkflow",
    "StateUpdate",
    "build_single_research_workflow",
]

"""LangGraph runtime for the bounded Single Researcher workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from specrag.models import (
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchBudget,
    ResearchGap,
    ResearchPlan,
    ResearchRouteDecision,
    ResearchTask,
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


_CHECKPOINT_MODEL_ALLOWLIST = (
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchBudget,
    ResearchGap,
    ResearchPlan,
    ResearchRouteDecision,
    ResearchTask,
    ResearcherEvidenceSelectionRecord,
    ResearchWorkerResult,
    SourceValidationResult,
    TaskSufficiencyAssessment,
    ToolCallRecord,
    ValidatedEvidence,
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
) -> Callable[[ResearchState], dict[str, object]]:
    def invoke(state: ResearchState) -> dict[str, object]:
        from time import perf_counter
        if guard is not None:
            guard(state)
        started = perf_counter()
        update = dict(handler(state))
        if event_sink is not None:
            event_sink(name, state, update, (perf_counter() - started) * 1000)
        return update

    return invoke


def _researcher_node(
    handler: NodeHandler,
    guard: RuntimeGuard | None = None,
    event_sink: EventSink | None = None,
) -> Callable[[ResearchState], dict[str, object]]:
    def invoke(state: ResearchState) -> dict[str, object]:
        from time import perf_counter
        if guard is not None:
            guard(state)
        started = perf_counter()
        update = dict(handler(state))
        update.setdefault("iteration", state["iteration"] + 1)
        if event_sink is not None:
            event_sink("researcher", state, update, (perf_counter() - started) * 1000)
        return update

    return invoke


def _replanner_node(
    handler: NodeHandler,
    guard: RuntimeGuard | None = None,
    event_sink: EventSink | None = None,
) -> Callable[[ResearchState], dict[str, object]]:
    def invoke(state: ResearchState) -> dict[str, object]:
        from time import perf_counter
        if guard is not None:
            guard(state)
        started = perf_counter()
        update = dict(handler(state))
        budget = cast(ResearchBudget, update.get("budget", state["budget"]))
        if "budget" not in update:
            update["budget"] = budget.model_copy(update={
                "replans_used": budget.replans_used + 1,
            })
        update.setdefault("evidence_sufficient", None)
        if event_sink is not None:
            event_sink("replanner", state, update, (perf_counter() - started) * 1000)
        return update

    return invoke


def _has_incomplete_attempted_task(state: ResearchState) -> bool:
    return any(
        task.status in {"running", "blocked"}
        for task in state["research_tasks"]
    )


def _has_ready_pending_task(state: ResearchState) -> bool:
    completed = {
        task.task_id
        for task in state["research_tasks"]
        if task.status == "completed"
    }
    return any(
        task.status == "pending"
        and set(task.dependencies).issubset(completed)
        for task in state["research_tasks"]
    )


def _can_advance_without_replan(state: ResearchState) -> bool:
    return (
        not _has_incomplete_attempted_task(state)
        and _has_ready_pending_task(state)
    )


def _has_judge_gap(state: ResearchState, task_id: str) -> bool:
    """Return whether a task has an unresolved downstream Judge gap."""

    return any(
        gap.task_id == task_id and not gap.resolved
        for gap in state["gaps"]
    )


def _budget_guard(state: ResearchState) -> dict[str, object]:
    if state["evidence_sufficient"] is True or state["stop_reason"] is not None:
        return {}
    budget = state["budget"]
    if state["iteration"] >= budget.max_iterations:
        return {"stop_reason": "iteration_budget"}
    completed = {
        task.task_id
        for task in state["research_tasks"]
        if task.status == "completed"
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
        ) > 0
        for task in ready
    ):
        return {"stop_reason": "tool_budget"}
    if (
        not _can_advance_without_replan(state)
        and budget.replans_used >= budget.max_replans
    ):
        return {"stop_reason": "replan_budget"}
    return {}


def _route_after_router(state: ResearchState) -> Literal["planner", "__end__"]:
    if state["route"] == "finish" or state["stop_reason"] is not None:
        return END
    return "planner"


def _route_after_guard(
    state: ResearchState,
) -> Literal["researcher", "replanner", "synthesizer", "__end__"]:
    if state["evidence_sufficient"] is True:
        return "synthesizer"
    if state["stop_reason"] is not None:
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
) -> SingleResearchWorkflow:
    """Compile the explicit Single Researcher graph with checkpointing."""

    builder = StateGraph(ResearchState)
    builder.add_node("router", _node("router", handlers.router, runtime_guard, event_sink))
    builder.add_node("planner", _node("planner", handlers.planner, runtime_guard, event_sink))
    builder.add_node("researcher", _researcher_node(handlers.researcher, runtime_guard, event_sink))
    builder.add_node(
        "source_validator",
        _node("source_validator", handlers.source_validator, runtime_guard, event_sink),
    )
    builder.add_node("sufficiency_judge", _node("sufficiency_judge", handlers.sufficiency_judge, runtime_guard, event_sink))
    builder.add_node("budget_guard", _budget_guard)
    builder.add_node("replanner", _replanner_node(handlers.replanner, runtime_guard, event_sink))
    builder.add_node("synthesizer", _node("synthesizer", handlers.synthesizer, runtime_guard, event_sink))

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", _route_after_router)
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "source_validator")
    builder.add_edge("source_validator", "sufficiency_judge")
    builder.add_edge("sufficiency_judge", "budget_guard")
    builder.add_conditional_edges("budget_guard", _route_after_guard)
    builder.add_edge("replanner", "researcher")
    builder.add_edge("synthesizer", END)

    compiled = builder.compile(
        checkpointer=checkpointer or _in_memory_checkpointer()
    )
    return SingleResearchWorkflow(graph=compiled)


__all__ = [
    "NodeHandler",
    "ResearchNodeHandlers",
    "SingleResearchWorkflow",
    "StateUpdate",
    "build_single_research_workflow",
]

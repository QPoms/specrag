"""LangGraph fan-out/fan-in workflow for independent Researcher tasks."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from specrag.agents.researcher import RegistryResearcherService
from specrag.models import ResearchBudget, ResearchTask, ResearchWorkerResult

from .adapters import researcher_execution_update
from .dependencies import build_dependency_context
from .gap_context import build_corrective_research_input
from .state import ResearchState, new_research_state
from .workflow import (
    ResearchNodeHandlers,
    StateUpdate,
    _budget_guard,
    _can_advance_without_replan,
    _has_judge_gap,
    _in_memory_checkpointer,
    _node,
    _replanner_node,
)


class ParallelResearchState(ResearchState):
    """Unified research state with an append-only worker result channel."""

    researcher_results: Annotated[
        tuple[ResearchWorkerResult, ...], operator.add
    ]


class WorkerInput(TypedDict):
    state: ParallelResearchState
    task: ResearchTask
    max_tool_calls: int


@dataclass(frozen=True)
class ParallelResearcherHandler:
    """Run one isolated worker assignment for a LangGraph Send packet."""

    service: RegistryResearcherService

    def __call__(
        self,
        state: ParallelResearchState,
        task: ResearchTask,
        *,
        max_tool_calls: int,
    ) -> ResearchWorkerResult:
        running = task.model_copy(update={"status": "running"})
        corrective = _has_judge_gap(state, task.task_id)
        shared_context: dict[str, object] = {
            "research_id": state["research_id"],
            "iteration": state["iteration"],
            "gap_retrieval_mode": "initial",
            "unresolved_gaps": (),
            "worker_task_id": task.task_id,
            "dependency_context": build_dependency_context(state, task),
        }
        corrective_options: dict[str, object] = {}
        if corrective:
            corrective_input = build_corrective_research_input(state, task)
            shared_context.update(corrective_input.shared_context)
            corrective_options = {
                "initial_evidence": corrective_input.initial_evidence,
                "initial_active_evidence_ids": (
                    corrective_input.initial_active_evidence_ids
                ),
            }
        execution, query_to_task = self.service.run(
            (running,),
            shared_context=shared_context,
            max_tool_calls=max_tool_calls,
            **corrective_options,
        )
        update = researcher_execution_update(
            cast(ResearchState, state),
            execution,
            query_to_task=query_to_task,
            task_id=task.task_id,
            corrective=corrective,
        )
        old_candidates = {
            evidence.evidence_id for evidence in state["candidate_evidence"]
        }
        old_calls = {call.call_id for call in state["tool_calls"]}
        old_selections = {
            item.selection_id
            for item in state["researcher_evidence_selections"]
        }
        return ResearchWorkerResult(
            task_id=task.task_id,
            candidate_evidence=tuple(
                evidence
                for evidence in update["candidate_evidence"]
                if evidence.evidence_id not in old_candidates
            ),
            tool_calls=tuple(
                call
                for call in update["tool_calls"]
                if call.call_id not in old_calls
            ),
            evidence_selections=tuple(
                item
                for item in update["researcher_evidence_selections"]
                if item.selection_id not in old_selections
            ),
            tool_calls_used=(
                execution.search_actions_used
                + execution.expansion_actions_used
            ),
            stop_reason=execution.stop_reason,
            errors=execution.errors,
        )


@dataclass(frozen=True)
class ParallelResearchWorkflow:
    """Compiled dynamic fan-out/fan-in graph with checkpoint recovery."""

    graph: Any

    @staticmethod
    def _config(thread_id: str) -> dict[str, dict[str, str]]:
        if not thread_id.strip():
            raise ValueError("thread_id must not be blank")
        return {"configurable": {"thread_id": thread_id}}

    def invoke(
        self,
        state: ParallelResearchState,
        *,
        thread_id: str,
    ) -> ParallelResearchState:
        return cast(
            ParallelResearchState,
            self.graph.invoke(state, config=self._config(thread_id)),
        )

    def resume(
        self,
        *,
        thread_id: str,
        update: StateUpdate | None = None,
    ) -> ParallelResearchState:
        config = self._config(thread_id)
        if update:
            self.graph.update_state(config, dict(update))
        return cast(ParallelResearchState, self.graph.invoke(None, config=config))

    def checkpoint(self, *, thread_id: str) -> Any:
        return self.graph.get_state(self._config(thread_id))


def new_parallel_research_state(
    *,
    research_id: str,
    user_question: str,
    budget: ResearchBudget | None = None,
) -> ParallelResearchState:
    return cast(
        ParallelResearchState,
        new_research_state(
            research_id=research_id,
            user_question=user_question,
            budget=budget,
        ),
    )


def _ready_tasks(state: ParallelResearchState) -> tuple[ResearchTask, ...]:
    completed = {
        task.task_id
        for task in state["research_tasks"]
        if task.status == "completed"
    }
    return tuple(
        task
        for task in state["research_tasks"]
        if task.status in {"pending", "running"}
        and set(task.dependencies).issubset(completed)
    )


def _dispatch_workers(state: ParallelResearchState) -> list[Send] | str:
    budget = state["budget"]
    tasks = tuple(
        task
        for task in _ready_tasks(state)
        if budget.remaining_tool_calls_for(
            task.task_id,
            corrective=_has_judge_gap(state, task.task_id),
        ) > 0
    )
    if not tasks:
        return "budget_guard"
    return [
        Send(
            "researcher_worker",
            {
                "state": state,
                "task": task,
                "max_tool_calls": budget.remaining_tool_calls_for(
                    task.task_id,
                    corrective=_has_judge_gap(state, task.task_id),
                ),
            },
        )
        for task in tasks
    ]


def _worker_node(
    handler: ParallelResearcherHandler,
) -> Callable[[WorkerInput], dict[str, object]]:
    def invoke(packet: WorkerInput) -> dict[str, object]:
        result = handler(
            packet["state"],
            packet["task"],
            max_tool_calls=packet["max_tool_calls"],
        )
        return {"researcher_results": (result,)}

    return invoke


def _fan_in(state: ParallelResearchState) -> dict[str, object]:
    existing_candidates = list(state["candidate_evidence"])
    candidate_ids = {item.evidence_id for item in existing_candidates}
    existing_calls = list(state["tool_calls"])
    call_ids = {item.call_id for item in existing_calls}
    existing_selections = list(state["researcher_evidence_selections"])
    selection_ids = {item.selection_id for item in existing_selections}
    tasks = list(state["research_tasks"])
    task_by_id = {task.task_id: task for task in tasks}
    errors = list(state["errors"])
    budget = state["budget"]
    new_results = state["researcher_results"][
        state["researcher_results_processed"]:
    ]
    for result in new_results:
        for candidate in result.candidate_evidence:
            if candidate.evidence_id not in candidate_ids:
                existing_candidates.append(candidate)
                candidate_ids.add(candidate.evidence_id)
        for call in result.tool_calls:
            if call.call_id not in call_ids:
                existing_calls.append(call)
                call_ids.add(call.call_id)
        for selection in result.evidence_selections:
            if selection.selection_id not in selection_ids:
                existing_selections.append(selection)
                selection_ids.add(selection.selection_id)
        budget = budget.record_tool_calls(
            result.task_id,
            result.tool_calls_used,
            corrective=_has_judge_gap(state, result.task_id),
        )
        errors.extend(result.errors)
        task = task_by_id[result.task_id]
        task_by_id[result.task_id] = task.model_copy(update={"status": "running"})
    return {
        "candidate_evidence": tuple(existing_candidates),
        "tool_calls": tuple(existing_calls),
        "researcher_evidence_selections": tuple(existing_selections),
        "research_tasks": tuple(task_by_id[task.task_id] for task in tasks),
        "budget": budget,
        "iteration": state["iteration"] + 1,
        "errors": tuple(errors),
        "researcher_results_processed": len(state["researcher_results"]),
    }


def _route_after_parallel_guard(
    state: ParallelResearchState,
) -> str:
    if state["evidence_sufficient"] is True:
        return "synthesizer"
    if state["stop_reason"] is not None:
        return END
    if _can_advance_without_replan(state):
        return "dispatch"
    return "replanner"


def build_parallel_research_workflow(
    handlers: ResearchNodeHandlers,
    researcher: ParallelResearcherHandler,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> ParallelResearchWorkflow:
    """Compile Router → Planner → Send workers → fan-in → evidence nodes."""

    builder = StateGraph(ParallelResearchState)
    builder.add_node("router", _node("router", handlers.router))
    builder.add_node("planner", _node("planner", handlers.planner))
    builder.add_node("dispatch", lambda state: {})
    builder.add_node("researcher_worker", _worker_node(researcher))
    builder.add_node("fan_in", _fan_in)
    builder.add_node(
        "source_validator",
        _node("source_validator", handlers.source_validator),
    )
    builder.add_node(
        "sufficiency_judge", _node("sufficiency_judge", handlers.sufficiency_judge)
    )
    builder.add_node("budget_guard", _budget_guard)
    builder.add_node("replanner", _replanner_node(handlers.replanner))
    builder.add_node("synthesizer", _node("synthesizer", handlers.synthesizer))

    builder.add_edge(START, "router")
    builder.add_edge("router", "planner")
    builder.add_edge("planner", "dispatch")
    builder.add_conditional_edges(
        "dispatch",
        _dispatch_workers,
        ["researcher_worker", "budget_guard"],
    )
    builder.add_edge("researcher_worker", "fan_in")
    builder.add_edge("fan_in", "source_validator")
    builder.add_edge("source_validator", "sufficiency_judge")
    builder.add_edge("sufficiency_judge", "budget_guard")
    builder.add_conditional_edges(
        "budget_guard",
        _route_after_parallel_guard,
        ["dispatch", "replanner", "synthesizer", END],
    )
    builder.add_edge("replanner", "dispatch")
    builder.add_edge("synthesizer", END)

    compiled = builder.compile(
        checkpointer=checkpointer or _in_memory_checkpointer()
    )
    return ParallelResearchWorkflow(graph=compiled)


__all__ = [
    "ParallelResearchState",
    "ParallelResearcherHandler",
    "ParallelResearchWorkflow",
    "build_parallel_research_workflow",
    "new_parallel_research_state",
]

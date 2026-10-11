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
from .state import ResearchState, new_research_state
from .workflow import (
    ResearchNodeHandlers,
    StateUpdate,
    _budget_guard,
    _in_memory_checkpointer,
    _node,
    _replanner_node,
    _route_after_guard,
    _route_after_router,
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
        relevant_gaps = tuple(
            gap.model_dump(mode="json")
            for gap in state["gaps"]
            if gap.task_id == task.task_id and not gap.resolved
        )
        execution, query_to_task = self.service.run(
            (running,),
            shared_context={
                "research_id": state["research_id"],
                "iteration": state["iteration"],
                "unresolved_gaps": relevant_gaps,
                "worker_task_id": task.task_id,
                "dependency_context": build_dependency_context(state, task),
            },
            max_tool_calls=max_tool_calls,
        )
        update = researcher_execution_update(
            cast(ResearchState, state),
            execution,
            query_to_task=query_to_task,
        )
        old_candidates = {
            evidence.evidence_id for evidence in state["candidate_evidence"]
        }
        old_calls = {call.call_id for call in state["tool_calls"]}
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
    tasks = _ready_tasks(state)
    if not tasks:
        return "budget_guard"
    remaining = state["budget"].max_tool_calls - state["budget"].tool_calls_used
    if remaining < len(tasks):
        tasks = tasks[:remaining]
    if not tasks:
        return "budget_guard"
    per_worker = max(1, remaining // len(tasks))
    return [
        Send(
            "researcher_worker",
            {
                "state": state,
                "task": task,
                "max_tool_calls": per_worker,
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
    tasks = list(state["research_tasks"])
    task_by_id = {task.task_id: task for task in tasks}
    total_used = 0
    errors = list(state["errors"])
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
        total_used += result.tool_calls_used
        errors.extend(result.errors)
        task = task_by_id[result.task_id]
        task_by_id[result.task_id] = task.model_copy(update={"status": "running"})
    budget = state["budget"]
    used = budget.tool_calls_used + total_used
    if used > budget.max_tool_calls:
        raise ValueError("parallel Researcher workers exceeded tool budget")
    return {
        "candidate_evidence": tuple(existing_candidates),
        "tool_calls": tuple(existing_calls),
        "research_tasks": tuple(task_by_id[task.task_id] for task in tasks),
        "budget": budget.model_copy(update={"tool_calls_used": used}),
        "iteration": state["iteration"] + 1,
        "errors": tuple(errors),
        "researcher_results_processed": len(state["researcher_results"]),
    }


def build_parallel_research_workflow(
    handlers: ResearchNodeHandlers,
    researcher: ParallelResearcherHandler,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> ParallelResearchWorkflow:
    """Compile Router → Planner → Send workers → fan-in → evidence nodes."""

    builder = StateGraph(ParallelResearchState)
    builder.add_node("router", _node(handlers.router))
    builder.add_node("planner", _node(handlers.planner))
    builder.add_node("dispatch", lambda state: {})
    builder.add_node("researcher_worker", _worker_node(researcher))
    builder.add_node("fan_in", _fan_in)
    builder.add_node("validator", _node(handlers.validator))
    builder.add_node("sufficiency_judge", _node(handlers.sufficiency_judge))
    builder.add_node("budget_guard", _budget_guard)
    builder.add_node("replanner", _replanner_node(handlers.replanner))
    builder.add_node("synthesizer", _node(handlers.synthesizer))

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", _route_after_router)
    builder.add_edge("planner", "dispatch")
    builder.add_conditional_edges("dispatch", _dispatch_workers)
    builder.add_edge("researcher_worker", "fan_in")
    builder.add_edge("fan_in", "validator")
    builder.add_edge("validator", "sufficiency_judge")
    builder.add_edge("sufficiency_judge", "budget_guard")
    builder.add_conditional_edges("budget_guard", _route_after_guard)
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

"""LangGraph fan-out/fan-in workflow for independent Researcher tasks."""

from __future__ import annotations

import operator
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from specrag.agents.researcher import RegistryResearcherService
from specrag.evidence import partial_synthesis_eligible
from specrag.models import (
    ResearchBudget,
    ResearchTask,
    ResearchTaskExecution,
    ResearchWorkerResult,
    TaskEvidenceSet,
)

from .adapters import researcher_execution_update
from .dependencies import build_dependency_context
from .gap_context import build_corrective_research_input
from .state import ResearchState, new_research_state
from .task_execution import (
    TERMINAL_TASK_STATUSES,
    prepare_task_executions,
    ready_tasks,
    terminalize_tasks,
)
from .workflow import (
    ResearchNodeHandlers,
    StateUpdate,
    _PARTIAL_SYNTHESIS_STOP_REASONS,
    _budget_guard,
    _has_judge_gap,
    _in_memory_checkpointer,
    _node,
    _replanner_node,
    EventSink,
    ReceiptCapture,
    RuntimeGuard,
    _receipt_payload,
)

_RUNTIME_COMPAT_KEYS = (
    "task_executions",
    "task_evidence_sets",
    "judge_review_records",
    "judge_skipped_task_ids",
)

class ParallelResearchState(ResearchState):
    """Unified research state with an append-only worker result channel."""

    researcher_results: Annotated[tuple[ResearchWorkerResult, ...], operator.add]


class WorkerInput(TypedDict):
    state: ParallelResearchState
    task: ResearchTask
    execution: ResearchTaskExecution
    max_tool_calls: int


@dataclass(frozen=True)
class ParallelResearcherHandler:
    """Run one isolated worker assignment for a LangGraph Send packet."""

    service: RegistryResearcherService

    def __call__(
        self,
        state: ParallelResearchState,
        task: ResearchTask,
        task_execution: ResearchTaskExecution,
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
        researcher_execution, query_to_task = self.service.run(
            (running,),
            shared_context=shared_context,
            max_tool_calls=max_tool_calls,
            **corrective_options,
        )
        update = researcher_execution_update(
            cast(ResearchState, state),
            researcher_execution,
            query_to_task=query_to_task,
            task_id=task.task_id,
            execution_id=task_execution.execution_id,
            corrective=corrective,
        )
        old_candidates = {
            evidence.evidence_id for evidence in state["candidate_evidence"]
        }
        old_calls = {call.call_id for call in state["tool_calls"]}
        old_selections = {
            item.selection_id for item in state["researcher_evidence_selections"]
        }
        new_receipts = tuple(
            {
                **receipt,
                "stage": "researcher",
            }
            for receipt in tuple(update["model_receipts"])[
                len(state["model_receipts"]) :
            ]
        )
        return ResearchWorkerResult(
            task_id=task.task_id,
            execution_id=task_execution.execution_id,
            candidate_evidence=tuple(
                evidence
                for evidence in update["candidate_evidence"]
                if evidence.evidence_id not in old_candidates
            ),
            active_evidence_ids=next(
                item.active_evidence_ids
                for item in update["task_evidence_sets"]
                if item.task_id == task.task_id
            ),
            tool_calls=tuple(
                call for call in update["tool_calls"] if call.call_id not in old_calls
            ),
            evidence_selections=tuple(
                item
                for item in update["researcher_evidence_selections"]
                if item.selection_id not in old_selections
            ),
            model_receipts=new_receipts,
            tool_calls_used=(
                researcher_execution.search_actions_used
                + researcher_execution.expansion_actions_used
            ),
            stop_reason=researcher_execution.stop_reason,
            errors=researcher_execution.errors,
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
            self.graph.invoke(
                _normalize_parallel_state(state),
                config=self._config(thread_id),
            ),
        )

    def resume(
        self,
        *,
        thread_id: str,
        update: StateUpdate | None = None,
    ) -> ParallelResearchState:
        config = self._config(thread_id)
        snapshot = self.graph.get_state(config)
        values = getattr(snapshot, "values", None)
        if values and any(
            key not in values for key in _RUNTIME_COMPAT_KEYS
        ):
            self.graph.update_state(
                config,
                {
                    "schema_version": "research-state/v1.10",
                    **{key: values.get(key, ()) for key in _RUNTIME_COMPAT_KEYS},
                },
            )
        if update:
            self.graph.update_state(config, dict(update))
        return cast(ParallelResearchState, self.graph.invoke(None, config=config))

    def checkpoint(self, *, thread_id: str) -> Any:
        return self.graph.get_state(self._config(thread_id))

    def update(self, *, thread_id: str, update: StateUpdate) -> None:
        self.graph.update_state(self._config(thread_id), dict(update))


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


def _normalize_parallel_state(
    state: ParallelResearchState,
) -> ParallelResearchState:
    if all(key in state for key in _RUNTIME_COMPAT_KEYS):
        return state
    return cast(
        ParallelResearchState,
        {
            **state,
            "schema_version": "research-state/v1.10",
            **{key: state.get(key, ()) for key in _RUNTIME_COMPAT_KEYS},
        },
    )


def _dispatch_workers(state: ParallelResearchState) -> list[Send] | str:
    budget = state["budget"]
    queued = tuple(item for item in state["task_executions"] if item.status == "queued")
    if not queued:
        return "budget_guard"
    task_by_id = {task.task_id: task for task in state["research_tasks"]}
    return [
        Send(
            "researcher_worker",
            {
                "state": state,
                "task": task_by_id[execution.task_id],
                "execution": execution,
                "max_tool_calls": budget.remaining_tool_calls_for(
                    execution.task_id,
                    corrective=execution.phase == "corrective",
                ),
            },
        )
        for execution in queued
    ]


def _worker_node(
    handler: ParallelResearcherHandler,
    guard: RuntimeGuard | None = None,
    event_sink: EventSink | None = None,
    receipt_capture: ReceiptCapture | None = None,
) -> Callable[[WorkerInput], dict[str, object]]:
    def invoke(packet: WorkerInput) -> dict[str, object]:
        from time import perf_counter

        state = packet["state"]
        if guard is not None:
            guard(state)
        started = perf_counter()
        capture = receipt_capture() if receipt_capture is not None else nullcontext([])
        with capture as receipts:
            try:
                result = handler(
                    state,
                    packet["task"],
                    packet["execution"],
                    max_tool_calls=packet["max_tool_calls"],
                )
            except Exception as exc:  # noqa: BLE001 - isolate sibling executions
                if getattr(exc, "reason", None) in {"cancelled", "elapsed_time_budget"}:
                    raise
                result = ResearchWorkerResult(
                    task_id=packet["task"].task_id,
                    execution_id=packet["execution"].execution_id,
                    execution_status="failed",
                    stop_reason="agent_error",
                    errors=("task execution failed",),
                )
        latency_ms = (perf_counter() - started) * 1000
        known_receipts = list(result.model_receipts)
        duplicate_counts: dict[tuple[object, object, object], int] = {}
        for item in known_receipts:
            key = (
                item.get("request_sha256"),
                item.get("response_sha256"),
                item.get("purpose"),
            )
            duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
        for receipt in receipts:
            payload = _receipt_payload(receipt, "researcher")
            key = (
                payload.get("request_sha256"),
                payload.get("response_sha256"),
                payload.get("purpose"),
            )
            duplicates = duplicate_counts.get(key, 0)
            if duplicates:
                duplicate_counts[key] = duplicates - 1
                continue
            known_receipts.append(payload)
        result = result.model_copy(
            update={
                "model_receipts": tuple(known_receipts),
                "latency_ms": latency_ms,
            }
        )
        update: dict[str, object] = {"researcher_results": (result,)}
        if event_sink is not None:
            event_sink(
                "researcher_worker",
                state,
                update,
                latency_ms,
            )
        return update

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
    executions = {item.execution_id: item for item in state["task_executions"]}
    errors = list(state["errors"])
    receipts = list(state["model_receipts"])
    budget = state["budget"]
    evidence_sets = {
        item.task_id: item for item in state.get("task_evidence_sets", ())
    }
    new_results = state["researcher_results"][state["researcher_results_processed"] :]
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
            corrective=(
                result.execution_id is not None
                and executions[result.execution_id].phase == "corrective"
            ),
        )
        errors.extend(result.errors)
        receipts.extend(result.model_receipts)
        task = task_by_id[result.task_id]
        task_by_id[result.task_id] = task.model_copy(
            update={
                "status": "failed" if result.execution_status == "failed" else "running"
            }
        )
        if result.execution_id is not None:
            execution = executions[result.execution_id]
            executions[result.execution_id] = execution.model_copy(
                update={
                    "status": result.execution_status,
                    "stop_reason": result.stop_reason,
                    "error": (
                        "task execution failed"
                        if result.execution_status == "failed"
                        else None
                    ),
                }
            )
        evidence_sets[result.task_id] = TaskEvidenceSet(
            task_id=result.task_id,
            execution_id=result.execution_id,
            active_evidence_ids=result.active_evidence_ids,
        )
    return {
        "candidate_evidence": tuple(existing_candidates),
        "tool_calls": tuple(existing_calls),
        "researcher_evidence_selections": tuple(existing_selections),
        "task_evidence_sets": tuple(
            evidence_sets[task.task_id]
            for task in tasks
            if task.task_id in evidence_sets
        ),
        "research_tasks": tuple(task_by_id[task.task_id] for task in tasks),
        "task_executions": tuple(
            executions[item.execution_id] for item in state["task_executions"]
        ),
        "budget": budget,
        "iteration": state["iteration"] + 1,
        "errors": tuple(errors),
        "model_receipts": tuple(receipts),
        "researcher_results_processed": len(state["researcher_results"]),
    }


def _route_after_parallel_guard(
    state: ParallelResearchState,
) -> str:
    if state["evidence_sufficient"] is True or state["stop_reason"] is not None:
        return "terminalize_tasks"
    if any(task.status == "pending" for task in ready_tasks(state)):
        return "prepare_executions"
    return "replanner"


def _route_after_terminalization(state: ParallelResearchState) -> str:
    if not state["research_tasks"] or any(
        task.status not in TERMINAL_TASK_STATUSES for task in state["research_tasks"]
    ):
        raise RuntimeError("research synthesis join barrier is not closed")
    if state["evidence_sufficient"] is True:
        return "synthesizer"
    if state[
        "stop_reason"
    ] in _PARTIAL_SYNTHESIS_STOP_REASONS and partial_synthesis_eligible(
        state["validated_evidence"],
        state["sufficiency_assessments"],
    ):
        return "synthesizer"
    return END


def build_parallel_research_workflow(
    handlers: ResearchNodeHandlers,
    researcher: ParallelResearcherHandler,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    runtime_guard: RuntimeGuard | None = None,
    event_sink: EventSink | None = None,
    receipt_capture: ReceiptCapture | None = None,
    research_deadline_at: datetime | None = None,
    max_parallel_tasks: int = 4,
) -> ParallelResearchWorkflow:
    """Compile Router → Planner → Send workers → fan-in → evidence nodes."""

    builder = StateGraph(ParallelResearchState)
    builder.add_node(
        "router",
        _node("router", handlers.router, runtime_guard, event_sink, receipt_capture),
    )
    builder.add_node(
        "planner",
        _node("planner", handlers.planner, runtime_guard, event_sink, receipt_capture),
    )
    builder.add_node(
        "prepare_executions",
        _node(
            "prepare_executions",
            lambda state: prepare_task_executions(
                state, max_parallel_tasks=max_parallel_tasks
            ),
            runtime_guard,
            event_sink,
            receipt_capture,
        ),
    )
    builder.add_node("dispatch", lambda state: {})
    builder.add_node(
        "researcher_worker",
        _worker_node(researcher, runtime_guard, None, receipt_capture),
    )
    builder.add_node(
        "fan_in", _node("fan_in", _fan_in, runtime_guard, event_sink, receipt_capture)
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
        lambda state: _budget_guard(state, research_deadline_at=research_deadline_at),
    )
    builder.add_node(
        "replanner",
        _replanner_node(handlers.replanner, runtime_guard, event_sink, receipt_capture),
    )
    builder.add_node(
        "terminalize_tasks",
        _node(
            "terminalize_tasks",
            terminalize_tasks,
            runtime_guard,
            event_sink,
            receipt_capture,
        ),
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
    builder.add_edge("planner", "prepare_executions")
    builder.add_edge("prepare_executions", "dispatch")
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
        ["prepare_executions", "replanner", "terminalize_tasks"],
    )
    builder.add_edge("replanner", "prepare_executions")
    builder.add_conditional_edges(
        "terminalize_tasks",
        _route_after_terminalization,
        ["synthesizer", END],
    )
    builder.add_edge("synthesizer", END)

    compiled = builder.compile(checkpointer=checkpointer or _in_memory_checkpointer())
    return ParallelResearchWorkflow(graph=compiled)


__all__ = [
    "ParallelResearchState",
    "ParallelResearcherHandler",
    "ParallelResearchWorkflow",
    "build_parallel_research_workflow",
    "new_parallel_research_state",
]

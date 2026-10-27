"""Deterministic ResearchTask execution scheduling and join semantics."""

from __future__ import annotations

from specrag.models import (
    ResearchGap,
    ResearchTask,
    ResearchTaskExecution,
    TaskSufficiencyAssessment,
)
from specrag.evidence import publication_decision

from .state import ResearchState
from .workflow import _has_judge_gap


TERMINAL_TASK_STATUSES = frozenset({"completed", "blocked", "failed"})


def ready_tasks(state: ResearchState) -> tuple[ResearchTask, ...]:
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


def prepare_task_executions(
    state: ResearchState,
    *,
    max_parallel_tasks: int,
) -> dict[str, object]:
    if max_parallel_tasks < 1:
        raise ValueError("max_parallel_tasks must be positive")
    active_task_ids = {
        item.task_id
        for item in state["task_executions"]
        if item.status in {"queued", "running"}
    }
    ready = tuple(
        task for task in ready_tasks(state)
        if task.task_id not in active_task_ids
        and state["budget"].remaining_tool_calls_for(
            task.task_id,
            corrective=_has_judge_gap(state, task.task_id),
        ) > 0
    )[:max_parallel_tasks]
    if not ready:
        return {}
    history = list(state["task_executions"])
    task_by_id = {task.task_id: task for task in state["research_tasks"]}
    for task in ready:
        attempt = 1 + sum(item.task_id == task.task_id for item in history)
        corrective = _has_judge_gap(state, task.task_id)
        history.append(ResearchTaskExecution(
            execution_id=(
                f"EXEC:{state['research_id']}:{task.task_id}:{attempt}"
            ),
            task_id=task.task_id,
            attempt=attempt,
            phase="corrective" if corrective else "initial",
            status="queued",
        ))
        task_by_id[task.task_id] = task.model_copy(update={"status": "running"})
    return {
        "task_executions": tuple(history),
        "research_tasks": tuple(
            task_by_id[task.task_id] for task in state["research_tasks"]
        ),
    }


def terminalize_tasks(state: ResearchState) -> dict[str, object]:
    """Close every task lifecycle without inventing source support."""

    assessment_by_task = {
        item.task_id: item for item in state["sufficiency_assessments"]
    }
    assessments = list(state["sufficiency_assessments"])
    gaps = list(state["gaps"])
    gap_ids = {item.gap_id for item in gaps}
    tasks: list[ResearchTask] = []
    for task in state["research_tasks"]:
        if task.task_id not in assessment_by_task:
            assessment = TaskSufficiencyAssessment(
                task_id=task.task_id,
                status="insufficient",
                missing_relation=(
                    "The task reached the research join barrier without a "
                    "validated sufficiency assessment."
                ),
                rationale="No specification conclusion is published for this task.",
            )
            assessments.append(assessment)
            assessment_by_task[task.task_id] = assessment
            gap_id = f"G:{task.task_id}:join_barrier"
            if gap_id not in gap_ids:
                gaps.append(ResearchGap(
                    gap_id=gap_id,
                    task_id=task.task_id,
                    kind="missing_relation",
                    description=assessment.missing_relation or assessment.rationale,
                    required_proposition=assessment.required_proposition,
                ))
                gap_ids.add(gap_id)
        if task.status in TERMINAL_TASK_STATUSES:
            tasks.append(task)
        elif task.status == "pending":
            tasks.append(task.model_copy(update={"status": "blocked"}))
        else:
            tasks.append(task.model_copy(update={"status": "completed"}))
    final_assessments = tuple(assessments)
    decision = publication_decision(
        tuple(tasks),
        state["validated_evidence"],
        final_assessments,
    )
    return {
        "research_tasks": tuple(tasks),
        "sufficiency_assessments": final_assessments,
        "gaps": tuple(gaps),
        "publication_path": decision.path,
        "evidence_sufficient": bool(tasks) and all(
            assessment_by_task[task.task_id].status == "sufficient"
            for task in tasks
        ),
    }


__all__ = [
    "TERMINAL_TASK_STATUSES",
    "prepare_task_executions",
    "ready_tasks",
    "terminalize_tasks",
]

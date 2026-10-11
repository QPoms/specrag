"""Build validated dependency inputs for Researcher workers."""

from __future__ import annotations

from typing import Any

from specrag.models import ResearchTask

from .state import ResearchState


def build_dependency_context(
    state: ResearchState,
    task: ResearchTask,
) -> tuple[dict[str, Any], ...]:
    """Return source-backed findings required by one dependent task.

    Only evidence selected by a sufficient predecessor assessment and validated
    as supported is exposed. Dependency evidence remains owned by its original
    task; the downstream Researcher uses it to scope new research rather than
    relabelling it as evidence for the dependent task.
    """

    if not task.dependencies:
        return ()

    tasks_by_id = {
        candidate_task.task_id: candidate_task
        for candidate_task in state["research_tasks"]
    }
    assessments_by_task = {
        assessment.task_id: assessment
        for assessment in state["sufficiency_assessments"]
    }
    candidates_by_id = {
        evidence.evidence_id: evidence
        for evidence in state["candidate_evidence"]
    }
    validations_by_id = {
        evidence.evidence_id: evidence
        for evidence in state["validated_evidence"]
    }

    context: list[dict[str, Any]] = []
    for dependency_id in task.dependencies:
        dependency = tasks_by_id.get(dependency_id)
        if dependency is None:
            raise ValueError(
                f"dependent task {task.task_id} references unknown task "
                f"{dependency_id}"
            )
        if dependency.status != "completed":
            raise ValueError(
                f"dependency {dependency_id} is not completed for {task.task_id}"
            )

        assessment = assessments_by_task.get(dependency_id)
        if assessment is None or assessment.status != "sufficient":
            raise ValueError(
                f"dependency {dependency_id} has no sufficient assessment for "
                f"{task.task_id}"
            )

        evidence_items: list[dict[str, Any]] = []
        for evidence_id in assessment.supporting_evidence_ids:
            candidate = candidates_by_id.get(evidence_id)
            validation = validations_by_id.get(evidence_id)
            if candidate is None or validation is None:
                raise ValueError(
                    f"dependency {dependency_id} supporting evidence {evidence_id} "
                    "is missing from workflow state"
                )
            if candidate.task_id != dependency_id:
                raise ValueError(
                    f"dependency evidence {evidence_id} belongs to "
                    f"{candidate.task_id}, not {dependency_id}"
                )
            if validation.task_id != dependency_id:
                raise ValueError(
                    f"dependency validation {evidence_id} belongs to "
                    f"{validation.task_id}, not {dependency_id}"
                )
            if not (
                validation.source_valid
                and validation.supports_target
                and validation.validation_status == "supported"
            ):
                continue
            evidence_items.append({
                **candidate.model_dump(mode="json"),
                "validation_rationale": validation.rationale,
            })

        if not evidence_items:
            raise ValueError(
                f"dependency {dependency_id} has no supported validated evidence "
                f"for {task.task_id}"
            )
        context.append({
            "task_id": dependency.task_id,
            "question": dependency.question,
            "objective": dependency.objective,
            "sufficiency_rationale": assessment.rationale,
            "supporting_evidence": evidence_items,
        })

    return tuple(context)


__all__ = ["build_dependency_context"]

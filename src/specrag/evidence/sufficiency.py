"""Task-level evidence sufficiency and explicit Gap generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from specrag.models import (
    CandidateEvidence,
    EvidenceConflict,
    ResearchGap,
    ResearchTask,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)


class TaskSufficiencyAssessor(Protocol):
    def assess(
        self,
        task: ResearchTask,
        candidates: tuple[CandidateEvidence, ...],
        evidence: tuple[ValidatedEvidence, ...],
    ) -> TaskSufficiencyAssessment: ...


@dataclass(frozen=True)
class EvidenceSufficiencyBatch:
    assessments: tuple[TaskSufficiencyAssessment, ...]
    gaps: tuple[ResearchGap, ...]
    tasks: tuple[ResearchTask, ...]
    evidence_sufficient: bool


class EvidenceSufficiencyService:
    def __init__(self, assessor: TaskSufficiencyAssessor) -> None:
        self.assessor = assessor

    def evaluate(
        self,
        tasks: tuple[ResearchTask, ...],
        candidates: tuple[CandidateEvidence, ...],
        evidence: tuple[ValidatedEvidence, ...],
        conflicts: tuple[EvidenceConflict, ...] = (),
    ) -> EvidenceSufficiencyBatch:
        assessments: list[TaskSufficiencyAssessment] = []
        gaps: list[ResearchGap] = []
        updated_tasks: list[ResearchTask] = []
        for task in tasks:
            task_candidates = tuple(
                item for item in candidates if item.task_id == task.task_id
            )
            task_evidence = tuple(
                item for item in evidence if item.task_id == task.task_id
            )
            task_conflicts = tuple(
                item for item in conflicts if item.task_id == task.task_id
            )
            if (
                task.status == "pending"
                and not task_candidates
                and not task_evidence
                and not task_conflicts
            ):
                # A dependency-blocked task has not run yet. It has no evidence
                # to assess and must not create a synthetic gap or consume a
                # re-plan before its predecessors complete.
                updated_tasks.append(task)
                continue
            if task_conflicts:
                description = "; ".join(
                    item.description for item in task_conflicts
                )
                assessment = TaskSufficiencyAssessment(
                    task_id=task.task_id,
                    status="conflict",
                    supporting_evidence_ids=(),
                    missing_relation=description,
                    rationale="Conflicting validated evidence requires resolution.",
                )
            else:
                assessment = self.assessor.assess(
                    task,
                    task_candidates,
                    task_evidence,
                )
            if assessment.task_id != task.task_id:
                raise ValueError(
                    "sufficiency assessment returned a mismatched task ID"
                )
            eligible = {
                item.evidence_id
                for item in task_evidence
                if item.source_valid
                and item.validation_status in {"supported", "partial"}
            }
            unknown = set(assessment.supporting_evidence_ids) - eligible
            if unknown:
                raise ValueError(
                    "sufficiency assessment cites ineligible evidence: "
                    + ", ".join(sorted(unknown))
                )
            if assessment.status == "sufficient" and not (
                set(assessment.supporting_evidence_ids)
                & {
                    item.evidence_id
                    for item in task_evidence
                    if item.source_valid
                    and item.validation_status == "supported"
                }
            ):
                raise ValueError(
                    "sufficient task requires at least one supported evidence item"
                )

            assessments.append(assessment)
            if assessment.status == "sufficient":
                status = "completed"
            elif assessment.status == "conflict":
                status = "blocked"
            else:
                status = "running"
            updated_tasks.append(task.model_copy(update={"status": status}))
            if assessment.status != "sufficient":
                kind = (
                    "source_conflict"
                    if assessment.status == "conflict"
                    else "missing_relation"
                )
                gaps.append(ResearchGap(
                    gap_id=f"G:{task.task_id}:{kind}",
                    task_id=task.task_id,
                    kind=kind,
                    description=assessment.missing_relation or assessment.rationale,
                ))

        sufficient = bool(tasks) and all(
            item.status == "completed" for item in updated_tasks
        )
        return EvidenceSufficiencyBatch(
            assessments=tuple(assessments),
            gaps=tuple(gaps),
            tasks=tuple(updated_tasks),
            evidence_sufficient=sufficient,
        )


__all__ = [
    "EvidenceSufficiencyBatch",
    "EvidenceSufficiencyService",
    "TaskSufficiencyAssessor",
]

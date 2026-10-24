"""Task-level evidence sufficiency and explicit Gap generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from specrag.models import (
    CandidateEvidence,
    ResearchGap,
    ResearchTask,
    SourceValidationResult,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)


class TaskSufficiencyAssessor(Protocol):
    def assess(
        self,
        task: ResearchTask,
        evidence: tuple[CandidateEvidence, ...],
        original_question: str | None = None,
    ) -> TaskSufficiencyAssessment: ...


@dataclass(frozen=True)
class EvidenceSufficiencyBatch:
    assessments: tuple[TaskSufficiencyAssessment, ...]
    gaps: tuple[ResearchGap, ...]
    tasks: tuple[ResearchTask, ...]
    validated_evidence: tuple[ValidatedEvidence, ...]
    evidence_sufficient: bool


class EvidenceSufficiencyService:
    def __init__(self, assessor: TaskSufficiencyAssessor) -> None:
        self.assessor = assessor

    def evaluate(
        self,
        tasks: tuple[ResearchTask, ...],
        candidates: tuple[CandidateEvidence, ...],
        source_validations: tuple[SourceValidationResult, ...],
        original_question: str | None = None,
    ) -> EvidenceSufficiencyBatch:
        candidate_by_id = {item.evidence_id: item for item in candidates}
        if len(candidate_by_id) != len(candidates):
            raise ValueError("candidate evidence IDs must be unique")
        validation_by_id = {
            item.evidence_id: item for item in source_validations
        }
        if len(validation_by_id) != len(source_validations):
            raise ValueError("source validation evidence IDs must be unique")
        unknown_validations = set(validation_by_id) - set(candidate_by_id)
        if unknown_validations:
            raise ValueError(
                "source validation references unknown evidence: "
                + ", ".join(sorted(unknown_validations))
            )

        assessments: list[TaskSufficiencyAssessment] = []
        gaps: list[ResearchGap] = []
        updated_tasks: list[ResearchTask] = []
        assessment_by_task: dict[str, TaskSufficiencyAssessment] = {}
        for task in tasks:
            task_candidates = tuple(
                item for item in candidates if item.task_id == task.task_id
            )
            valid_candidates = tuple(
                item
                for item in task_candidates
                if (
                    item.evidence_id in validation_by_id
                    and validation_by_id[item.evidence_id].valid
                )
            )
            if (
                task.status == "pending"
                and not task_candidates
            ):
                # A dependency-blocked task has not run yet. It has no evidence
                # to assess and must not create a synthetic gap or consume a
                # re-plan before its predecessors complete.
                updated_tasks.append(task)
                continue
            if original_question is None:
                assessment = self.assessor.assess(task, valid_candidates)
            else:
                assessment = self.assessor.assess(
                    task,
                    valid_candidates,
                    original_question,
                )
            eligible = {
                item.evidence_id for item in valid_candidates
            }
            unknown = set(assessment.supporting_evidence_ids) - eligible
            rejection_reason = None
            if assessment.task_id != task.task_id:
                rejection_reason = "The assessment returned a mismatched task ID."
            elif unknown:
                rejection_reason = (
                    "The assessment referenced evidence that is not source-valid "
                    "and eligible for this task."
                )
            elif (
                assessment.status == "sufficient"
                and not assessment.supporting_evidence_ids
            ):
                rejection_reason = (
                    "The assessment claimed sufficiency without selecting evidence."
                )
            if rejection_reason is not None:
                assessment = TaskSufficiencyAssessment(
                    task_id=task.task_id,
                    status="insufficient",
                    supporting_evidence_ids=tuple(
                        evidence_id
                        for evidence_id in assessment.supporting_evidence_ids
                        if evidence_id in eligible
                    ),
                    missing_relation=rejection_reason,
                    required_proposition=task.objective,
                    evidence_need="direct_fact",
                    rationale=(
                        "The model-proposed sufficiency decision was rejected by "
                        "the deterministic evidence boundary."
                    ),
                )

            grounded_anchors = _grounded_gap_anchors(
                assessment.trusted_anchors,
                task=task,
                original_question=original_question,
                evidence=valid_candidates,
            )
            if grounded_anchors != assessment.trusted_anchors:
                assessment = assessment.model_copy(update={
                    "trusted_anchors": grounded_anchors,
                })

            assessments.append(assessment)
            assessment_by_task[task.task_id] = assessment
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
                    required_proposition=assessment.required_proposition,
                    missing_dimensions=assessment.missing_dimensions,
                    evidence_need=assessment.evidence_need or "relationship",
                    trusted_anchors=assessment.trusted_anchors,
                    supporting_evidence_ids=(
                        assessment.supporting_evidence_ids
                    ),
                ))

        validated: list[ValidatedEvidence] = []
        for candidate in candidates:
            source_result = validation_by_id.get(candidate.evidence_id)
            if source_result is None:
                continue
            assessment = assessment_by_task.get(candidate.task_id)
            selected = bool(
                assessment
                and candidate.evidence_id
                in assessment.supporting_evidence_ids
            )
            if not source_result.valid:
                validated.append(ValidatedEvidence(
                    evidence_id=candidate.evidence_id,
                    task_id=candidate.task_id,
                    source_valid=False,
                    supports_target=False,
                    validation_status="rejected",
                    rationale=(
                        "; ".join(source_result.errors)
                        or "Source provenance validation failed."
                    ),
                ))
            elif not selected:
                validated.append(ValidatedEvidence(
                    evidence_id=candidate.evidence_id,
                    task_id=candidate.task_id,
                    source_valid=True,
                    supports_target=False,
                    validation_status="rejected",
                    rationale=(
                        "Not selected by the task-level sufficiency assessment."
                    ),
                ))
            else:
                assert assessment is not None
                validation_status = {
                    "sufficient": "supported",
                    "partial": "partial",
                    "insufficient": "partial",
                    "conflict": "conflict",
                }[assessment.status]
                validated.append(ValidatedEvidence(
                    evidence_id=candidate.evidence_id,
                    task_id=candidate.task_id,
                    source_valid=True,
                    supports_target=assessment.status != "conflict",
                    validation_status=validation_status,
                    rationale=assessment.rationale,
                ))

        sufficient = bool(tasks) and all(
            item.status == "completed" for item in updated_tasks
        )
        return EvidenceSufficiencyBatch(
            assessments=tuple(assessments),
            gaps=tuple(gaps),
            tasks=tuple(updated_tasks),
            validated_evidence=tuple(validated),
            evidence_sufficient=sufficient,
        )


def _grounded_gap_anchors(
    anchors: tuple[str, ...],
    *,
    task: ResearchTask,
    original_question: str | None,
    evidence: tuple[CandidateEvidence, ...],
) -> tuple[str, ...]:
    """Keep only exact anchors present in trusted task or source content."""

    trusted_text = "\n".join((
        task.question,
        original_question or "",
        *(
            "\n".join((
                item.specification,
                *item.heading_path,
                item.excerpt,
            ))
            for item in evidence
        ),
    )).casefold()
    grounded: list[str] = []
    seen: set[str] = set()
    for raw_anchor in anchors:
        anchor = raw_anchor.strip()
        normalized = anchor.casefold()
        if not anchor or normalized in seen or normalized not in trusted_text:
            continue
        seen.add(normalized)
        grounded.append(anchor)
    return tuple(grounded)


__all__ = [
    "EvidenceSufficiencyBatch",
    "EvidenceSufficiencyService",
    "TaskSufficiencyAssessor",
]

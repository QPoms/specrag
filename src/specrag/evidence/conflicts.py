"""Validated boundary for semantic and specification-version conflicts."""

from __future__ import annotations

from typing import Protocol

from specrag.models import (
    CandidateEvidence,
    EvidenceConflict,
    ResearchTask,
    ValidatedEvidence,
)


class EvidenceConflictAssessor(Protocol):
    def detect(
        self,
        task: ResearchTask,
        evidence: tuple[CandidateEvidence, ...],
    ) -> tuple[EvidenceConflict, ...]: ...


class EvidenceConflictService:
    def __init__(self, assessor: EvidenceConflictAssessor) -> None:
        self.assessor = assessor

    def detect(
        self,
        tasks: tuple[ResearchTask, ...],
        candidates: tuple[CandidateEvidence, ...],
        validated: tuple[ValidatedEvidence, ...],
    ) -> tuple[EvidenceConflict, ...]:
        eligible = {
            item.evidence_id
            for item in validated
            if item.source_valid
            and item.validation_status in {"supported", "partial", "conflict"}
        }
        candidate_by_id = {
            item.evidence_id: item
            for item in candidates
            if item.evidence_id in eligible
        }
        conflicts: list[EvidenceConflict] = []
        seen_ids: set[str] = set()
        for task in tasks:
            task_evidence = tuple(
                item
                for item in candidate_by_id.values()
                if item.task_id == task.task_id
            )
            for conflict in self.assessor.detect(task, task_evidence):
                if conflict.conflict_id in seen_ids:
                    raise ValueError(
                        f"duplicate conflict ID: {conflict.conflict_id}"
                    )
                if conflict.task_id != task.task_id:
                    raise ValueError("conflict returned a mismatched task ID")
                unknown = set(conflict.evidence_ids) - set(candidate_by_id)
                if unknown:
                    raise ValueError(
                        "conflict cites ineligible evidence: "
                        + ", ".join(sorted(unknown))
                    )
                if any(
                    candidate_by_id[evidence_id].task_id != task.task_id
                    for evidence_id in conflict.evidence_ids
                ):
                    raise ValueError("conflict crosses research-task boundaries")
                conflicts.append(conflict)
                seen_ids.add(conflict.conflict_id)
        return tuple(conflicts)


__all__ = ["EvidenceConflictAssessor", "EvidenceConflictService"]

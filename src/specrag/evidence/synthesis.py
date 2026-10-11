"""Grounded answer synthesis constrained to validated evidence identifiers."""

from __future__ import annotations

from typing import Protocol

from specrag.models import (
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchTask,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)


class EvidenceSynthesisGenerator(Protocol):
    def synthesize(
        self,
        question: str,
        tasks: tuple[ResearchTask, ...],
        evidence: tuple[CandidateEvidence, ...],
    ) -> EvidenceSynthesis: ...


class EvidenceSynthesisService:
    def __init__(self, generator: EvidenceSynthesisGenerator) -> None:
        self.generator = generator

    def synthesize(
        self,
        question: str,
        tasks: tuple[ResearchTask, ...],
        candidates: tuple[CandidateEvidence, ...],
        validated: tuple[ValidatedEvidence, ...],
        assessments: tuple[TaskSufficiencyAssessment, ...],
    ) -> EvidenceSynthesis:
        sufficient_tasks = {
            item.task_id
            for item in assessments
            if item.status == "sufficient"
        }
        expected_tasks = {task.task_id for task in tasks}
        if not expected_tasks or sufficient_tasks != expected_tasks:
            raise ValueError("cannot synthesize before every task is sufficient")

        eligible_ids = {
            item.evidence_id
            for item in validated
            if item.source_valid and item.validation_status == "supported"
        }
        evidence = tuple(
            item for item in candidates if item.evidence_id in eligible_ids
        )
        if not evidence:
            raise ValueError("cannot synthesize without supported evidence")
        proposal = self.generator.synthesize(question, tasks, evidence)
        claim_ids = [claim.claim_id for claim in proposal.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("synthesis claim IDs must be unique")
        for claim in proposal.claims:
            unknown = set(claim.evidence_ids) - eligible_ids
            if unknown:
                raise ValueError(
                    f"claim {claim.claim_id} cites ineligible evidence: "
                    + ", ".join(sorted(unknown))
                )
        return proposal


__all__ = ["EvidenceSynthesisGenerator", "EvidenceSynthesisService"]

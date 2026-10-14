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
            if (
                item.task_id in expected_tasks
                and item.source_valid
                and item.validation_status == "supported"
            )
        }
        evidence = tuple(
            item for item in candidates if item.evidence_id in eligible_ids
        )
        if not evidence:
            raise ValueError("cannot synthesize without supported evidence")
        evidence_by_id = {item.evidence_id: item for item in evidence}
        evidence_tasks = {item.task_id for item in evidence}
        missing_evidence_tasks = expected_tasks - evidence_tasks
        if missing_evidence_tasks:
            raise ValueError(
                "cannot synthesize tasks without supported evidence: "
                + ", ".join(sorted(missing_evidence_tasks))
            )
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
        covered_tasks = {
            evidence_by_id[evidence_id].task_id
            for claim in proposal.claims
            for evidence_id in claim.evidence_ids
        }
        missing_claim_tasks = expected_tasks - covered_tasks
        if missing_claim_tasks:
            raise ValueError(
                "synthesis claims do not cover tasks: "
                + ", ".join(sorted(missing_claim_tasks))
            )
        return proposal


__all__ = ["EvidenceSynthesisGenerator", "EvidenceSynthesisService"]

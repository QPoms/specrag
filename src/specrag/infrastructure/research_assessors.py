"""Structured-model adapters for evidence research interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from specrag.models import (
    CandidateEvidence,
    EvidenceConflict,
    EvidenceSupportAssessment,
    EvidenceSynthesis,
    ResearchTask,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)

from .structured_agent import CodexStructuredAgent


class _ConflictBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conflicts: tuple[EvidenceConflict, ...] = ()


@dataclass(frozen=True)
class StructuredEvidenceSupportAssessor:
    agent: CodexStructuredAgent

    def assess(
        self,
        task: ResearchTask,
        evidence: CandidateEvidence,
    ) -> EvidenceSupportAssessment:
        response = self.agent.parse(
            purpose="assess whether one source passage supports a research task",
            prompt_version="evidence-support/v1.0",
            instructions=(
                "Judge only the supplied excerpt. Use supported only when it "
                "directly supports the task objective, partial when it supports "
                "only part, rejected when it is irrelevant, and conflict when "
                "it directly contradicts the target. Preserve evidence_id."
            ),
            input_payload={
                "task": task.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
            },
            output_type=EvidenceSupportAssessment,
        )
        return response.output


@dataclass(frozen=True)
class StructuredTaskSufficiencyAssessor:
    agent: CodexStructuredAgent

    def assess(
        self,
        task: ResearchTask,
        candidates: tuple[CandidateEvidence, ...],
        evidence: tuple[ValidatedEvidence, ...],
    ) -> TaskSufficiencyAssessment:
        response = self.agent.parse(
            purpose="judge task-level evidence sufficiency",
            prompt_version="evidence-sufficiency/v1.0",
            instructions=(
                "Decide whether the validated evidence covers the complete task "
                "objective. Cite only supplied eligible evidence IDs. If the "
                "task is not sufficient, state one concrete missing_relation "
                "that can guide the next retrieval. Preserve task_id."
            ),
            input_payload={
                "task": task.model_dump(mode="json"),
                "candidate_evidence": [
                    item.model_dump(mode="json") for item in candidates
                ],
                "validation": [
                    item.model_dump(mode="json") for item in evidence
                ],
            },
            output_type=TaskSufficiencyAssessment,
        )
        return response.output


@dataclass(frozen=True)
class StructuredEvidenceConflictAssessor:
    agent: CodexStructuredAgent

    def detect(
        self,
        task: ResearchTask,
        evidence: tuple[CandidateEvidence, ...],
    ) -> tuple[EvidenceConflict, ...]:
        response = self.agent.parse(
            purpose="detect conflicts between validated source passages",
            prompt_version="evidence-conflict/v1.0",
            instructions=(
                "Return only direct semantic contradictions or incompatible "
                "version conditions between two supplied evidence items. Every "
                "conflict must cite exactly two supplied IDs from this task."
            ),
            input_payload={
                "task": task.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
            output_type=_ConflictBatch,
        )
        return response.output.conflicts


@dataclass(frozen=True)
class StructuredEvidenceSynthesisGenerator:
    agent: CodexStructuredAgent

    def synthesize(
        self,
        question: str,
        tasks: tuple[ResearchTask, ...],
        evidence: tuple[CandidateEvidence, ...],
    ) -> EvidenceSynthesis:
        response = self.agent.parse(
            purpose="synthesize an evidence-grounded technical answer",
            prompt_version="evidence-synthesis/v1.0",
            instructions=(
                "Answer only from the supplied validated source passages. Split "
                "the answer into explicit claims and attach one or more supplied "
                "evidence IDs to every claim. Do not invent an evidence ID."
            ),
            input_payload={
                "question": question,
                "tasks": [item.model_dump(mode="json") for item in tasks],
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
            output_type=EvidenceSynthesis,
        )
        return response.output


__all__ = [
    "StructuredEvidenceConflictAssessor",
    "StructuredEvidenceSupportAssessor",
    "StructuredEvidenceSynthesisGenerator",
    "StructuredTaskSufficiencyAssessor",
]

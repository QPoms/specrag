"""Structured-model adapters for evidence research interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from specrag.models import (
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchTask,
    TaskSufficiencyAssessment,
)
from specrag.prompts import (
    SUFFICIENCY_INSTRUCTIONS,
    SUFFICIENCY_PROMPT_VERSION,
)

from .structured_agent import CodexStructuredAgent


@dataclass(frozen=True)
class StructuredTaskSufficiencyAssessor:
    agent: CodexStructuredAgent

    def assess(
        self,
        task: ResearchTask,
        evidence: tuple[CandidateEvidence, ...],
    ) -> TaskSufficiencyAssessment:
        response = self.agent.parse(
            purpose="select supporting evidence and judge task sufficiency",
            prompt_version=SUFFICIENCY_PROMPT_VERSION,
            instructions=SUFFICIENCY_INSTRUCTIONS,
            input_payload={
                "task": {
                    "task_id": task.task_id,
                    "question": task.question,
                },
                "source_valid_evidence": [
                    item.model_dump(mode="json") for item in evidence
                ],
            },
            output_type=TaskSufficiencyAssessment,
        )
        return response.output


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
    "StructuredEvidenceSynthesisGenerator",
    "StructuredTaskSufficiencyAssessor",
]

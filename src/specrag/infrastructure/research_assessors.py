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
    SYNTHESIS_INSTRUCTIONS,
    SYNTHESIS_PROMPT_VERSION,
)

from .structured_agent import CodexStructuredAgent


@dataclass(frozen=True)
class StructuredTaskSufficiencyAssessor:
    agent: CodexStructuredAgent

    def assess(
        self,
        task: ResearchTask,
        evidence: tuple[CandidateEvidence, ...],
        original_question: str | None = None,
    ) -> TaskSufficiencyAssessment:
        response = self.agent.parse(
            purpose="select supporting evidence and judge task sufficiency",
            prompt_version=SUFFICIENCY_PROMPT_VERSION,
            instructions=SUFFICIENCY_INSTRUCTIONS,
            input_payload={
                "original_question": original_question,
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
    answer_language: str = "English"
    detail_level: str = "standard"

    def synthesize(
        self,
        question: str,
        tasks: tuple[ResearchTask, ...],
        evidence: tuple[CandidateEvidence, ...],
    ) -> EvidenceSynthesis:
        response = self.agent.parse(
            purpose="synthesize an evidence-grounded technical answer",
            prompt_version=SYNTHESIS_PROMPT_VERSION,
            instructions=SYNTHESIS_INSTRUCTIONS,
            input_payload={
                "question": question,
                "presentation": {
                    "answer_language": self.answer_language,
                    "detail_level": self.detail_level,
                },
                "task_evidence": [
                    {
                        "task_id": task.task_id,
                        "task_question": task.question,
                        "supported_evidence": [
                            item.model_dump(mode="json")
                            for item in evidence
                            if item.task_id == task.task_id
                        ],
                    }
                    for task in tasks
                ],
            },
            output_type=EvidenceSynthesis,
        )
        return response.output


__all__ = [
    "StructuredEvidenceSynthesisGenerator",
    "StructuredTaskSufficiencyAssessor",
]

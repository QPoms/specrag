"""Structured-model adapters for evidence research interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from specrag.models import (
    CandidateEvidence,
    EvidenceSynthesis,
    EvidenceSynthesisDraft,
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
        *,
        mode: Literal["complete", "partial"],
        covered_task_ids: tuple[str, ...],
        unresolved_task_ids: tuple[str, ...],
        conflicting_task_ids: tuple[str, ...],
    ) -> EvidenceSynthesis:
        response = self.agent.parse(
            purpose="synthesize an evidence-grounded technical answer",
            prompt_version=SYNTHESIS_PROMPT_VERSION,
            instructions=SYNTHESIS_INSTRUCTIONS,
            input_payload={
                "question": question,
                "synthesis_mode": mode,
                "covered_task_ids": covered_task_ids,
                "unresolved_task_ids": unresolved_task_ids,
                "conflicting_task_ids": conflicting_task_ids,
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
            output_type=EvidenceSynthesisDraft,
        )
        return EvidenceSynthesis(
            answer=response.output.answer,
            evidence_ids=response.output.evidence_ids,
            mode=mode,
            covered_task_ids=covered_task_ids,
            unresolved_task_ids=unresolved_task_ids,
            conflicting_task_ids=conflicting_task_ids,
        )


__all__ = [
    "StructuredEvidenceSynthesisGenerator",
    "StructuredTaskSufficiencyAssessor",
]

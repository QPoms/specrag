"""Structured offline quality assessments for generated answers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


FactStatus = Literal["covered", "partial", "missing", "contradicted"]
QualityVerdict = Literal["pass", "review", "fail"]


class FactCoverage(BaseModel):
    fact: str
    status: FactStatus
    explanation: str


class SynthesisQualityAssessment(BaseModel):
    question_id: str
    factual_completeness: int = Field(ge=0, le=4)
    evidence_faithfulness: int = Field(ge=0, le=4)
    scope_precision: int = Field(ge=0, le=4)
    clarity: int = Field(ge=0, le=4)
    fact_coverage: tuple[FactCoverage, ...]
    unsupported_claims: tuple[str, ...] = ()
    scope_issues: tuple[str, ...] = ()
    clarity_issues: tuple[str, ...] = ()
    verdict: QualityVerdict
    rationale: str

    @property
    def overall(self) -> float:
        return round(
            (
                self.factual_completeness
                + self.evidence_faithfulness
                + self.scope_precision
                + self.clarity
            )
            / 4,
            2,
        )


__all__ = [
    "FactCoverage",
    "FactStatus",
    "QualityVerdict",
    "SynthesisQualityAssessment",
]

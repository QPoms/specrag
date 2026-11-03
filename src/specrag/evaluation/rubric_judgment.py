"""Strict contracts and deterministic scoring for final-answer judgment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .benchmark_models import BenchmarkCase, BenchmarkRubric


JudgmentLevel = Literal["full", "partial", "none"]

_LEVEL_RATIOS: dict[JudgmentLevel, float] = {
    "full": 1.0,
    "partial": 0.5,
    "none": 0.0,
}

class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FinalAnswerJudgeInput(_StrictModel):
    """The complete and exclusive semantic input supplied to the Judge."""

    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    rubric: BenchmarkRubric
    candidate_answer: str = Field(min_length=1)

    @model_validator(mode="after")
    def _nonblank_text(self) -> "FinalAnswerJudgeInput":
        for field_name in ("question", "reference_answer", "candidate_answer"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        return self


class JudgeRubricAssessment(_StrictModel):
    """One provider-authored semantic assessment with no score fields."""

    rubric_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    level: JudgmentLevel
    answer_support: tuple[str, ...]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _support_matches_level(self) -> "JudgeRubricAssessment":
        if not self.reason.strip():
            raise ValueError("reason must not be blank")
        if any(not excerpt.strip() for excerpt in self.answer_support):
            raise ValueError("answer_support excerpts must not be blank")
        return self


class FinalAnswerJudgmentDraft(_StrictModel):
    """The only structured object the LLM Judge is allowed to author."""

    rubric_assessments: tuple[JudgeRubricAssessment, ...] = Field(min_length=1)
    overall_reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_draft(self) -> "FinalAnswerJudgmentDraft":
        identifiers = [item.rubric_id for item in self.rubric_assessments]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Judge rubric IDs must be unique")
        if not self.overall_reasoning.strip():
            raise ValueError("overall_reasoning must not be blank")
        return self


class ScoredRubricAssessment(_StrictModel):
    """A validated Judge assessment enriched with program-owned arithmetic."""

    rubric_id: str = Field(pattern=r"^R[1-9][0-9]*$")
    weight: int = Field(gt=0, le=100)
    level: JudgmentLevel
    score_ratio: Literal[0.0, 0.5, 1.0]
    awarded_score: float = Field(ge=0.0, le=100.0)
    answer_support: tuple[str, ...] = ()
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_score(self) -> "ScoredRubricAssessment":
        expected_ratio = _LEVEL_RATIOS[self.level]
        if self.score_ratio != expected_ratio:
            raise ValueError("score_ratio must match the assessment level")
        expected_score = round(self.weight * expected_ratio, 1)
        if self.awarded_score != expected_score:
            raise ValueError("awarded_score must equal weight * score_ratio")
        if any(not excerpt.strip() for excerpt in self.answer_support):
            raise ValueError("answer_support excerpts must not be blank")
        if not self.reason.strip():
            raise ValueError("reason must not be blank")
        return self


class FinalAnswerEvaluation(_StrictModel):
    """Auditable formal record with a deterministic 0-100 score."""

    schema_version: Literal["specrag.final-answer-evaluation/v1"] = (
        "specrag.final-answer-evaluation/v1"
    )
    question_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=100.0)
    rubric_assessments: tuple[ScoredRubricAssessment, ...] = Field(min_length=1)
    overall_reasoning: str = Field(min_length=1)
    judge_model: str | None = Field(default=None, min_length=1)
    judge_prompt_version: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _coherent_evaluation(self) -> "FinalAnswerEvaluation":
        identifiers = [item.rubric_id for item in self.rubric_assessments]
        expected = [f"R{index}" for index in range(1, len(identifiers) + 1)]
        if identifiers != expected:
            raise ValueError("Scored rubric IDs must be ordered R1..Rn")
        if sum(item.weight for item in self.rubric_assessments) != 100:
            raise ValueError("Scored rubric weights must sum to 100")
        expected_score = round(
            sum(item.awarded_score for item in self.rubric_assessments),
            1,
        )
        if self.score != expected_score:
            raise ValueError("score must equal the sum of awarded scores")
        for field_name in (
            "question_id",
            "run_id",
            "overall_reasoning",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        for field_name in ("judge_model", "judge_prompt_version"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        if (self.judge_model is None) != (self.judge_prompt_version is None):
            raise ValueError(
                "judge_model and judge_prompt_version must both be set or both be null"
            )
        return self


def build_final_answer_judge_input(
    case: BenchmarkCase,
    *,
    candidate_answer: str,
) -> FinalAnswerJudgeInput:
    """Project one Benchmark case into the four-field Judge boundary."""

    return FinalAnswerJudgeInput(
        question=case.question,
        reference_answer=case.reference_answer,
        rubric=case.rubric,
        candidate_answer=candidate_answer,
    )


def score_rubric_judgment(
    *,
    question_id: str,
    run_id: str,
    judge_input: FinalAnswerJudgeInput,
    judgment: FinalAnswerJudgmentDraft,
    judge_model: str,
    judge_prompt_version: str,
) -> FinalAnswerEvaluation:
    """Validate one Judge draft and calculate every score deterministically."""

    rubric_items = judge_input.rubric.items
    expected_ids = [item.rubric_id for item in rubric_items]
    actual_ids = [item.rubric_id for item in judgment.rubric_assessments]
    if actual_ids != expected_ids:
        raise ValueError(
            "Judge rubric assessments must contain every rubric ID once "
            "in input order"
        )

    scored: list[ScoredRubricAssessment] = []
    for rubric_item, assessment in zip(
        rubric_items,
        judgment.rubric_assessments,
        strict=True,
    ):
        ratio = _LEVEL_RATIOS[assessment.level]
        scored.append(
            ScoredRubricAssessment(
                rubric_id=assessment.rubric_id,
                weight=rubric_item.weight,
                level=assessment.level,
                score_ratio=ratio,
                awarded_score=round(rubric_item.weight * ratio, 1),
                answer_support=assessment.answer_support,
                reason=assessment.reason,
            )
        )

    score = round(sum(item.awarded_score for item in scored), 1)
    return FinalAnswerEvaluation(
        question_id=question_id,
        run_id=run_id,
        score=score,
        rubric_assessments=tuple(scored),
        overall_reasoning=judgment.overall_reasoning,
        judge_model=judge_model,
        judge_prompt_version=judge_prompt_version,
    )

__all__ = [
    "FinalAnswerEvaluation",
    "FinalAnswerJudgeInput",
    "FinalAnswerJudgmentDraft",
    "JudgeRubricAssessment",
    "JudgmentLevel",
    "ScoredRubricAssessment",
    "build_final_answer_judge_input",
    "score_rubric_judgment",
]

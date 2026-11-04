"""Deterministic descriptive analysis across compatible experiments."""

from __future__ import annotations

from itertools import combinations
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .benchmark_models import Difficulty
from .model_comparison import (
    ComparisonJudgeStatus,
    ModelComparisonSummary,
    ModelScoreSummary,
    WorkflowStatus,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CrossExperimentSource(_StrictModel):
    """One parsed source summary and its immutable file identity."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary: ModelComparisonSummary

    @model_validator(mode="after")
    def _coherent_source(self) -> "CrossExperimentSource":
        if not self.path.strip() or self.path != self.path.strip():
            raise ValueError("source path must be non-blank and trimmed")
        _validate_summary(self.summary)
        return self


class CrossExperimentSourceRecord(_StrictModel):
    """Source provenance retained in the generated analysis."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_id: str = Field(min_length=1)


class ModelObservationSummary(_StrictModel):
    """One model observation, identified by experiment and model together."""

    experiment_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    score_summary: ModelScoreSummary

    @model_validator(mode="after")
    def _matching_model(self) -> "ModelObservationSummary":
        if self.score_summary.model_id != self.model_id:
            raise ValueError("model observation and score summary IDs disagree")
        return self


class CaseObservation(_StrictModel):
    """One model's status and score for one Benchmark case."""

    experiment_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    workflow_status: WorkflowStatus
    judge_status: ComparisonJudgeStatus
    score: float | None = Field(default=None, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _coherent_score(self) -> "CaseObservation":
        if self.workflow_status == "failed":
            if self.judge_status != "not_run" or self.score is not None:
                raise ValueError("failed workflow cannot have a Judge score")
        elif self.judge_status == "scored" and self.score is None:
            raise ValueError("scored observation requires a score")
        elif self.judge_status == "no_answer" and self.score != 0.0:
            raise ValueError("no-answer observation requires score zero")
        elif self.judge_status == "evaluation_failed" and self.score is not None:
            raise ValueError("failed Judge observation cannot have a score")
        return self


class CrossExperimentCase(_StrictModel):
    """All experiment/model observations for one ordered Benchmark case."""

    question_id: str = Field(min_length=1)
    difficulty: Difficulty
    observations: tuple[CaseObservation, ...] = Field(min_length=2)


class PairedScoreSlice(_StrictModel):
    """Descriptive paired scores using the explicit right-minus-left direction."""

    expected_case_count: int = Field(ge=1)
    paired_score_count: int = Field(ge=0)
    score_coverage: float = Field(ge=0.0, le=1.0)
    left_average_score: float | None = Field(default=None, ge=0.0, le=100.0)
    right_average_score: float | None = Field(default=None, ge=0.0, le=100.0)
    mean_delta_right_minus_left: float | None = Field(
        default=None,
        ge=-100.0,
        le=100.0,
    )
    median_delta_right_minus_left: float | None = Field(
        default=None,
        ge=-100.0,
        le=100.0,
    )
    right_higher_count: int = Field(ge=0)
    equal_count: int = Field(ge=0)
    left_higher_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _coherent_slice(self) -> "PairedScoreSlice":
        if self.paired_score_count > self.expected_case_count:
            raise ValueError("paired score count exceeds expected count")
        if self.score_coverage != round(
            self.paired_score_count / self.expected_case_count, 4
        ):
            raise ValueError("paired score coverage disagrees with counts")
        compared = self.right_higher_count + self.equal_count + self.left_higher_count
        if compared != self.paired_score_count:
            raise ValueError("higher/equal/lower counts must cover paired scores")
        values = (
            self.left_average_score,
            self.right_average_score,
            self.mean_delta_right_minus_left,
            self.median_delta_right_minus_left,
        )
        if self.paired_score_count == 0 and any(item is not None for item in values):
            raise ValueError("empty paired slice cannot have score statistics")
        if self.paired_score_count > 0 and any(item is None for item in values):
            raise ValueError("non-empty paired slice requires score statistics")
        return self


class PairwiseCaseDelta(_StrictModel):
    """Statuses and score delta for one case within a model pair."""

    question_id: str = Field(min_length=1)
    difficulty: Difficulty
    left_judge_status: ComparisonJudgeStatus
    right_judge_status: ComparisonJudgeStatus
    left_score: float | None = Field(default=None, ge=0.0, le=100.0)
    right_score: float | None = Field(default=None, ge=0.0, le=100.0)
    delta_right_minus_left: float | None = Field(
        default=None,
        ge=-100.0,
        le=100.0,
    )

    @model_validator(mode="after")
    def _coherent_delta(self) -> "PairwiseCaseDelta":
        expected = (
            round(self.right_score - self.left_score, 3)
            if self.left_score is not None and self.right_score is not None
            else None
        )
        if self.delta_right_minus_left != expected:
            raise ValueError("case delta disagrees with paired scores")
        return self


class PairwiseModelComparison(_StrictModel):
    """One ordered pair; all deltas are right minus left."""

    left_experiment_id: str = Field(min_length=1)
    left_model_id: str = Field(min_length=1)
    right_experiment_id: str = Field(min_length=1)
    right_model_id: str = Field(min_length=1)
    overall: PairedScoreSlice
    simple: PairedScoreSlice
    complex: PairedScoreSlice
    answered_only_overall: PairedScoreSlice
    answered_only_simple: PairedScoreSlice
    answered_only_complex: PairedScoreSlice
    case_deltas: tuple[PairwiseCaseDelta, ...] = Field(min_length=1)


class CrossExperimentAnalysis(_StrictModel):
    """Reproducible one-trial descriptive comparison across experiments."""

    schema_version: Literal["specrag.cross-experiment-analysis/v1"] = (
        "specrag.cross-experiment-analysis/v1"
    )
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    judge_prompt_version: str = Field(min_length=1)
    trial_count: Literal[1] = 1
    sources: tuple[CrossExperimentSourceRecord, ...] = Field(min_length=2)
    model_observations: tuple[ModelObservationSummary, ...] = Field(min_length=2)
    pairwise_comparisons: tuple[PairwiseModelComparison, ...] = Field(min_length=1)
    cases: tuple[CrossExperimentCase, ...] = Field(min_length=1)


def _validate_summary(summary: ModelComparisonSummary) -> None:
    model_ids = tuple(item.model_id for item in summary.model_summaries)
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise ValueError("source model summaries must contain unique model IDs")
    question_ids = tuple(item.question_id for item in summary.case_summaries)
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("source case summaries must contain unique question IDs")
    if {item.difficulty for item in summary.case_summaries} != {"simple", "complex"}:
        raise ValueError("source cases must include simple and complex difficulties")
    for case in summary.case_summaries:
        case_model_ids = tuple(item.model_id for item in case.model_results)
        if case_model_ids != model_ids:
            raise ValueError("case model order must match model summary order")
    computed_common = tuple(
        case.question_id
        for case in summary.case_summaries
        if all(item.score is not None for item in case.model_results)
    )
    if computed_common != summary.common_question_ids:
        raise ValueError("source common question IDs disagree with case scores")
    difficulty_by_question = {
        item.question_id: item.difficulty for item in summary.case_summaries
    }
    for model_summary in summary.model_summaries:
        results = tuple(
            next(
                result
                for result in case.model_results
                if result.model_id == model_summary.model_id
            )
            for case in summary.case_summaries
        )
        if model_summary.expected_case_count != len(results):
            raise ValueError("source model expected-case count disagrees with cases")
        outcome_counts = (
            sum(item.judge_status == "scored" for item in results),
            sum(item.judge_status == "no_answer" for item in results),
            sum(item.workflow_status == "failed" for item in results),
            sum(item.judge_status == "evaluation_failed" for item in results),
        )
        recorded_counts = (
            model_summary.scored_answer_count,
            model_summary.no_answer_count,
            model_summary.workflow_failed_count,
            model_summary.judge_failed_count,
        )
        if outcome_counts != recorded_counts:
            raise ValueError("source model outcome counts disagree with cases")
        attempt_counts = (
            sum(item.workflow_attempt_count for item in results),
            sum(item.workflow_retry_count for item in results),
            sum(item.workflow_retry_count > 0 for item in results),
        )
        recorded_attempt_counts = (
            model_summary.workflow_attempt_count,
            model_summary.workflow_retry_count,
            model_summary.retried_case_count,
        )
        if attempt_counts != recorded_attempt_counts:
            raise ValueError("source model workflow attempts disagree with cases")
        scores = tuple(float(item.score) for item in results if item.score is not None)
        expected_average = round(sum(scores) / len(scores), 1) if scores else None
        if (
            model_summary.overall.observed_score_count != len(scores)
            or model_summary.overall.observed_average_score != expected_average
        ):
            raise ValueError("source model overall scores disagree with cases")
        for difficulty, recorded_slice in (
            ("simple", model_summary.simple),
            ("complex", model_summary.complex),
        ):
            difficulty_results = tuple(
                item
                for item in results
                if difficulty_by_question[item.question_id] == difficulty
            )
            difficulty_scores = tuple(
                float(item.score)
                for item in difficulty_results
                if item.score is not None
            )
            difficulty_average = (
                round(sum(difficulty_scores) / len(difficulty_scores), 1)
                if difficulty_scores
                else None
            )
            if (
                recorded_slice.expected_count != len(difficulty_results)
                or recorded_slice.observed_score_count != len(difficulty_scores)
                or recorded_slice.observed_average_score != difficulty_average
            ):
                raise ValueError(
                    f"source model {difficulty} scores disagree with cases"
                )
        common_scores = tuple(
            float(item.score)
            for item in results
            if item.question_id in computed_common and item.score is not None
        )
        common_average = (
            round(sum(common_scores) / len(common_scores), 1)
            if common_scores
            else None
        )
        if (
            model_summary.common_case_count != len(common_scores)
            or model_summary.common_case_average_score != common_average
        ):
            raise ValueError("source model common-case scores disagree with cases")


def _paired_slice(
    items: tuple[PairwiseCaseDelta, ...],
    *,
    expected_count: int | None = None,
) -> PairedScoreSlice:
    paired = tuple(item for item in items if item.delta_right_minus_left is not None)
    expected = expected_count if expected_count is not None else len(items)
    left = tuple(
        float(item.left_score) for item in paired if item.left_score is not None
    )
    right = tuple(
        float(item.right_score) for item in paired if item.right_score is not None
    )
    deltas = tuple(
        float(item.delta_right_minus_left)
        for item in paired
        if item.delta_right_minus_left is not None
    )
    return PairedScoreSlice(
        expected_case_count=expected,
        paired_score_count=len(paired),
        score_coverage=round(len(paired) / expected, 4),
        left_average_score=round(sum(left) / len(left), 3) if left else None,
        right_average_score=round(sum(right) / len(right), 3) if right else None,
        mean_delta_right_minus_left=(
            round(sum(deltas) / len(deltas), 3) if deltas else None
        ),
        median_delta_right_minus_left=(round(median(deltas), 3) if deltas else None),
        right_higher_count=sum(item > 0 for item in deltas),
        equal_count=sum(item == 0 for item in deltas),
        left_higher_count=sum(item < 0 for item in deltas),
    )


def analyze_cross_experiments(
    sources: tuple[CrossExperimentSource, ...],
) -> CrossExperimentAnalysis:
    """Validate and descriptively compare two or more experiment summaries."""

    if len(sources) < 2:
        raise ValueError("cross-experiment analysis requires at least two sources")
    experiment_ids = tuple(item.summary.experiment_id for item in sources)
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("cross-experiment source IDs must be unique")

    reference = sources[0].summary
    reference_cases = tuple(
        (item.question_id, item.difficulty) for item in reference.case_summaries
    )
    compatibility_fields = (
        "benchmark_id",
        "benchmark_version",
        "judge_model",
        "judge_prompt_version",
    )
    for source in sources[1:]:
        summary = source.summary
        for field_name in compatibility_fields:
            if getattr(summary, field_name) != getattr(reference, field_name):
                raise ValueError(f"incompatible cross-experiment {field_name}")
        cases = tuple(
            (item.question_id, item.difficulty) for item in summary.case_summaries
        )
        if cases != reference_cases:
            raise ValueError(
                "cross-experiment question order and difficulty must match"
            )

    observations: list[ModelObservationSummary] = []
    result_maps: dict[tuple[str, str], dict[str, CaseObservation]] = {}
    for source in sources:
        summary = source.summary
        case_by_question = {
            item.question_id: item for item in summary.case_summaries
        }
        for model_summary in summary.model_summaries:
            identity = (summary.experiment_id, model_summary.model_id)
            observations.append(
                ModelObservationSummary(
                    experiment_id=identity[0],
                    model_id=identity[1],
                    score_summary=model_summary,
                )
            )
            result_maps[identity] = {
                question_id: CaseObservation(
                    experiment_id=identity[0],
                    model_id=identity[1],
                    workflow_status=next(
                        item.workflow_status
                        for item in case_by_question[question_id].model_results
                        if item.model_id == identity[1]
                    ),
                    judge_status=next(
                        item.judge_status
                        for item in case_by_question[question_id].model_results
                        if item.model_id == identity[1]
                    ),
                    score=next(
                        item.score
                        for item in case_by_question[question_id].model_results
                        if item.model_id == identity[1]
                    ),
                )
                for question_id, _ in reference_cases
            }

    identities = tuple((item.experiment_id, item.model_id) for item in observations)
    cases = tuple(
        CrossExperimentCase(
            question_id=question_id,
            difficulty=difficulty,
            observations=tuple(
                result_maps[identity][question_id] for identity in identities
            ),
        )
        for question_id, difficulty in reference_cases
    )
    pairs: list[PairwiseModelComparison] = []
    for left, right in combinations(identities, 2):
        deltas = tuple(
            PairwiseCaseDelta(
                question_id=case.question_id,
                difficulty=case.difficulty,
                left_judge_status=result_maps[left][case.question_id].judge_status,
                right_judge_status=result_maps[right][case.question_id].judge_status,
                left_score=result_maps[left][case.question_id].score,
                right_score=result_maps[right][case.question_id].score,
                delta_right_minus_left=(
                    round(
                        float(result_maps[right][case.question_id].score)
                        - float(result_maps[left][case.question_id].score),
                        3,
                    )
                    if result_maps[left][case.question_id].score is not None
                    and result_maps[right][case.question_id].score is not None
                    else None
                ),
            )
            for case in cases
        )
        pairs.append(
            PairwiseModelComparison(
                left_experiment_id=left[0],
                left_model_id=left[1],
                right_experiment_id=right[0],
                right_model_id=right[1],
                overall=_paired_slice(deltas),
                simple=_paired_slice(
                    tuple(item for item in deltas if item.difficulty == "simple")
                ),
                complex=_paired_slice(
                    tuple(item for item in deltas if item.difficulty == "complex")
                ),
                answered_only_overall=_paired_slice(
                    tuple(
                        item
                        for item in deltas
                        if item.left_judge_status == "scored"
                        and item.right_judge_status == "scored"
                    ),
                    expected_count=len(deltas),
                ),
                answered_only_simple=_paired_slice(
                    tuple(
                        item
                        for item in deltas
                        if item.difficulty == "simple"
                        and item.left_judge_status == "scored"
                        and item.right_judge_status == "scored"
                    ),
                    expected_count=sum(
                        item.difficulty == "simple" for item in deltas
                    ),
                ),
                answered_only_complex=_paired_slice(
                    tuple(
                        item
                        for item in deltas
                        if item.difficulty == "complex"
                        and item.left_judge_status == "scored"
                        and item.right_judge_status == "scored"
                    ),
                    expected_count=sum(
                        item.difficulty == "complex" for item in deltas
                    ),
                ),
                case_deltas=deltas,
            )
        )

    return CrossExperimentAnalysis(
        benchmark_id=reference.benchmark_id,
        benchmark_version=reference.benchmark_version,
        judge_model=reference.judge_model,
        judge_prompt_version=reference.judge_prompt_version,
        sources=tuple(
            CrossExperimentSourceRecord(
                path=item.path,
                sha256=item.sha256,
                experiment_id=item.summary.experiment_id,
            )
            for item in sources
        ),
        model_observations=tuple(observations),
        pairwise_comparisons=tuple(pairs),
        cases=cases,
    )


__all__ = [
    "CaseObservation",
    "CrossExperimentAnalysis",
    "CrossExperimentCase",
    "CrossExperimentSource",
    "CrossExperimentSourceRecord",
    "ModelObservationSummary",
    "PairedScoreSlice",
    "PairwiseCaseDelta",
    "PairwiseModelComparison",
    "analyze_cross_experiments",
]

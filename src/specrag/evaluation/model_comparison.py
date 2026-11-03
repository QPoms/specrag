"""Provider-neutral contracts and aggregation for one-trial model comparisons."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .benchmark_models import Difficulty


ProbeStatus = Literal["ready", "incompatible", "unavailable"]
WorkflowStatus = Literal["completed", "failed"]
ComparisonJudgeStatus = Literal[
    "scored",
    "no_answer",
    "evaluation_failed",
    "not_run",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelTarget(_StrictModel):
    """One requested workflow model and whether its participation is mandatory."""

    model_id: str = Field(min_length=1)
    required: bool

    @model_validator(mode="after")
    def _nonblank_model(self) -> "ModelTarget":
        if not self.model_id.strip() or self.model_id != self.model_id.strip():
            raise ValueError("model_id must be non-blank and trimmed")
        return self


class ComparisonCase(_StrictModel):
    """The evaluator-visible identity and reporting class of one case."""

    question_id: str = Field(min_length=1)
    difficulty: Difficulty

    @model_validator(mode="after")
    def _nonblank_question(self) -> "ComparisonCase":
        if (
            not self.question_id.strip()
            or self.question_id != self.question_id.strip()
        ):
            raise ValueError("question_id must be non-blank and trimmed")
        return self


class ModelProbeResult(_StrictModel):
    """Secret-free result of testing one model against the workflow contract."""

    schema_version: Literal["specrag.model-probe-result/v1"] = (
        "specrag.model-probe-result/v1"
    )
    model_id: str = Field(min_length=1)
    status: ProbeStatus
    observed_model: str | None = Field(default=None, min_length=1)
    failure_kind: str | None = Field(default=None, min_length=1)
    message: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_probe(self) -> "ModelProbeResult":
        for field_name in ("model_id", "message"):
            value = getattr(self, field_name)
            if not value.strip() or value != value.strip():
                raise ValueError(f"{field_name} must be non-blank and trimmed")
        if self.status == "ready":
            if self.observed_model is None or not self.observed_model.strip():
                raise ValueError("ready probe requires observed_model")
            if self.observed_model != self.observed_model.strip():
                raise ValueError("observed_model must be trimmed")
            if self.observed_model != self.model_id:
                raise ValueError("ready probe observed a different model")
            if self.failure_kind is not None:
                raise ValueError("ready probe cannot contain failure_kind")
        else:
            if self.failure_kind is None or not self.failure_kind.strip():
                raise ValueError("non-ready probe requires failure_kind")
            if self.failure_kind != self.failure_kind.strip():
                raise ValueError("failure_kind must be trimmed")
            if self.observed_model is not None and (
                not self.observed_model.strip()
                or self.observed_model != self.observed_model.strip()
            ):
                raise ValueError("observed_model must be non-blank and trimmed")
        return self


class ModelComparisonExperimentManifest(_StrictModel):
    """Frozen evaluator input shared by workflow and Judge artifact producers."""

    schema_version: Literal["specrag.model-comparison-experiment/v1"] = (
        "specrag.model-comparison-experiment/v1"
    )
    experiment_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    judge_prompt_version: str = Field(min_length=1)
    trial_count: Literal[1] = 1
    cases: tuple[ComparisonCase, ...] = Field(min_length=1)
    model_targets: tuple[ModelTarget, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_manifest(self) -> "ModelComparisonExperimentManifest":
        for field_name in (
            "experiment_id",
            "benchmark_id",
            "benchmark_version",
            "judge_model",
            "judge_prompt_version",
        ):
            value = getattr(self, field_name)
            if not value.strip() or value != value.strip():
                raise ValueError(f"{field_name} must be non-blank and trimmed")
        case_ids = [item.question_id for item in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("comparison case question IDs must be unique")
        difficulties = {item.difficulty for item in self.cases}
        if difficulties != {"simple", "complex"}:
            raise ValueError("comparison cases must include simple and complex cases")
        model_ids = [item.model_id for item in self.model_targets]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("comparison model IDs must be unique")
        if not any(item.required for item in self.model_targets):
            raise ValueError("comparison requires at least one required model")
        return self


class ModelCaseResult(_StrictModel):
    """One workflow/Judge outcome without a success threshold."""

    schema_version: Literal["specrag.model-case-result/v1"] = (
        "specrag.model-case-result/v1"
    )
    model_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    trial: Literal[1] = 1
    workflow_status: WorkflowStatus
    judge_status: ComparisonJudgeStatus
    score: float | None = Field(default=None, ge=0.0, le=100.0)
    workflow_attempt_count: int = Field(default=1, ge=1)
    workflow_retry_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _coherent_result(self) -> "ModelCaseResult":
        identifiers = (self.model_id, self.question_id)
        if any(not value.strip() or value != value.strip() for value in identifiers):
            raise ValueError("model_id and question_id must be non-blank and trimmed")
        if self.workflow_retry_count != self.workflow_attempt_count - 1:
            raise ValueError("workflow retry count must equal attempt count minus one")
        if self.workflow_status == "failed":
            if self.judge_status != "not_run" or self.score is not None:
                raise ValueError(
                    "failed workflow requires judge_status=not_run and no score"
                )
            return self
        if self.judge_status == "not_run":
            raise ValueError("completed workflow requires a Judge outcome")
        if self.judge_status == "scored" and self.score is None:
            raise ValueError("scored Judge outcome requires a score")
        if self.judge_status == "no_answer" and self.score != 0.0:
            raise ValueError("no-answer Judge outcome requires score zero")
        if self.judge_status == "evaluation_failed" and self.score is not None:
            raise ValueError("failed Judge outcome cannot contain a score")
        return self


class ModelComparisonInput(_StrictModel):
    """Complete provider-neutral input assembled from generated artifacts."""

    schema_version: Literal["specrag.model-comparison-input/v1"] = (
        "specrag.model-comparison-input/v1"
    )
    manifest: ModelComparisonExperimentManifest
    probes: tuple[ModelProbeResult, ...]
    results: tuple[ModelCaseResult, ...] = ()

    @model_validator(mode="after")
    def _known_and_unique_records(self) -> "ModelComparisonInput":
        target_ids = tuple(item.model_id for item in self.manifest.model_targets)
        probe_ids = tuple(item.model_id for item in self.probes)
        if probe_ids != target_ids:
            raise ValueError("probe results must match model targets in manifest order")

        case_ids = {item.question_id for item in self.manifest.cases}
        probe_by_model = {item.model_id: item for item in self.probes}
        seen: set[tuple[str, str]] = set()
        for result in self.results:
            key = (result.model_id, result.question_id)
            if key in seen:
                raise ValueError(f"duplicate model/case result: {key}")
            seen.add(key)
            if result.model_id not in probe_by_model:
                raise ValueError(f"result references unknown model: {result.model_id}")
            if result.question_id not in case_ids:
                raise ValueError(
                    f"result references unknown question: {result.question_id}"
                )
            if probe_by_model[result.model_id].status != "ready":
                raise ValueError(
                    f"non-ready model cannot contain case results: {result.model_id}"
                )
        return self


class ScoreSliceSummary(_StrictModel):
    """Observed and complete-batch score views for one fixed case slice."""

    expected_count: int = Field(ge=1)
    observed_score_count: int = Field(ge=0)
    score_coverage: float = Field(ge=0.0, le=1.0)
    observed_average_score: float | None = Field(default=None, ge=0.0, le=100.0)
    complete_batch_average_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )

    @model_validator(mode="after")
    def _coherent_slice(self) -> "ScoreSliceSummary":
        if self.observed_score_count > self.expected_count:
            raise ValueError("observed score count exceeds expected count")
        expected_coverage = round(
            self.observed_score_count / self.expected_count,
            4,
        )
        if self.score_coverage != expected_coverage:
            raise ValueError("score coverage disagrees with observed count")
        if (self.observed_score_count == 0) != (
            self.observed_average_score is None
        ):
            raise ValueError("observed average presence disagrees with score count")
        complete = self.observed_score_count == self.expected_count
        if complete:
            if self.complete_batch_average_score != self.observed_average_score:
                raise ValueError(
                    "complete-batch average must equal the observed average"
                )
        elif self.complete_batch_average_score is not None:
            raise ValueError("incomplete score slice cannot have a complete average")
        return self


class ModelScoreSummary(_StrictModel):
    """Score coverage and failure counts for one ready model."""

    model_id: str = Field(min_length=1)
    expected_case_count: int = Field(ge=1)
    scored_answer_count: int = Field(ge=0)
    no_answer_count: int = Field(ge=0)
    workflow_failed_count: int = Field(ge=0)
    judge_failed_count: int = Field(ge=0)
    workflow_attempt_count: int = Field(ge=1)
    workflow_retry_count: int = Field(ge=0)
    retried_case_count: int = Field(ge=0)
    overall: ScoreSliceSummary
    simple: ScoreSliceSummary
    complex: ScoreSliceSummary
    common_case_count: int = Field(ge=0)
    common_case_average_score: float | None = Field(default=None, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _coherent_model_summary(self) -> "ModelScoreSummary":
        categorized = (
            self.scored_answer_count
            + self.no_answer_count
            + self.workflow_failed_count
            + self.judge_failed_count
        )
        if categorized != self.expected_case_count:
            raise ValueError("model outcome counts must cover every expected case")
        if self.workflow_attempt_count != (
            self.expected_case_count + self.workflow_retry_count
        ):
            raise ValueError("workflow attempts must equal cases plus retries")
        if self.retried_case_count > self.expected_case_count:
            raise ValueError("retried case count exceeds expected cases")
        if self.retried_case_count > self.workflow_retry_count:
            raise ValueError("retried cases cannot exceed retry attempts")
        if self.overall.expected_count != self.expected_case_count:
            raise ValueError("overall slice must cover every expected case")
        if self.simple.expected_count + self.complex.expected_count != (
            self.expected_case_count
        ):
            raise ValueError("difficulty slices must partition expected cases")
        if self.common_case_count > self.overall.observed_score_count:
            raise ValueError("common cases must be observed for the model")
        if (self.common_case_count == 0) != (
            self.common_case_average_score is None
        ):
            raise ValueError("common-case average presence disagrees with count")
        return self


class CaseComparisonSummary(_StrictModel):
    """Ordered per-model outcomes for one Benchmark question."""

    question_id: str = Field(min_length=1)
    difficulty: Difficulty
    model_results: tuple[ModelCaseResult, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_case(self) -> "CaseComparisonSummary":
        if (
            not self.question_id.strip()
            or self.question_id != self.question_id.strip()
        ):
            raise ValueError("question_id must be non-blank and trimmed")
        model_ids = [item.model_id for item in self.model_results]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("case comparison model IDs must be unique")
        if any(item.question_id != self.question_id for item in self.model_results):
            raise ValueError("case comparison results must share one question ID")
        return self


class ModelComparisonSummary(_StrictModel):
    """Cross-model report over complete experiment artifacts."""

    schema_version: Literal["specrag.model-comparison-summary/v1"] = (
        "specrag.model-comparison-summary/v1"
    )
    experiment_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    judge_prompt_version: str = Field(min_length=1)
    probes: tuple[ModelProbeResult, ...]
    skipped_model_ids: tuple[str, ...]
    common_question_ids: tuple[str, ...]
    model_summaries: tuple[ModelScoreSummary, ...]
    case_summaries: tuple[CaseComparisonSummary, ...]


def validate_comparison_completeness(value: ModelComparisonInput) -> None:
    """Reject missing required models or missing cases for any ready model."""

    probes = {item.model_id: item for item in value.probes}
    unavailable_required = [
        target.model_id
        for target in value.manifest.model_targets
        if target.required and probes[target.model_id].status != "ready"
    ]
    if unavailable_required:
        raise ValueError(
            "required comparison models are not ready: "
            + ", ".join(unavailable_required)
        )

    expected = {item.question_id for item in value.manifest.cases}
    results_by_model: dict[str, set[str]] = {}
    for item in value.results:
        results_by_model.setdefault(item.model_id, set()).add(item.question_id)
    missing_details: list[str] = []
    for target in value.manifest.model_targets:
        if probes[target.model_id].status != "ready":
            continue
        missing = sorted(expected - results_by_model.get(target.model_id, set()))
        if missing:
            missing_details.append(f"{target.model_id}: {', '.join(missing)}")
    if missing_details:
        raise ValueError(
            "comparison results are incomplete (" + "; ".join(missing_details) + ")"
        )


def _score_slice(
    results: tuple[ModelCaseResult, ...],
    *,
    expected_count: int,
) -> ScoreSliceSummary:
    scores = tuple(float(item.score) for item in results if item.score is not None)
    average = round(sum(scores) / len(scores), 1) if scores else None
    complete_average = average if len(scores) == expected_count else None
    return ScoreSliceSummary(
        expected_count=expected_count,
        observed_score_count=len(scores),
        score_coverage=round(len(scores) / expected_count, 4),
        observed_average_score=average,
        complete_batch_average_score=complete_average,
    )


def summarize_model_comparison(
    value: ModelComparisonInput,
) -> ModelComparisonSummary:
    """Aggregate comparable Judge scores without deriving a success decision."""

    validate_comparison_completeness(value)
    probes = {item.model_id: item for item in value.probes}
    case_by_id = {item.question_id: item for item in value.manifest.cases}
    result_by_key = {
        (item.model_id, item.question_id): item for item in value.results
    }
    ready_model_ids = tuple(
        target.model_id
        for target in value.manifest.model_targets
        if probes[target.model_id].status == "ready"
    )
    common_question_ids = tuple(
        case.question_id
        for case in value.manifest.cases
        if all(
            result_by_key[(model_id, case.question_id)].score is not None
            for model_id in ready_model_ids
        )
    )

    summaries: list[ModelScoreSummary] = []
    for model_id in ready_model_ids:
        results = tuple(
            result_by_key[(model_id, case.question_id)]
            for case in value.manifest.cases
        )
        simple = tuple(
            item
            for item in results
            if case_by_id[item.question_id].difficulty == "simple"
        )
        complex_items = tuple(
            item
            for item in results
            if case_by_id[item.question_id].difficulty == "complex"
        )
        common_scores = tuple(
            float(result_by_key[(model_id, question_id)].score)
            for question_id in common_question_ids
        )
        summaries.append(ModelScoreSummary(
            model_id=model_id,
            expected_case_count=len(results),
            scored_answer_count=sum(
                item.judge_status == "scored" for item in results
            ),
            no_answer_count=sum(
                item.judge_status == "no_answer" for item in results
            ),
            workflow_failed_count=sum(
                item.workflow_status == "failed" for item in results
            ),
            judge_failed_count=sum(
                item.judge_status == "evaluation_failed" for item in results
            ),
            workflow_attempt_count=sum(
                item.workflow_attempt_count for item in results
            ),
            workflow_retry_count=sum(
                item.workflow_retry_count for item in results
            ),
            retried_case_count=sum(
                item.workflow_retry_count > 0 for item in results
            ),
            overall=_score_slice(results, expected_count=len(results)),
            simple=_score_slice(simple, expected_count=len(simple)),
            complex=_score_slice(
                complex_items,
                expected_count=len(complex_items),
            ),
            common_case_count=len(common_question_ids),
            common_case_average_score=(
                round(sum(common_scores) / len(common_scores), 1)
                if common_scores else None
            ),
        ))

    case_summaries = tuple(
        CaseComparisonSummary(
            question_id=case.question_id,
            difficulty=case.difficulty,
            model_results=tuple(
                result_by_key[(model_id, case.question_id)]
                for model_id in ready_model_ids
            ),
        )
        for case in value.manifest.cases
    )

    return ModelComparisonSummary(
        experiment_id=value.manifest.experiment_id,
        benchmark_id=value.manifest.benchmark_id,
        benchmark_version=value.manifest.benchmark_version,
        judge_model=value.manifest.judge_model,
        judge_prompt_version=value.manifest.judge_prompt_version,
        probes=value.probes,
        skipped_model_ids=tuple(
            target.model_id
            for target in value.manifest.model_targets
            if probes[target.model_id].status != "ready"
        ),
        common_question_ids=common_question_ids,
        model_summaries=tuple(summaries),
        case_summaries=case_summaries,
    )


__all__ = [
    "CaseComparisonSummary",
    "ComparisonCase",
    "ComparisonJudgeStatus",
    "ModelCaseResult",
    "ModelComparisonExperimentManifest",
    "ModelComparisonInput",
    "ModelComparisonSummary",
    "ModelProbeResult",
    "ModelScoreSummary",
    "ModelTarget",
    "ProbeStatus",
    "ScoreSliceSummary",
    "WorkflowStatus",
    "summarize_model_comparison",
    "validate_comparison_completeness",
]

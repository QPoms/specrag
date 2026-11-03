from __future__ import annotations

import pytest
from pydantic import ValidationError

from specrag.evaluation import (
    ComparisonCase,
    ModelCaseResult,
    ModelComparisonExperimentManifest,
    ModelComparisonInput,
    ModelProbeResult,
    ModelTarget,
    summarize_model_comparison,
    validate_comparison_completeness,
)


def _manifest() -> ModelComparisonExperimentManifest:
    return ModelComparisonExperimentManifest(
        experiment_id="model-comparison-20261103",
        benchmark_id="specrag-release17-final-answer",
        benchmark_version="1.0.0",
        judge_model="deepseek-v4-pro",
        judge_prompt_version="final-answer-rubric-judge/v1.1",
        cases=(
            ComparisonCase(question_id="r17-qa-001", difficulty="simple"),
            ComparisonCase(question_id="r17-qa-002", difficulty="simple"),
            ComparisonCase(question_id="r17-qa-003", difficulty="complex"),
        ),
        model_targets=(
            ModelTarget(model_id="gpt-5.6-sol", required=True),
            ModelTarget(model_id="deepseek-v4.1-flash", required=True),
            ModelTarget(model_id="claude-opus-5", required=False),
        ),
    )


def _probes() -> tuple[ModelProbeResult, ...]:
    return (
        ModelProbeResult(
            model_id="gpt-5.6-sol",
            status="ready",
            observed_model="gpt-5.6-sol",
            message="The model passed the workflow capability probe.",
        ),
        ModelProbeResult(
            model_id="deepseek-v4.1-flash",
            status="ready",
            observed_model="deepseek-v4.1-flash",
            message="The model passed the workflow capability probe.",
        ),
        ModelProbeResult(
            model_id="claude-opus-5",
            status="unavailable",
            failure_kind="model_not_found",
            message="The configured gateway does not expose this optional model.",
        ),
    )


def _result(
    model_id: str,
    question_id: str,
    *,
    workflow_status: str = "completed",
    judge_status: str = "scored",
    score: float | None = 80.0,
) -> ModelCaseResult:
    return ModelCaseResult(
        model_id=model_id,
        question_id=question_id,
        workflow_status=workflow_status,
        judge_status=judge_status,
        score=score,
    )


def test_case_result_enforces_workflow_and_judge_outcome_contract() -> None:
    with pytest.raises(ValidationError, match="judge_status=not_run"):
        _result(
            "gpt-5.6-sol",
            "r17-qa-001",
            workflow_status="failed",
            judge_status="scored",
        )

    with pytest.raises(ValidationError, match="score zero"):
        _result(
            "gpt-5.6-sol",
            "r17-qa-001",
            judge_status="no_answer",
            score=1.0,
        )

    with pytest.raises(ValidationError, match="cannot contain a score"):
        _result(
            "gpt-5.6-sol",
            "r17-qa-001",
            judge_status="evaluation_failed",
            score=50.0,
        )


def test_input_rejects_results_for_non_ready_model() -> None:
    with pytest.raises(ValidationError, match="non-ready model"):
        ModelComparisonInput(
            manifest=_manifest(),
            probes=_probes(),
            results=(
                _result("claude-opus-5", "r17-qa-001"),
            ),
        )


def test_probe_rejects_model_alias_or_gateway_substitution() -> None:
    with pytest.raises(ValidationError, match="observed a different model"):
        ModelProbeResult(
            model_id="deepseek-v4.1-flash",
            status="ready",
            observed_model="gpt-5.6-sol",
            message="The gateway substituted another model.",
        )


def test_identifiers_are_trimmed_before_they_enter_the_frozen_contract() -> None:
    with pytest.raises(ValidationError, match="trimmed"):
        ModelTarget(model_id=" gpt-5.6-sol", required=True)
    with pytest.raises(ValidationError, match="trimmed"):
        ComparisonCase(question_id="r17-qa-001 ", difficulty="simple")


def test_completeness_requires_every_case_for_each_ready_model() -> None:
    value = ModelComparisonInput(
        manifest=_manifest(),
        probes=_probes(),
        results=(
            _result("gpt-5.6-sol", "r17-qa-001"),
            _result("deepseek-v4.1-flash", "r17-qa-001"),
        ),
    )

    with pytest.raises(ValueError, match="results are incomplete") as exc_info:
        validate_comparison_completeness(value)

    assert "r17-qa-002" in str(exc_info.value)
    assert "r17-qa-003" in str(exc_info.value)


def test_completeness_rejects_a_required_non_ready_model() -> None:
    probes = list(_probes())
    probes[1] = ModelProbeResult(
        model_id="deepseek-v4.1-flash",
        status="incompatible",
        failure_kind="native_tool_calling",
        message="The model rejected the required retrieval tool contract.",
    )
    value = ModelComparisonInput(
        manifest=_manifest(),
        probes=tuple(probes),
    )

    with pytest.raises(ValueError, match="required comparison models"):
        validate_comparison_completeness(value)


def test_summary_reports_observed_complete_and_common_case_averages() -> None:
    value = ModelComparisonInput(
        manifest=_manifest(),
        probes=_probes(),
        results=(
            _result("gpt-5.6-sol", "r17-qa-001", score=80.0),
            _result(
                "gpt-5.6-sol",
                "r17-qa-002",
                judge_status="no_answer",
                score=0.0,
            ),
            _result(
                "gpt-5.6-sol",
                "r17-qa-003",
                workflow_status="failed",
                judge_status="not_run",
                score=None,
            ),
            _result("deepseek-v4.1-flash", "r17-qa-001", score=60.0),
            _result(
                "deepseek-v4.1-flash",
                "r17-qa-002",
                judge_status="evaluation_failed",
                score=None,
            ),
            _result("deepseek-v4.1-flash", "r17-qa-003", score=90.0),
        ),
    )

    summary = summarize_model_comparison(value)

    assert summary.skipped_model_ids == ("claude-opus-5",)
    assert summary.common_question_ids == ("r17-qa-001",)
    gpt, deepseek = summary.model_summaries
    assert gpt.model_id == "gpt-5.6-sol"
    assert gpt.workflow_failed_count == 1
    assert gpt.judge_failed_count == 0
    assert gpt.no_answer_count == 1
    assert gpt.overall.observed_score_count == 2
    assert gpt.overall.score_coverage == 0.6667
    assert gpt.overall.observed_average_score == 40.0
    assert gpt.overall.complete_batch_average_score is None
    assert gpt.simple.complete_batch_average_score == 40.0
    assert gpt.complex.observed_average_score is None
    assert gpt.common_case_average_score == 80.0

    assert deepseek.workflow_failed_count == 0
    assert deepseek.judge_failed_count == 1
    assert deepseek.no_answer_count == 0
    assert deepseek.overall.observed_average_score == 75.0
    assert deepseek.simple.observed_average_score == 60.0
    assert deepseek.simple.complete_batch_average_score is None
    assert deepseek.complex.complete_batch_average_score == 90.0
    assert deepseek.common_case_average_score == 60.0

    first_case = summary.case_summaries[0]
    assert first_case.question_id == "r17-qa-001"
    assert first_case.difficulty == "simple"
    assert tuple(item.model_id for item in first_case.model_results) == (
        "gpt-5.6-sol",
        "deepseek-v4.1-flash",
    )
    assert tuple(item.score for item in first_case.model_results) == (80.0, 60.0)


def test_serialized_summary_has_no_success_pass_or_threshold_metrics() -> None:
    manifest = _manifest().model_copy(update={
        "model_targets": (ModelTarget(model_id="gpt-5.6-sol", required=True),),
    })
    value = ModelComparisonInput(
        manifest=manifest,
        probes=(_probes()[0],),
        results=tuple(
            _result("gpt-5.6-sol", case.question_id)
            for case in manifest.cases
        ),
    )

    payload = summarize_model_comparison(value).model_dump(mode="json")
    serialized = str(payload).lower()

    assert "task_success" not in serialized
    assert "pass@" not in serialized
    assert "pass^" not in serialized
    assert "threshold" not in serialized


def test_complete_model_batch_exposes_complete_average() -> None:
    manifest = _manifest().model_copy(update={
        "model_targets": (ModelTarget(model_id="gpt-5.6-sol", required=True),),
    })
    probes = (_probes()[0],)
    value = ModelComparisonInput(
        manifest=manifest,
        probes=probes,
        results=(
            _result("gpt-5.6-sol", "r17-qa-001", score=70.0),
            _result("gpt-5.6-sol", "r17-qa-002", score=80.0),
            _result("gpt-5.6-sol", "r17-qa-003", score=90.0),
        ),
    )

    model = summarize_model_comparison(value).model_summaries[0]

    assert model.overall.score_coverage == 1.0
    assert model.overall.observed_average_score == 80.0
    assert model.overall.complete_batch_average_score == 80.0
    assert model.common_case_count == 3
    assert model.common_case_average_score == 80.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate_model_comparison import analyze_experiments
from specrag.evaluation import (
    ComparisonCase,
    CrossExperimentSource,
    ModelCaseResult,
    ModelComparisonExperimentManifest,
    ModelComparisonInput,
    ModelComparisonSummary,
    ModelProbeResult,
    ModelTarget,
    analyze_cross_experiments,
    render_cross_experiment_markdown,
    summarize_model_comparison,
)


CASES = (
    ComparisonCase(question_id="q1", difficulty="simple"),
    ComparisonCase(question_id="q2", difficulty="simple"),
    ComparisonCase(question_id="q3", difficulty="complex"),
    ComparisonCase(question_id="q4", difficulty="complex"),
)


def _summary(
    *,
    experiment_id: str,
    model_id: str,
    outcomes: tuple[tuple[str, float], ...],
) -> ModelComparisonSummary:
    manifest = ModelComparisonExperimentManifest(
        experiment_id=experiment_id,
        benchmark_id="benchmark",
        benchmark_version="1.0.0",
        judge_model="judge",
        judge_prompt_version="judge-prompt/v1",
        cases=CASES,
        model_targets=(ModelTarget(model_id=model_id, required=True),),
    )
    probe = ModelProbeResult(
        model_id=model_id,
        status="ready",
        observed_model=model_id,
        message="The model passed the capability probe.",
    )
    results = tuple(
        ModelCaseResult(
            model_id=model_id,
            question_id=case.question_id,
            workflow_status="completed",
            judge_status=status,
            score=score,
        )
        for case, (status, score) in zip(CASES, outcomes, strict=True)
    )
    return summarize_model_comparison(ModelComparisonInput(
        manifest=manifest,
        probes=(probe,),
        results=results,
    ))


def _left_summary() -> ModelComparisonSummary:
    return _summary(
        experiment_id="left-experiment",
        model_id="left-model",
        outcomes=(
            ("scored", 100.0),
            ("scored", 100.0),
            ("scored", 60.0),
            ("scored", 80.0),
        ),
    )


def _right_summary() -> ModelComparisonSummary:
    return _summary(
        experiment_id="right-experiment",
        model_id="right-model",
        outcomes=(
            ("scored", 100.0),
            ("no_answer", 0.0),
            ("scored", 100.0),
            ("scored", 70.0),
        ),
    )


def _source(path: str, summary: ModelComparisonSummary) -> CrossExperimentSource:
    return CrossExperimentSource(
        path=path,
        sha256=hashlib.sha256(path.encode("utf-8")).hexdigest(),
        summary=summary,
    )


def _analysis():
    return analyze_cross_experiments((
        _source("left.json", _left_summary()),
        _source("right.json", _right_summary()),
    ))


def _changed_summary(
    summary: ModelComparisonSummary,
    **changes: object,
) -> ModelComparisonSummary:
    payload = summary.model_dump(mode="json")
    payload.update(changes)
    return ModelComparisonSummary.model_validate(payload)


def _write_summary(path: Path, summary: ModelComparisonSummary) -> None:
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def test_analysis_reports_both_full_and_joint_answered_views() -> None:
    pair = _analysis().pairwise_comparisons[0]

    assert pair.left_model_id == "left-model"
    assert pair.right_model_id == "right-model"
    assert pair.overall.mean_delta_right_minus_left == -17.5
    assert (
        pair.overall.right_higher_count,
        pair.overall.equal_count,
        pair.overall.left_higher_count,
    ) == (1, 1, 2)
    assert pair.simple.mean_delta_right_minus_left == -50.0
    assert pair.complex.mean_delta_right_minus_left == 15.0

    joint = pair.answered_only_overall
    assert joint.paired_score_count == 3
    assert joint.expected_case_count == 4
    assert joint.score_coverage == 0.75
    assert joint.mean_delta_right_minus_left == 10.0
    assert (
        joint.right_higher_count,
        joint.equal_count,
        joint.left_higher_count,
    ) == (1, 1, 1)


def test_no_answer_zero_is_scored_but_preserves_its_status() -> None:
    analysis = _analysis()
    question = next(item for item in analysis.cases if item.question_id == "q2")
    right = question.observations[1]
    delta = next(
        item
        for item in analysis.pairwise_comparisons[0].case_deltas
        if item.question_id == "q2"
    )

    assert right.judge_status == "no_answer"
    assert right.score == 0.0
    assert delta.right_judge_status == "no_answer"
    assert delta.delta_right_minus_left == -100.0


@pytest.mark.parametrize(
    ("field_name", "replacement", "error"),
    (
        ("benchmark_id", "different-benchmark", "benchmark_id"),
        ("benchmark_version", "2.0.0", "benchmark_version"),
        ("judge_model", "different-judge", "judge_model"),
        ("judge_prompt_version", "different-prompt/v1", "judge_prompt_version"),
    ),
)
def test_incompatible_experiment_contract_fields_fail_closed(
    field_name: str,
    replacement: str,
    error: str,
) -> None:
    right = _changed_summary(_right_summary(), **{field_name: replacement})

    with pytest.raises(ValueError, match=error):
        analyze_cross_experiments((
            _source("left.json", _left_summary()),
            _source("right.json", right),
        ))


def test_question_order_and_difficulty_must_match() -> None:
    payload = _right_summary().model_dump(mode="json")
    payload["case_summaries"][0], payload["case_summaries"][1] = (
        payload["case_summaries"][1],
        payload["case_summaries"][0],
    )
    payload["common_question_ids"][0], payload["common_question_ids"][1] = (
        payload["common_question_ids"][1],
        payload["common_question_ids"][0],
    )
    reordered = ModelComparisonSummary.model_validate(payload)

    with pytest.raises(ValueError, match="question order and difficulty"):
        analyze_cross_experiments((
            _source("left.json", _left_summary()),
            _source("right.json", reordered),
        ))

    payload = _right_summary().model_dump(mode="json")
    payload["case_summaries"][0]["difficulty"] = "complex"
    changed_difficulty = ModelComparisonSummary.model_validate(payload)
    with pytest.raises(
        ValueError,
        match="question order and difficulty|simple scores disagree",
    ):
        analyze_cross_experiments((
            _source("left.json", _left_summary()),
            _source("right.json", changed_difficulty),
        ))


def test_tampered_derived_slice_and_duplicate_experiment_fail_closed() -> None:
    payload = _right_summary().model_dump(mode="json")
    payload["model_summaries"][0]["simple"]["observed_average_score"] = 99.0
    payload["model_summaries"][0]["simple"]["complete_batch_average_score"] = 99.0
    tampered = ModelComparisonSummary.model_validate(payload)

    with pytest.raises(ValueError, match="simple scores disagree"):
        _source("right.json", tampered)

    duplicate = _changed_summary(
        _right_summary(),
        experiment_id="left-experiment",
    )
    with pytest.raises(ValueError, match="source IDs must be unique"):
        analyze_cross_experiments((
            _source("left.json", _left_summary()),
            _source("right.json", duplicate),
        ))


def test_markdown_is_deterministic_and_states_interpretation_boundaries() -> None:
    analysis = _analysis()

    first = render_cross_experiment_markdown(analysis, top_deltas=2)
    second = render_cross_experiment_markdown(analysis, top_deltas=2)

    assert first == second
    assert "Joint answered cases" in first
    assert "3/4" in first
    assert "`no_answer`" in first
    assert "`completed` / `no_answer`" in first
    assert "one trial per model and question" in first
    assert "does not define Task Success" in first
    assert "statistical significance" in first


def test_analysis_writer_records_hashes_and_freezes_both_outputs(
    tmp_path: Path,
) -> None:
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    _write_summary(left_path, _left_summary())
    _write_summary(right_path, _right_summary())
    output = tmp_path / "report"
    payload = analyze_experiments(
        output,
        (left_path, right_path),
        top_deltas=2,
    )
    generated = json.loads((output / "cross-experiment.json").read_text("utf-8"))

    assert generated == payload
    assert generated["sources"][0]["sha256"] == hashlib.sha256(
        left_path.read_bytes()
    ).hexdigest()
    assert (output / "cross-experiment.md").is_file()

    same = analyze_experiments(
        output,
        (left_path, right_path),
        top_deltas=2,
    )
    assert same == payload

    with pytest.raises(ValueError, match="already exists with different content"):
        analyze_experiments(
            output,
            (left_path, right_path),
            top_deltas=1,
        )


def test_analysis_requires_at_least_two_sources(tmp_path: Path) -> None:
    summary_path = tmp_path / "left.json"
    _write_summary(summary_path, _left_summary())

    with pytest.raises(ValueError, match="at least two"):
        analyze_experiments(tmp_path / "report", (summary_path,))

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from specrag.evaluation import (
    BenchmarkRubric,
    RubricAnnotation,
    RubricItem,
    ScoreBand,
    build_final_answer_judge_input,
    load_end_to_end_benchmark,
    render_benchmark_review,
    validate_benchmark_review,
)


ROOT = Path(__file__).resolve().parents[1]


def _band(ratio: float, description: str = "Reviewed scoring guidance.") -> ScoreBand:
    return ScoreBand(ratio=ratio, description=description)


def _item(identifier: str, weight: int) -> RubricItem:
    return RubricItem(
        rubric_id=identifier,
        criterion="A reviewed semantic requirement.",
        weight=weight,
        full=_band(1.0),
        partial=_band(0.5),
        none=_band(0.0),
    )


def test_loads_complete_frozen_benchmark() -> None:
    benchmark = load_end_to_end_benchmark(ROOT)

    assert benchmark.manifest.benchmark_version == "1.0.0"
    assert benchmark.manifest.trial_count == 1
    assert benchmark.manifest.scoring_scope == "final_answer_only"
    assert len(benchmark.cases) == 50
    assert sum(len(case.rubric.items) for case in benchmark.cases) == 129
    assert sum(len(case.gold_evidence) for case in benchmark.cases) == 72
    assert Counter(case.task_metadata.difficulty for case in benchmark.cases) == {
        "simple": 34,
        "complex": 16,
    }
    assert all(sum(item.weight for item in case.rubric.items) == 100 for case in benchmark.cases)


def test_workflow_projection_excludes_evaluator_only_fields() -> None:
    case = load_end_to_end_benchmark(ROOT).cases[0]

    payload = case.workflow_input().model_dump(mode="json")

    assert set(payload) == {"task_id", "question", "environment"}
    serialized = str(payload)
    assert case.reference_answer not in serialized
    assert case.gold_evidence[0].excerpt not in serialized
    assert case.rubric.items[0].criterion not in serialized


def test_judge_projection_contains_only_the_frozen_four_inputs() -> None:
    case = load_end_to_end_benchmark(ROOT).cases[0]

    payload = build_final_answer_judge_input(
        case,
        candidate_answer="A workflow-generated final answer.",
    ).model_dump(mode="json")

    assert list(payload) == [
        "question",
        "reference_answer",
        "rubric",
        "candidate_answer",
    ]
    serialized = str(payload)
    assert case.gold_evidence[0].excerpt not in serialized
    assert case.environment.environment_id not in serialized


def test_every_gold_document_is_in_exact_environment() -> None:
    benchmark = load_end_to_end_benchmark(ROOT)
    environment = {
        (
            document.source_path,
            document.document_sha256.lower(),
            document.specification,
            document.version,
        )
        for document in benchmark.manifest.environment.documents
    }

    for case in benchmark.cases:
        assert case.environment is benchmark.manifest.environment
        for evidence in case.gold_evidence:
            assert (
                evidence.source_path,
                evidence.document_sha256.lower(),
                evidence.specification,
                evidence.version,
            ) in environment


def test_materialized_review_file_matches_composed_benchmark() -> None:
    benchmark = load_end_to_end_benchmark(ROOT)
    review = validate_benchmark_review(ROOT)

    assert review.case_count == 50
    assert len(review.cases) == 50
    assert review.cases[0].task_metadata.task_id == "r17-qa-001"
    assert review.cases[-1].task_metadata.task_id == "r17-qa-050"
    assert (
        review.cases[0].environment.environment_id
        == review.frozen_execution_environment.environment_id
    )
    assert {
        document.source_path
        for document in review.cases[0].environment.relevant_documents
    } == {evidence.source_path for evidence in review.cases[0].gold_evidence}
    assert (ROOT / "data/QA/end_to_end_benchmark_review.json").read_text(
        encoding="utf-8"
    ) == render_benchmark_review(benchmark)


def test_rubric_rejects_non_contiguous_ids_and_bad_total() -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        BenchmarkRubric(items=(_item("R2", 100),))

    with pytest.raises(ValidationError, match="sum to 100"):
        BenchmarkRubric(items=(_item("R1", 60), _item("R2", 35)))


def test_rubric_rejects_nonstandard_bands() -> None:
    with pytest.raises(ValidationError, match="full=1"):
        RubricItem(
            rubric_id="R1",
            criterion="A reviewed semantic requirement.",
            weight=100,
            full=_band(1.0),
            partial=_band(0.25),
            none=_band(0.0),
        )


def test_source_annotation_rejects_incomplete_weight_budget() -> None:
    with pytest.raises(ValidationError, match="sum to 100"):
        RubricAnnotation.model_validate({
            "question_id": "r17-qa-001",
            "curation_status": "self_reviewed",
            "curated_by": "reviewer",
            "curated_at": "2026-11-02T00:00:00+08:00",
            "items": [
                {"answer_unit_id": "AU1", "weight": 40},
                {"answer_unit_id": "AU2", "weight": 55},
            ],
        })

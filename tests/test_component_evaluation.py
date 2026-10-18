from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from specrag.evaluation.component_metrics import (
    evaluate_planner_artifact,
    evaluate_router_artifact,
    load_component_annotations,
    summarize_planner_evaluations,
    summarize_router_evaluations,
)
from specrag.evaluation.component_models import (
    ComponentQuestionAnnotation,
    ComponentUnitDraft,
    PlannerJudgmentDraft,
    PlannerSemanticJudgment,
)
from specrag.interfaces.baseline_inputs import load_component_route_labels
from scripts.draft_component_evaluation import (
    _annotation_from_draft,
    _validate_unit_draft,
)
from scripts.evaluate_components import _write_json, main as evaluate_components_main


def _annotation(
    question_id: str,
    *,
    route: str = "decompose",
) -> ComponentQuestionAnnotation:
    return ComponentQuestionAnnotation.model_validate({
        "question_id": question_id,
        "review_status": "proposed",
        "expected_route": route,
        "route_rationale": "Reviewed routing rationale.",
        "research_units": [
            {"unit_id": "RU1", "description": "First required objective."},
            {"unit_id": "RU2", "description": "Second required objective."},
        ],
    })


def _write(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_component_reports_are_write_once(tmp_path) -> None:
    path = tmp_path / "report.json"

    _write_json(path, {"score": 1})
    _write_json(path, {"score": 1})

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json(path, {"score": 0})


def test_seed_router_builds_full_annotation_file_and_preserves_base(tmp_path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "\n".join((
            json.dumps({"id": "Q1", "split": "development"}),
            json.dumps({"id": "Q2", "split": "development"}),
        )) + "\n",
        encoding="utf-8",
    )
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(
        "\n".join((
            json.dumps({
                "id": "Q1",
                "proposed_route": "decompose",
                "rationale": "Two objectives.",
                "review_status": "proposed",
            }),
            json.dumps({
                "id": "Q2",
                "proposed_route": "single_task",
                "rationale": "One objective.",
                "review_status": "proposed",
            }),
        )) + "\n",
        encoding="utf-8",
    )
    base = tmp_path / "base.jsonl"
    base.write_text(_annotation("Q1").model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "full.jsonl"

    status = evaluate_components_main([
        "seed-router",
        "--questions",
        str(questions),
        "--route-review",
        str(reviews),
        "--base-annotations",
        str(base),
        "--output",
        str(output),
    ])

    assert status == 0
    annotations = load_component_annotations(output)
    assert tuple(annotations) == ("Q1", "Q2")
    assert annotations["Q1"].research_units
    assert annotations["Q2"].expected_route == "single_task"


def test_component_annotation_rejects_unknown_unit_references() -> None:
    with pytest.raises(ValidationError, match="unknown IDs"):
        ComponentQuestionAnnotation.model_validate({
            "question_id": "Q1",
            "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-10-17T00:00:00+08:00",
            "expected_route": "single_task",
            "route_rationale": "One objective.",
            "research_units": [
                {"unit_id": "RU1", "description": "One objective."}
            ],
            "evidence_units": [{
                "unit_id": "EU1",
                "proposition": "Required proposition.",
                "research_unit_ids": ["RU-missing"],
            }],
        })


def test_reviewed_component_record_requires_auditable_human_review() -> None:
    value = _annotation("Q1").model_dump(mode="json")
    value["review_status"] = "reviewed"

    with pytest.raises(ValidationError, match="reviewed_by and reviewed_at"):
        ComponentQuestionAnnotation.model_validate(value)

    value["reviewed_by"] = "project-owner"
    value["reviewed_at"] = "2026-10-17T09:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        ComponentQuestionAnnotation.model_validate(value)


def test_component_unit_draft_requires_route_shape_and_stable_ids() -> None:
    draft = ComponentUnitDraft.model_validate({
        "research_units": [
            {"unit_id": "RU1", "description": "First objective."},
            {"unit_id": "RU2", "description": "Second objective."},
        ],
        "gold_tasks": [
            {
                "task_id": "T1",
                "question": "First question?",
                "objective": "First objective.",
                "research_unit_ids": ["RU1"],
            },
            {
                "task_id": "T2",
                "question": "Second question?",
                "objective": "Second objective.",
                "research_unit_ids": ["RU2"],
            },
        ],
        "evidence_units": [
            {
                "unit_id": "EU1",
                "proposition": "First proposition.",
                "research_unit_ids": ["RU1"],
            },
            {
                "unit_id": "EU2",
                "proposition": "Second proposition.",
                "research_unit_ids": ["RU2"],
            },
        ],
        "answer_units": [
            {
                "unit_id": "AU1",
                "statement": "First answer unit.",
                "evidence_unit_ids": ["EU1"],
            },
            {
                "unit_id": "AU2",
                "statement": "Second answer unit.",
                "evidence_unit_ids": ["EU2"],
            },
        ],
    })

    _validate_unit_draft(
        draft,
        expected_route="decompose",
        expected_fact_count=2,
    )
    with pytest.raises(ValueError, match="single_task draft"):
        _validate_unit_draft(draft, expected_route="single_task")
    with pytest.raises(ValueError, match="one AnswerUnit per expected fact"):
        _validate_unit_draft(
            draft,
            expected_route="decompose",
            expected_fact_count=1,
        )

    annotation = _annotation_from_draft(_annotation("Q1"), draft)
    assert tuple(item.unit_id for item in annotation.research_units) == (
        "RU1",
        "RU2",
    )


def test_component_unit_draft_rejects_unknown_cross_layer_reference() -> None:
    with pytest.raises(ValidationError, match="unknown IDs"):
        ComponentUnitDraft.model_validate({
            "research_units": [
                {"unit_id": "RU1", "description": "One objective."}
            ],
            "gold_tasks": [{
                "task_id": "T1",
                "question": "One question?",
                "objective": "One objective.",
                "research_unit_ids": ["RU-missing"],
            }],
            "evidence_units": [{
                "unit_id": "EU1",
                "proposition": "One proposition.",
                "research_unit_ids": ["RU1"],
            }],
            "answer_units": [{
                "unit_id": "AU1",
                "statement": "One answer unit.",
                "evidence_unit_ids": ["EU1"],
            }],
        })


def test_planner_judgment_schema_avoids_provider_tuple_prefix_items() -> None:
    schema = PlannerJudgmentDraft.model_json_schema()

    assert "prefixItems" not in json.dumps(schema)
    pair_ids = schema["$defs"]["OverlappingTaskPair"]["properties"]["task_ids"]
    assert pair_ids["minItems"] == 2
    assert pair_ids["maxItems"] == 2


def test_component_route_labels_require_explicit_proposed_opt_in(tmp_path) -> None:
    path = tmp_path / "annotations.jsonl"
    path.write_text(
        _annotation("Q1").model_dump_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require --allow-proposed"):
        load_component_route_labels(path, allow_proposed=False)

    labels = load_component_route_labels(path, allow_proposed=True)
    assert labels["Q1"].difficulty == "complex"
    assert labels["Q1"].route == "decompose"
    assert labels["Q1"].source == "proposed_annotation"
    assert labels["Q1"].reviewed_by is None
    assert labels["Q1"].reviewed_at is None


def test_router_metrics_separate_both_error_directions(tmp_path) -> None:
    cases = (
        ("Q1", "single_task", "single_task"),
        ("Q2", "single_task", "decompose"),
        ("Q3", "decompose", "single_task"),
        ("Q4", "decompose", "decompose"),
    )
    evaluations = []
    for question_id, expected, predicted in cases:
        path = tmp_path / f"{question_id}.json"
        _write(path, {
            "question": {"id": question_id},
            "route": {"route": predicted},
            "receipt": {"prompt_version": "router/test"},
        })
        evaluations.append(
            evaluate_router_artifact(
                path,
                _annotation(question_id, route=expected),
            )
        )

    summary = summarize_router_evaluations(tuple(evaluations))

    assert summary["route_accuracy"] == 0.5
    assert summary["macro_f1"] == 0.5
    assert summary["over_decomposition_count"] == 1
    assert summary["under_decomposition_count"] == 1
    assert summary["calibration_only"] is True


def test_planner_metrics_use_structured_semantic_mapping(tmp_path) -> None:
    artifact_path = tmp_path / "Q1.json"
    _write(artifact_path, {
        "question": {"id": "Q1"},
        "plan": {
            "tasks": [
                {"task_id": "T1", "question": "First"},
                {"task_id": "T2", "question": "Second"},
                {"task_id": "T3", "question": "Extra"},
            ]
        },
    })
    judgment = PlannerSemanticJudgment.model_validate({
        "question_id": "Q1",
        "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-10-17T00:00:00+08:00",
        "unit_mappings": [
            {"unit_id": "RU1", "task_ids": ["T1"], "rationale": "Covered."},
            {"unit_id": "RU2", "task_ids": [], "rationale": "Missing."},
        ],
        "overlapping_task_pairs": [{
            "task_ids": ["T1", "T2"],
            "rationale": "Both seek the same conclusion.",
        }],
        "out_of_scope_task_ids": ["T3"],
    })

    result = evaluate_planner_artifact(
        artifact_path,
        _annotation("Q1"),
        judgment,
    )

    assert result.task_coverage == 0.5
    assert result.full_task_coverage is False
    assert result.task_overlap_rate == pytest.approx(1 / 3)
    assert result.out_of_scope_task_rate == pytest.approx(1 / 3)


def test_planner_rejects_unknown_task_in_judgment(tmp_path) -> None:
    artifact_path = tmp_path / "Q1.json"
    _write(artifact_path, {
        "question": {"id": "Q1"},
        "plan": {"tasks": [{"task_id": "T1"}]},
    })
    judgment = PlannerSemanticJudgment.model_validate({
        "question_id": "Q1",
        "review_status": "proposed",
        "unit_mappings": [
            {"unit_id": "RU1", "task_ids": ["T9"], "rationale": "Bad."},
            {"unit_id": "RU2", "task_ids": [], "rationale": "Missing."},
        ],
    })

    with pytest.raises(ValueError, match="unknown tasks"):
        evaluate_planner_artifact(
            artifact_path,
            _annotation("Q1"),
            judgment,
        )


def test_planner_summary_reports_macro_and_micro_rates(tmp_path) -> None:
    evaluations = []
    for index, task_ids in enumerate((("T1", "T2"), ("T1", "T2", "T3")), 1):
        question_id = f"Q{index}"
        path = tmp_path / f"{question_id}.json"
        _write(path, {
            "question": {"id": question_id},
            "plan": {"tasks": [{"task_id": item} for item in task_ids]},
        })
        judgment = PlannerSemanticJudgment.model_validate({
            "question_id": question_id,
            "review_status": "proposed",
            "unit_mappings": [
                {"unit_id": "RU1", "task_ids": ["T1"], "rationale": "Covered."},
                {"unit_id": "RU2", "task_ids": ["T2"], "rationale": "Covered."},
            ],
            "out_of_scope_task_ids": ["T3"] if index == 2 else [],
        })
        evaluations.append(
            evaluate_planner_artifact(path, _annotation(question_id), judgment)
        )

    summary = summarize_planner_evaluations(tuple(evaluations))

    assert summary["task_coverage"] == 1.0
    assert summary["full_task_coverage_rate"] == 1.0
    assert summary["out_of_scope_task_rate"] == pytest.approx(0.1667)
    assert summary["micro_out_of_scope_task_rate"] == 0.2
    assert summary["calibration_only"] is True

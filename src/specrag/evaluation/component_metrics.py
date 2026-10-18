"""Deterministic Router and Planner component metrics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .component_models import (
    ComponentQuestionAnnotation,
    ExpectedRoute,
    PlannerSemanticJudgment,
    ReviewStatus,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RouterEvaluation(_StrictModel):
    schema_version: Literal["specrag.router-evaluation/v1"] = (
        "specrag.router-evaluation/v1"
    )
    question_id: str
    annotation_schema_version: str
    artifact_schema_version: str | None = None
    annotation_review_status: ReviewStatus
    annotation_reviewed_by: str | None = None
    annotation_reviewed_at: datetime | None = None
    expected_route: ExpectedRoute
    predicted_route: ExpectedRoute
    correct: bool
    error_kind: Literal["over_decomposition", "under_decomposition"] | None
    artifact_path: str
    artifact_sha256: str
    prompt_version: str | None = None
    runtime_model: str | None = None


class PlannerEvaluation(_StrictModel):
    schema_version: Literal["specrag.planner-evaluation/v1"] = (
        "specrag.planner-evaluation/v1"
    )
    question_id: str
    annotation_schema_version: str
    judgment_schema_version: str
    artifact_schema_version: str | None = None
    annotation_review_status: ReviewStatus
    judgment_review_status: ReviewStatus
    annotation_reviewed_by: str | None = None
    annotation_reviewed_at: datetime | None = None
    judgment_reviewed_by: str | None = None
    judgment_reviewed_at: datetime | None = None
    research_unit_count: int = Field(ge=1)
    covered_research_unit_count: int = Field(ge=0)
    task_count: int = Field(ge=1)
    overlapping_task_pair_count: int = Field(ge=0)
    task_pair_count: int = Field(ge=0)
    out_of_scope_task_count: int = Field(ge=0)
    task_coverage: float = Field(ge=0, le=1)
    full_task_coverage: bool
    task_overlap_rate: float = Field(ge=0, le=1)
    out_of_scope_task_rate: float = Field(ge=0, le=1)
    unit_mappings: tuple[dict[str, Any], ...]
    overlapping_task_pairs: tuple[dict[str, Any], ...]
    out_of_scope_task_ids: tuple[str, ...]
    diagnostic_tags: tuple[str, ...]
    artifact_path: str
    artifact_sha256: str
    runtime_prompt_version: str | None = None
    runtime_model: str | None = None
    judge_prompt_version: str | None = None
    judge_model: str | None = None


def load_component_annotations(
    path: Path,
) -> dict[str, ComponentQuestionAnnotation]:
    annotations: dict[str, ComponentQuestionAnnotation] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        annotation = ComponentQuestionAnnotation.model_validate_json(raw_line)
        if annotation.question_id in annotations:
            raise ValueError(
                f"duplicate component annotation for {annotation.question_id} "
                f"at line {line_number}"
            )
        annotations[annotation.question_id] = annotation
    if not annotations:
        raise ValueError("component annotation file is empty")
    return annotations


def load_planner_judgments(path: Path) -> dict[str, PlannerSemanticJudgment]:
    judgments: dict[str, PlannerSemanticJudgment] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        judgment = PlannerSemanticJudgment.model_validate_json(raw_line)
        if judgment.question_id in judgments:
            raise ValueError(
                f"duplicate Planner judgment for {judgment.question_id} "
                f"at line {line_number}"
            )
        judgments[judgment.question_id] = judgment
    if not judgments:
        raise ValueError("Planner judgment file is empty")
    return judgments


def evaluate_router_artifact(
    artifact_path: Path,
    annotation: ComponentQuestionAnnotation,
) -> RouterEvaluation:
    raw_bytes = artifact_path.read_bytes()
    artifact = json.loads(raw_bytes)
    question_id = _question_id(artifact)
    if question_id != annotation.question_id:
        raise ValueError(
            f"Router artifact question {question_id} does not match annotation "
            f"{annotation.question_id}"
        )
    route = artifact.get("route")
    if not isinstance(route, Mapping):
        state = artifact.get("state")
        route = state.get("route_decision") if isinstance(state, Mapping) else None
    if not isinstance(route, Mapping):
        raise ValueError("Router artifact does not contain a route decision")
    predicted = str(route.get("route", ""))
    if predicted not in {"single_task", "decompose"}:
        raise ValueError(f"unsupported Router prediction: {predicted!r}")
    expected = annotation.expected_route
    correct = predicted == expected
    error_kind = None
    if not correct:
        error_kind = (
            "over_decomposition"
            if predicted == "decompose"
            else "under_decomposition"
        )
    receipt = artifact.get("receipt")
    prompt_version = (
        str(receipt.get("prompt_version"))
        if isinstance(receipt, Mapping) and receipt.get("prompt_version")
        else None
    )
    return RouterEvaluation(
        question_id=question_id,
        annotation_schema_version=annotation.schema_version,
        artifact_schema_version=(
            str(artifact["schema_version"]) if artifact.get("schema_version") else None
        ),
        annotation_review_status=annotation.review_status,
        annotation_reviewed_by=annotation.reviewed_by,
        annotation_reviewed_at=annotation.reviewed_at,
        expected_route=expected,
        predicted_route=predicted,  # type: ignore[arg-type]
        correct=correct,
        error_kind=error_kind,
        artifact_path=artifact_path.as_posix(),
        artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        prompt_version=prompt_version,
        runtime_model=(
            str(receipt.get("model"))
            if isinstance(receipt, Mapping) and receipt.get("model")
            else None
        ),
    )


def summarize_router_evaluations(
    evaluations: tuple[RouterEvaluation, ...],
) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("Router summary requires at least one evaluation")
    labels: tuple[ExpectedRoute, ...] = ("single_task", "decompose")
    confusion = {
        expected: {
            predicted: sum(
                item.expected_route == expected and item.predicted_route == predicted
                for item in evaluations
            )
            for predicted in labels
        }
        for expected in labels
    }
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in labels if other != label
        )
        false_negative = sum(
            confusion[label][other] for other in labels if other != label
        )
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[label] = {
            "support": sum(item.expected_route == label for item in evaluations),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return {
        "schema_version": "specrag.router-evaluation-summary/v1",
        "question_count": len(evaluations),
        "reviewed_question_count": sum(
            item.annotation_review_status == "reviewed" for item in evaluations
        ),
        "calibration_only": any(
            item.annotation_review_status != "reviewed" for item in evaluations
        ),
        "route_accuracy": round(
            sum(item.correct for item in evaluations) / len(evaluations), 4
        ),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4),
        "over_decomposition_count": sum(
            item.error_kind == "over_decomposition" for item in evaluations
        ),
        "under_decomposition_count": sum(
            item.error_kind == "under_decomposition" for item in evaluations
        ),
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def evaluate_planner_artifact(
    artifact_path: Path,
    annotation: ComponentQuestionAnnotation,
    judgment: PlannerSemanticJudgment,
) -> PlannerEvaluation:
    if not annotation.research_units:
        raise ValueError(
            f"Planner annotation {annotation.question_id} has no ResearchUnits"
        )
    if annotation.question_id != judgment.question_id:
        raise ValueError("Planner annotation and judgment question IDs differ")
    raw_bytes = artifact_path.read_bytes()
    artifact = json.loads(raw_bytes)
    question_id = _question_id(artifact)
    if question_id != annotation.question_id:
        raise ValueError("Planner artifact and annotation question IDs differ")
    plan = artifact.get("plan")
    if not isinstance(plan, Mapping):
        state = artifact.get("state")
        plan = state.get("research_plan") if isinstance(state, Mapping) else None
    if not isinstance(plan, Mapping):
        raise ValueError("Planner artifact does not contain a plan")
    tasks = tuple(item for item in plan.get("tasks", ()) if isinstance(item, Mapping))
    task_ids = tuple(str(item.get("task_id", "")) for item in tasks)
    if not task_ids or any(not item for item in task_ids):
        raise ValueError("Planner artifact contains no valid task IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Planner artifact task IDs must be unique")
    known_tasks = set(task_ids)
    expected_units = {item.unit_id for item in annotation.research_units}
    mappings = {item.unit_id: item for item in judgment.unit_mappings}
    if set(mappings) != expected_units:
        missing = expected_units - set(mappings)
        extra = set(mappings) - expected_units
        raise ValueError(
            "Planner judgment must map every and only annotated ResearchUnit; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for mapping in judgment.unit_mappings:
        _require_task_ids(mapping.task_ids, known_tasks, f"mapping {mapping.unit_id}")
    normalized_pairs: set[tuple[str, str]] = set()
    for pair in judgment.overlapping_task_pairs:
        _require_task_ids(pair.task_ids, known_tasks, "overlapping pair")
        normalized_pairs.add(tuple(sorted(pair.task_ids)))
    _require_task_ids(
        judgment.out_of_scope_task_ids,
        known_tasks,
        "out-of-scope tasks",
    )

    covered = sum(bool(mapping.task_ids) for mapping in judgment.unit_mappings)
    unit_count = len(annotation.research_units)
    task_count = len(task_ids)
    pair_count = comb(task_count, 2)
    overlap_count = len(normalized_pairs)
    out_of_scope_count = len(judgment.out_of_scope_task_ids)
    return PlannerEvaluation(
        question_id=question_id,
        annotation_schema_version=annotation.schema_version,
        judgment_schema_version=judgment.schema_version,
        artifact_schema_version=(
            str(artifact["schema_version"]) if artifact.get("schema_version") else None
        ),
        annotation_review_status=annotation.review_status,
        judgment_review_status=judgment.review_status,
        annotation_reviewed_by=annotation.reviewed_by,
        annotation_reviewed_at=annotation.reviewed_at,
        judgment_reviewed_by=judgment.reviewed_by,
        judgment_reviewed_at=judgment.reviewed_at,
        research_unit_count=unit_count,
        covered_research_unit_count=covered,
        task_count=task_count,
        overlapping_task_pair_count=overlap_count,
        task_pair_count=pair_count,
        out_of_scope_task_count=out_of_scope_count,
        task_coverage=covered / unit_count,
        full_task_coverage=covered == unit_count,
        task_overlap_rate=overlap_count / pair_count if pair_count else 0.0,
        out_of_scope_task_rate=out_of_scope_count / task_count,
        unit_mappings=tuple(
            item.model_dump(mode="json") for item in judgment.unit_mappings
        ),
        overlapping_task_pairs=tuple(
            item.model_dump(mode="json")
            for item in judgment.overlapping_task_pairs
        ),
        out_of_scope_task_ids=judgment.out_of_scope_task_ids,
        diagnostic_tags=judgment.diagnostic_tags,
        artifact_path=artifact_path.as_posix(),
        artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        runtime_prompt_version=_metadata_value(artifact, "prompt_version"),
        runtime_model=_metadata_value(artifact, "model"),
        judge_prompt_version=judgment.judge_prompt_version,
        judge_model=judgment.judge_model,
    )


def summarize_planner_evaluations(
    evaluations: tuple[PlannerEvaluation, ...],
) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("Planner summary requires at least one evaluation")
    return {
        "schema_version": "specrag.planner-evaluation-summary/v1",
        "question_count": len(evaluations),
        "reviewed_question_count": sum(
            item.annotation_review_status == "reviewed"
            and item.judgment_review_status == "reviewed"
            for item in evaluations
        ),
        "calibration_only": any(
            item.annotation_review_status != "reviewed"
            or item.judgment_review_status != "reviewed"
            for item in evaluations
        ),
        "task_coverage": _mean(evaluations, "task_coverage"),
        "full_task_coverage_rate": round(
            sum(item.full_task_coverage for item in evaluations) / len(evaluations),
            4,
        ),
        "task_overlap_rate": _mean(evaluations, "task_overlap_rate"),
        "out_of_scope_task_rate": _mean(
            evaluations, "out_of_scope_task_rate"
        ),
        "micro_task_overlap_rate": _ratio(
            sum(item.overlapping_task_pair_count for item in evaluations),
            sum(item.task_pair_count for item in evaluations),
        ),
        "micro_out_of_scope_task_rate": _ratio(
            sum(item.out_of_scope_task_count for item in evaluations),
            sum(item.task_count for item in evaluations),
        ),
    }


def _question_id(artifact: Mapping[str, Any]) -> str:
    question = artifact.get("question")
    if isinstance(question, Mapping) and question.get("id"):
        return str(question["id"])
    value = artifact.get("question_id")
    if value:
        return str(value)
    raise ValueError("artifact does not identify its question")


def _require_task_ids(
    values: tuple[str, ...], known_tasks: set[str], label: str
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate task IDs")
    unknown = set(values) - known_tasks
    if unknown:
        raise ValueError(f"{label} references unknown tasks: {sorted(unknown)}")


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _metadata_value(artifact: Mapping[str, Any], key: str) -> str | None:
    value = artifact.get(key)
    if value:
        return str(value)
    receipt = artifact.get("receipt")
    if isinstance(receipt, Mapping) and receipt.get(key):
        return str(receipt[key])
    return None


def _mean(evaluations: tuple[PlannerEvaluation, ...], field: str) -> float:
    return round(
        sum(float(getattr(item, field)) for item in evaluations) / len(evaluations),
        4,
    )


__all__ = [
    "PlannerEvaluation",
    "RouterEvaluation",
    "evaluate_planner_artifact",
    "evaluate_router_artifact",
    "load_component_annotations",
    "load_planner_judgments",
    "summarize_planner_evaluations",
    "summarize_router_evaluations",
]

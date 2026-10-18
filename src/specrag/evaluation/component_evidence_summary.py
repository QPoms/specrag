"""Aggregate evidence-path component metrics without hiding component scores."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def summarize_evaluations(
    evaluations: tuple[BaseModel, ...],
    component: str,
) -> dict[str, Any]:
    if not evaluations:
        raise ValueError(f"{component} summary requires at least one evaluation")
    fields = {
        "retriever": (
            "target_recall_at_5",
            "target_recall_at_12",
            "ndcg_at_5",
            "ndcg_at_12",
        ),
        "researcher": ("evidence_coverage", "evidence_precision"),
        "judge": (
            "final_evidence_coverage",
            "final_evidence_precision",
            "evidence_overlap_rate",
        ),
        "synthesizer": (
            "answer_coverage",
            "answer_groundedness",
            "citation_correctness",
        ),
    }.get(component)
    if fields is None:
        raise ValueError(f"unsupported component summary: {component}")
    result: dict[str, Any] = {
        "schema_version": f"specrag.{component}-evaluation-summary/v1",
        "case_count": len(evaluations),
        "reviewed_case_count": sum(_is_reviewed(item) for item in evaluations),
        "calibration_only": any(not _is_reviewed(item) for item in evaluations),
    }
    for field in fields:
        result[field] = round(
            sum(float(getattr(item, field)) for item in evaluations) / len(evaluations),
            4,
        )
    return result


def _is_reviewed(item: BaseModel) -> bool:
    values = item.model_dump()
    statuses = tuple(
        value for key, value in values.items() if key.endswith("review_status")
    )
    return bool(statuses) and all(value == "reviewed" for value in statuses)


__all__ = ["summarize_evaluations"]

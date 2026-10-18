"""Curated QA inputs shared by diagnostic and baseline commands."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from specrag.config import REPO_ROOT
from specrag.evaluation.component_metrics import load_component_annotations


QUESTION_SET = REPO_ROOT / "data" / "QA" / "questions.jsonl"


@dataclass(frozen=True)
class BaselineQuestion:
    """Only fields permitted to cross the QA-to-runtime boundary."""

    question_id: str
    question: str


@dataclass(frozen=True)
class CuratedRouteLabel:
    difficulty: str
    review_status: str = "reviewed"
    reviewed_by: str | None = None
    reviewed_at: str | None = None

    @property
    def route(self) -> str:
        """Derive the expected Router action from the reviewed label."""

        return "single_task" if self.difficulty == "simple" else "decompose"

    @property
    def source(self) -> str:
        return (
            "reviewed_label"
            if self.review_status == "reviewed"
            else "proposed_annotation"
        )


def _load_questions(
    question_set: Path = QUESTION_SET,
) -> dict[str, BaselineQuestion]:
    selected: dict[str, BaselineQuestion] = {}
    for line_number, raw_line in enumerate(
        question_set.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        question = BaselineQuestion(
            question_id=str(record["id"]),
            question=str(record["question"]),
        )
        if question.question_id in selected:
            raise ValueError(f"duplicate question ID on line {line_number}")
        selected[question.question_id] = question
    return selected


def load_route_labels(
    question_set: Path = QUESTION_SET,
) -> dict[str, CuratedRouteLabel]:
    labels: dict[str, CuratedRouteLabel] = {}
    for raw_line in question_set.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        labels[str(record["id"])] = CuratedRouteLabel(
            difficulty=str(record["difficulty"]),
        )
    return labels


def load_component_route_labels(
    path: Path,
    *,
    allow_proposed: bool,
) -> dict[str, CuratedRouteLabel]:
    """Load versioned component annotations as fixed Planner route inputs."""

    annotations = load_component_annotations(path)
    proposed = tuple(
        sorted(
            item.question_id
            for item in annotations.values()
            if item.review_status != "reviewed"
        )
    )
    if proposed and not allow_proposed:
        raise ValueError(
            "proposed component route annotations require "
            "--allow-proposed-route-annotations: " + ", ".join(proposed)
        )
    return {
        question_id: CuratedRouteLabel(
            difficulty=(
                "simple" if annotation.expected_route == "single_task" else "complex"
            ),
            review_status=annotation.review_status,
            reviewed_by=annotation.reviewed_by,
            reviewed_at=(
                annotation.reviewed_at.isoformat()
                if annotation.reviewed_at is not None
                else None
            ),
        )
        for question_id, annotation in annotations.items()
    }


def select_baseline_questions(
    *,
    identifiers: tuple[str, ...],
    select_all: bool,
    limit: int | None = None,
    question_set: Path = QUESTION_SET,
) -> tuple[BaselineQuestion, ...]:
    available = _load_questions(question_set)
    if select_all:
        questions = tuple(available.values())
    else:
        missing = tuple(
            identifier for identifier in identifiers if identifier not in available
        )
        if missing:
            raise ValueError("unknown question IDs: " + ", ".join(missing))
        questions = tuple(available[identifier] for identifier in identifiers)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        questions = questions[:limit]
    if not questions:
        raise ValueError("no questions were selected")
    return questions


__all__ = [
    "BaselineQuestion",
    "CuratedRouteLabel",
    "QUESTION_SET",
    "load_component_route_labels",
    "load_route_labels",
    "select_baseline_questions",
]

"""Versioned annotation and judgment contracts for component evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ReviewStatus = Literal["proposed", "reviewed"]
ExpectedRoute = Literal["single_task", "decompose"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReviewableModel(_StrictModel):
    """A proposed record or an auditable human-reviewed record."""

    review_status: ReviewStatus
    reviewed_by: str | None = Field(default=None, min_length=1)
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def _auditable_review(self) -> "ReviewableModel":
        if self.review_status == "reviewed":
            if self.reviewed_by is None or self.reviewed_at is None:
                raise ValueError(
                    "reviewed records require reviewed_by and reviewed_at"
                )
            if self.reviewed_at.utcoffset() is None:
                raise ValueError("reviewed_at must include a timezone")
        elif self.reviewed_by is not None or self.reviewed_at is not None:
            raise ValueError("proposed records cannot contain review metadata")
        return self


class ResearchUnit(_StrictModel):
    unit_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class EvidenceUnit(_StrictModel):
    unit_id: str = Field(min_length=1)
    proposition: str = Field(min_length=1)
    research_unit_ids: tuple[str, ...] = Field(min_length=1)


class AnswerUnit(_StrictModel):
    unit_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence_unit_ids: tuple[str, ...] = Field(min_length=1)


class GoldResearchTask(_StrictModel):
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    research_unit_ids: tuple[str, ...] = Field(min_length=1)
    dependencies: tuple[str, ...] = ()


class ComponentUnitDraft(_StrictModel):
    """Provider-neutral semantic-unit proposal before review metadata."""

    research_units: tuple[ResearchUnit, ...] = Field(min_length=1)
    gold_tasks: tuple[GoldResearchTask, ...] = Field(min_length=1)
    evidence_units: tuple[EvidenceUnit, ...] = Field(min_length=1)
    answer_units: tuple[AnswerUnit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_references(self) -> "ComponentUnitDraft":
        ComponentQuestionAnnotation(
            question_id="draft-validation",
            review_status="proposed",
            expected_route="single_task",
            route_rationale="Validate semantic-unit references.",
            research_units=self.research_units,
            gold_tasks=self.gold_tasks,
            evidence_units=self.evidence_units,
            answer_units=self.answer_units,
        )
        return self


class ComponentQuestionAnnotation(ReviewableModel):
    """Reviewed semantic units for one question, populated progressively."""

    schema_version: Literal["specrag.component-annotation/v1"] = (
        "specrag.component-annotation/v1"
    )
    question_id: str = Field(min_length=1)
    expected_route: ExpectedRoute
    route_rationale: str = Field(min_length=1)
    research_units: tuple[ResearchUnit, ...] = ()
    gold_tasks: tuple[GoldResearchTask, ...] = ()
    evidence_units: tuple[EvidenceUnit, ...] = ()
    answer_units: tuple[AnswerUnit, ...] = ()

    @model_validator(mode="after")
    def _valid_references(self) -> "ComponentQuestionAnnotation":
        research_ids = _unique_ids(self.research_units, "research unit")
        evidence_ids = _unique_ids(self.evidence_units, "evidence unit")
        answer_ids = _unique_ids(self.answer_units, "answer unit")
        del answer_ids
        task_ids = _unique_ids(self.gold_tasks, "gold task", field="task_id")

        for task in self.gold_tasks:
            _require_known(
                task.research_unit_ids,
                research_ids,
                f"gold task {task.task_id} research units",
            )
            _require_known(
                task.dependencies,
                task_ids,
                f"gold task {task.task_id} dependencies",
            )
            if task.task_id in task.dependencies:
                raise ValueError(f"gold task {task.task_id} cannot depend on itself")
        for unit in self.evidence_units:
            _require_known(
                unit.research_unit_ids,
                research_ids,
                f"evidence unit {unit.unit_id} research units",
            )
        for unit in self.answer_units:
            _require_known(
                unit.evidence_unit_ids,
                evidence_ids,
                f"answer unit {unit.unit_id} evidence units",
            )
        return self


class ResearchUnitMapping(_StrictModel):
    unit_id: str = Field(min_length=1)
    task_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)


class OverlappingTaskPair(_StrictModel):
    # Use a homogeneous fixed-length collection so provider JSON Schema uses
    # ``items`` instead of unsupported tuple ``prefixItems``.
    task_ids: tuple[str, ...] = Field(min_length=2, max_length=2)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _two_distinct_tasks(self) -> "OverlappingTaskPair":
        if len(self.task_ids) != 2:
            raise ValueError("overlapping task pair requires exactly two task IDs")
        if self.task_ids[0] == self.task_ids[1]:
            raise ValueError("overlapping task pair requires two distinct task IDs")
        return self


class PlannerSemanticJudgment(ReviewableModel):
    """Semantic mappings supplied by a human or independent LLM judge."""

    schema_version: Literal["specrag.planner-judgment/v1"] = (
        "specrag.planner-judgment/v1"
    )
    question_id: str = Field(min_length=1)
    unit_mappings: tuple[ResearchUnitMapping, ...]
    overlapping_task_pairs: tuple[OverlappingTaskPair, ...] = ()
    out_of_scope_task_ids: tuple[str, ...] = ()
    diagnostic_tags: tuple[str, ...] = ()
    judge_prompt_version: str | None = None
    judge_model: str | None = None

    @model_validator(mode="after")
    def _unique_entries(self) -> "PlannerSemanticJudgment":
        _unique_ids(self.unit_mappings, "research-unit mapping", field="unit_id")
        if len(self.out_of_scope_task_ids) != len(set(self.out_of_scope_task_ids)):
            raise ValueError("out-of-scope task IDs must be unique")
        pairs = {
            tuple(sorted(pair.task_ids)) for pair in self.overlapping_task_pairs
        }
        if len(pairs) != len(self.overlapping_task_pairs):
            raise ValueError("overlapping task pairs must be unique")
        if len(self.diagnostic_tags) != len(set(self.diagnostic_tags)):
            raise ValueError("diagnostic tags must be unique")
        return self


class PlannerJudgmentDraft(_StrictModel):
    """Provider-neutral Planner mapping before audit metadata is attached."""

    unit_mappings: tuple[ResearchUnitMapping, ...] = Field(min_length=1)
    overlapping_task_pairs: tuple[OverlappingTaskPair, ...] = ()
    out_of_scope_task_ids: tuple[str, ...] = ()
    diagnostic_tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _unique_entries(self) -> "PlannerJudgmentDraft":
        PlannerSemanticJudgment(
            question_id="draft-validation",
            review_status="proposed",
            unit_mappings=self.unit_mappings,
            overlapping_task_pairs=self.overlapping_task_pairs,
            out_of_scope_task_ids=self.out_of_scope_task_ids,
            diagnostic_tags=self.diagnostic_tags,
        )
        return self


def _unique_ids(
    values: tuple[object, ...],
    label: str,
    *,
    field: str = "unit_id",
) -> set[str]:
    identifiers = [str(getattr(value, field)) for value in values]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{label} IDs must be unique")
    return set(identifiers)


def _require_known(values: tuple[str, ...], known: set[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"{label} reference unknown IDs: {', '.join(sorted(unknown))}")


__all__ = [
    "AnswerUnit",
    "ComponentQuestionAnnotation",
    "ComponentUnitDraft",
    "EvidenceUnit",
    "ExpectedRoute",
    "GoldResearchTask",
    "OverlappingTaskPair",
    "PlannerSemanticJudgment",
    "PlannerJudgmentDraft",
    "ResearchUnit",
    "ResearchUnitMapping",
    "ReviewableModel",
    "ReviewStatus",
]

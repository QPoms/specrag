"""Framework-independent contracts for technical research workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskStatus = Literal["pending", "running", "completed", "blocked", "failed"]
ResearcherStopReason = Literal[
    "agent_finish",
    "round_budget",
    "action_budget",
    "agent_error",
]
StopReason = Literal[
    "completed",
    "iteration_budget",
    "tool_budget",
    "replan_budget",
    "insufficient_evidence",
    "unrecoverable_error",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResearchTask(_StrictModel):
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    dependencies: tuple[str, ...] = ()
    status: TaskStatus = "pending"


class ResearchRouteDecision(_StrictModel):
    route: Literal["single_task", "decompose"]
    complexity: Literal["simple", "complex"] = Field(
        description=(
            "simple for one coherent research task; complex when Planner must "
            "decompose the question before research"
        )
    )
    rationale: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _coherent_route(self) -> "ResearchRouteDecision":
        expected_route = (
            "single_task" if self.complexity == "simple" else "decompose"
        )
        if self.route != expected_route:
            raise ValueError(
                "simple questions require single_task; complex questions require "
                "decompose"
            )
        return self


class ResearchPlan(_StrictModel):
    plan_id: str = Field(min_length=1)
    tasks: tuple[ResearchTask, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_dependencies(self) -> "ResearchPlan":
        identifiers = [task.task_id for task in self.tasks]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("research task IDs must be unique")
        known = set(identifiers)
        for task in self.tasks:
            unknown = set(task.dependencies) - known
            if unknown:
                raise ValueError(
                    f"{task.task_id} has unknown dependencies: "
                    + ", ".join(sorted(unknown))
                )
            if task.task_id in task.dependencies:
                raise ValueError(f"{task.task_id} cannot depend on itself")
        visited: set[str] = set()
        visiting: set[str] = set()
        by_id = {task.task_id: task for task in self.tasks}

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("research task dependencies contain a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in by_id[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in identifiers:
            visit(task_id)
        return self


class CandidateEvidence(_StrictModel):
    evidence_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    release: int = Field(ge=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    source_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")
    heading_path: tuple[str, ...] = ()
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_source(self) -> "CandidateEvidence":
        if self.line_end < self.line_start:
            raise ValueError("evidence line_end precedes line_start")
        if int(self.version.split(".", 1)[0]) != self.release:
            raise ValueError("evidence Release does not match its exact version")
        return self


class ValidatedEvidence(_StrictModel):
    evidence_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    source_valid: bool
    supports_target: bool
    validation_status: Literal["supported", "partial", "rejected", "conflict"]
    rationale: str = Field(min_length=1)


class SourceValidationResult(_StrictModel):
    evidence_id: str = Field(min_length=1)
    valid: bool
    path_allowed: bool
    document_hash_valid: bool
    locator_valid: bool
    excerpt_valid: bool
    errors: tuple[str, ...] = ()


class EvidenceSupportAssessment(_StrictModel):
    evidence_id: str = Field(min_length=1)
    status: Literal["supported", "partial", "rejected", "conflict"]
    rationale: str = Field(min_length=1)


class TaskSufficiencyAssessment(_StrictModel):
    task_id: str = Field(min_length=1)
    status: Literal["sufficient", "partial", "insufficient", "conflict"]
    supporting_evidence_ids: tuple[str, ...] = ()
    missing_relation: str | None = None
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_gap(self) -> "TaskSufficiencyAssessment":
        if self.status == "sufficient" and self.missing_relation is not None:
            raise ValueError("sufficient assessment cannot contain a gap")
        if self.status != "sufficient" and not self.missing_relation:
            raise ValueError("non-sufficient assessment requires missing_relation")
        if len(self.supporting_evidence_ids) != len(
            set(self.supporting_evidence_ids)
        ):
            raise ValueError("supporting evidence IDs must be unique")
        return self


class EvidenceBackedClaim(_StrictModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_evidence(self) -> "EvidenceBackedClaim":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("claim evidence IDs must be unique")
        return self


class EvidenceSynthesis(_StrictModel):
    answer: str = Field(min_length=1)
    claims: tuple[EvidenceBackedClaim, ...] = Field(min_length=1)


class EvidenceConflict(_StrictModel):
    conflict_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    evidence_ids: tuple[str, str]
    kind: Literal["semantic_conflict", "version_conflict"]
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct_evidence(self) -> "EvidenceConflict":
        if self.evidence_ids[0] == self.evidence_ids[1]:
            raise ValueError("conflict requires two different evidence items")
        return self


class ResearchGap(_StrictModel):
    gap_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    kind: Literal[
        "missing_fact",
        "missing_relation",
        "source_conflict",
        "version_conflict",
    ]
    description: str = Field(min_length=1)
    resolved: bool = False


class ToolCallRecord(_StrictModel):
    call_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    status: Literal["completed", "partial", "no_hits", "unavailable", "rejected"]
    evidence_ids: tuple[str, ...] = ()
    error: str | None = None


class ResearchWorkerResult(_StrictModel):
    """Serializable output of one parallel Researcher assignment."""

    task_id: str = Field(min_length=1)
    candidate_evidence: tuple[CandidateEvidence, ...] = ()
    tool_calls: tuple[ToolCallRecord, ...] = ()
    tool_calls_used: int = Field(default=0, ge=0)
    stop_reason: ResearcherStopReason
    errors: tuple[str, ...] = ()


class ResearchBudget(_StrictModel):
    max_iterations: int = Field(default=4, ge=1)
    max_tool_calls: int = Field(default=12, ge=1)
    max_replans: int = Field(default=3, ge=0)
    tool_calls_used: int = Field(default=0, ge=0)
    replans_used: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _within_limits(self) -> "ResearchBudget":
        if self.tool_calls_used > self.max_tool_calls:
            raise ValueError("tool call usage exceeds its budget")
        if self.replans_used > self.max_replans:
            raise ValueError("re-plan usage exceeds its budget")
        return self


__all__ = [
    "CandidateEvidence",
    "EvidenceSupportAssessment",
    "EvidenceBackedClaim",
    "EvidenceConflict",
    "EvidenceSynthesis",
    "ResearchBudget",
    "ResearchGap",
    "ResearchPlan",
    "ResearchRouteDecision",
    "ResearcherStopReason",
    "ResearchTask",
    "SourceValidationResult",
    "StopReason",
    "TaskSufficiencyAssessment",
    "TaskStatus",
    "ToolCallRecord",
    "ResearchWorkerResult",
    "ValidatedEvidence",
]

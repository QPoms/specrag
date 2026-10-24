"""Framework-independent contracts for technical research workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskStatus = Literal["pending", "running", "completed", "blocked", "failed"]
TaskExecutionStatus = Literal["queued", "running", "completed", "failed"]
GapDimension = Literal[
    "actor",
    "action",
    "object",
    "message_or_value",
    "direction",
    "condition",
    "result",
    "state_transition",
    "scope",
]
GapEvidenceNeed = Literal[
    "direct_fact",
    "relationship",
    "procedure",
    "condition_or_exception",
    "cross_reference",
    "conflict",
]
ResearcherStopReason = Literal[
    "submitted_for_validation",
    "round_budget",
    "action_budget",
    "agent_error",
    "cancelled",
    "elapsed_time_budget",
]
StopReason = Literal[
    "completed",
    "iteration_budget",
    "tool_budget",
    "replan_budget",
    "insufficient_evidence",
    "unrecoverable_error",
    "cancelled",
    "elapsed_time_budget",
    "research_time_budget",
]
SynthesisMode = Literal["complete", "partial"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResearchTask(_StrictModel):
    task_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    dependencies: tuple[str, ...] = ()
    status: TaskStatus = "pending"


class ResearchTaskExecution(_StrictModel):
    """Immutable audit record for one initial or corrective task attempt."""

    execution_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    phase: Literal["initial", "corrective"] = "initial"
    status: TaskExecutionStatus = "queued"
    stop_reason: ResearcherStopReason | None = None
    error: str | None = None


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
        expected_route = "single_task" if self.complexity == "simple" else "decompose"
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
        earlier: set[str] = set()
        for task in self.tasks:
            forward = set(task.dependencies) - earlier
            if forward:
                raise ValueError(
                    f"{task.task_id} dependencies must reference earlier tasks: "
                    + ", ".join(sorted(forward))
                )
            earlier.add(task.task_id)
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


class TaskSufficiencyAssessment(_StrictModel):
    task_id: str = Field(min_length=1)
    status: Literal["sufficient", "partial", "insufficient", "conflict"]
    supporting_evidence_ids: tuple[str, ...] = ()
    missing_relation: str | None = None
    required_proposition: str | None = None
    missing_dimensions: tuple[GapDimension, ...] = ()
    evidence_need: GapEvidenceNeed | None = None
    trusted_anchors: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_gap(self) -> "TaskSufficiencyAssessment":
        if self.status == "sufficient" and self.missing_relation is not None:
            raise ValueError("sufficient assessment cannot contain a gap")
        if self.status != "sufficient" and not self.missing_relation:
            raise ValueError("non-sufficient assessment requires missing_relation")
        if self.status == "sufficient" and any(
            (
                self.required_proposition is not None,
                self.missing_dimensions,
                self.evidence_need is not None,
                self.trusted_anchors,
            )
        ):
            raise ValueError("sufficient assessment cannot contain gap details")
        if self.status != "sufficient":
            if self.required_proposition is None:
                object.__setattr__(
                    self,
                    "required_proposition",
                    self.missing_relation,
                )
            if self.evidence_need is None:
                object.__setattr__(
                    self,
                    "evidence_need",
                    ("conflict" if self.status == "conflict" else "relationship"),
                )
        if len(self.supporting_evidence_ids) != len(set(self.supporting_evidence_ids)):
            raise ValueError("supporting evidence IDs must be unique")
        if len(self.missing_dimensions) != len(set(self.missing_dimensions)):
            raise ValueError("missing dimensions must be unique")
        if len(self.trusted_anchors) != len(set(self.trusted_anchors)):
            raise ValueError("trusted anchors must be unique")
        return self


class EvidenceSynthesis(_StrictModel):
    answer: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    mode: SynthesisMode = "complete"
    covered_task_ids: tuple[str, ...] = ()
    unresolved_task_ids: tuple[str, ...] = ()
    conflicting_task_ids: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _read_legacy_claim_artifacts(cls, value: Any) -> Any:
        """Read historical JSON artifacts without retaining claims at runtime."""

        if not isinstance(value, Mapping) or "claims" not in value:
            return value
        claims = value.get("claims") or ()
        evidence_ids = tuple(
            dict.fromkeys(
                str(evidence_id)
                for claim in claims
                if isinstance(claim, Mapping)
                for evidence_id in claim.get("evidence_ids", ())
            )
        )
        answer = str(value.get("answer") or "").strip()
        if not answer:
            answer = "\n\n".join(
                str(claim.get("text", "")).strip()
                for claim in claims
                if isinstance(claim, Mapping) and str(claim.get("text", "")).strip()
            )
        return {"answer": answer, "evidence_ids": evidence_ids}

    @model_validator(mode="after")
    def _unique_evidence(self) -> "EvidenceSynthesis":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("synthesis evidence IDs must be unique")
        for name, values in (
            ("covered task", self.covered_task_ids),
            ("unresolved task", self.unresolved_task_ids),
            ("conflicting task", self.conflicting_task_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"synthesis {name} IDs must be unique")
        if set(self.conflicting_task_ids) - set(self.unresolved_task_ids):
            raise ValueError("conflicting synthesis tasks must also be unresolved")
        if self.mode == "complete" and (
            self.unresolved_task_ids or self.conflicting_task_ids
        ):
            raise ValueError("complete synthesis cannot contain unresolved tasks")
        return self


class EvidenceSynthesisDraft(_StrictModel):
    """Only the evidence-grounded content entrusted to the model."""

    answer: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_evidence(self) -> "EvidenceSynthesisDraft":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("synthesis evidence IDs must be unique")
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
    required_proposition: str | None = None
    missing_dimensions: tuple[GapDimension, ...] = ()
    evidence_need: GapEvidenceNeed = "relationship"
    trusted_anchors: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    resolved: bool = False

    @model_validator(mode="after")
    def _structured_requirement(self) -> "ResearchGap":
        if self.required_proposition is None:
            object.__setattr__(self, "required_proposition", self.description)
        for name, values in (
            ("missing dimensions", self.missing_dimensions),
            ("trusted anchors", self.trusted_anchors),
            ("supporting evidence IDs", self.supporting_evidence_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        return self


class ToolCallRecord(_StrictModel):
    call_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    query_id: str = ""
    lexical_query: str = ""
    semantic_query: str = ""
    hypothetical_document: str = ""
    clause: str = ""
    target_specifications: tuple[str, ...] = ()
    seed_evidence_ids: tuple[str, ...] = ()
    status: Literal["completed", "partial", "no_hits", "unavailable", "rejected"]
    evidence_ids: tuple[str, ...] = ()
    new_evidence_ids: tuple[str, ...] = ()
    observed_evidence: tuple[CandidateEvidence, ...] = ()
    error: str | None = None


class ResearcherEvidenceSelectionRecord(_StrictModel):
    """One audited replacement of a task's bounded active evidence set."""

    selection_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    round_number: int = Field(ge=1)
    phase: Literal["retrieval", "finalization"]
    query_id: str = Field(min_length=1)
    retained_evidence_ids: tuple[str, ...] = ()
    assessment: str = Field(min_length=1)


class ResearchWorkerResult(_StrictModel):
    """Serializable output of one parallel Researcher assignment."""

    task_id: str = Field(min_length=1)
    execution_id: str | None = None
    execution_status: Literal["completed", "failed"] = "completed"
    candidate_evidence: tuple[CandidateEvidence, ...] = ()
    tool_calls: tuple[ToolCallRecord, ...] = ()
    evidence_selections: tuple[ResearcherEvidenceSelectionRecord, ...] = ()
    model_receipts: tuple[dict[str, object], ...] = ()
    latency_ms: float = Field(default=0.0, ge=0)
    tool_calls_used: int = Field(default=0, ge=0)
    stop_reason: ResearcherStopReason
    errors: tuple[str, ...] = ()


class ResearchBudget(_StrictModel):
    max_iterations: int = Field(default=4, ge=1)
    max_tool_calls: int = Field(default=12, ge=1)
    max_gap_tool_calls: int = Field(default=6, ge=0)
    max_replans: int = Field(default=3, ge=0)
    tool_calls_used: int = Field(default=0, ge=0)
    tool_calls_used_by_task: tuple[tuple[str, int], ...] = ()
    gap_tool_calls_used_by_task: tuple[tuple[str, int], ...] = ()
    replans_used: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _within_limits(self) -> "ResearchBudget":
        seen: set[str] = set()
        accounted = 0
        for task_id, used in self.tool_calls_used_by_task:
            if not task_id.strip():
                raise ValueError("task tool-call usage requires a task ID")
            if task_id in seen:
                raise ValueError("task tool-call usage IDs must be unique")
            if used < 0:
                raise ValueError("task tool-call usage cannot be negative")
            if used > self.max_tool_calls:
                raise ValueError("task tool-call usage exceeds its budget")
            seen.add(task_id)
            accounted += used
        gap_seen: set[str] = set()
        for task_id, used in self.gap_tool_calls_used_by_task:
            if not task_id.strip():
                raise ValueError("gap tool-call usage requires a task ID")
            if task_id in gap_seen:
                raise ValueError("gap tool-call usage IDs must be unique")
            if used < 0:
                raise ValueError("gap tool-call usage cannot be negative")
            if used > self.max_gap_tool_calls:
                raise ValueError("gap tool-call usage exceeds its budget")
            gap_seen.add(task_id)
            accounted += used
        if accounted != self.tool_calls_used:
            raise ValueError("aggregate tool-call usage must match task usage")
        if self.replans_used > self.max_replans:
            raise ValueError("re-plan usage exceeds its budget")
        return self

    def initial_tool_calls_used_for(self, task_id: str) -> int:
        """Return initial-phase calls charged to one ResearchTask."""

        return dict(self.tool_calls_used_by_task).get(task_id, 0)

    def gap_tool_calls_used_for(self, task_id: str) -> int:
        """Return Judge-gap corrective calls for one ResearchTask."""

        return dict(self.gap_tool_calls_used_by_task).get(task_id, 0)

    def tool_calls_used_for(self, task_id: str) -> int:
        """Return all retrieval calls charged to one ResearchTask."""

        return self.initial_tool_calls_used_for(task_id) + self.gap_tool_calls_used_for(
            task_id
        )

    def remaining_tool_calls_for(
        self,
        task_id: str,
        *,
        corrective: bool = False,
    ) -> int:
        """Return the phase-specific allowance remaining for one task."""

        if corrective:
            return self.max_gap_tool_calls - self.gap_tool_calls_used_for(task_id)
        return self.max_tool_calls - self.initial_tool_calls_used_for(task_id)

    def record_tool_calls(
        self,
        task_id: str,
        count: int,
        *,
        corrective: bool = False,
    ) -> "ResearchBudget":
        """Charge phase-specific calls while retaining aggregate audit data."""

        if not task_id.strip():
            raise ValueError("task ID must not be blank")
        if count < 0:
            raise ValueError("tool-call increment cannot be negative")
        usage = dict(
            self.gap_tool_calls_used_by_task
            if corrective
            else self.tool_calls_used_by_task
        )
        updated = usage.get(task_id, 0) + count
        allowance = self.max_gap_tool_calls if corrective else self.max_tool_calls
        if updated > allowance:
            phase = "Judge-gap corrective" if corrective else "initial"
            raise ValueError(
                f"ResearchTask {task_id} used {updated} {phase} tool calls "
                f"against its budget {allowance}"
            )
        usage[task_id] = updated
        ordered = tuple(sorted(usage.items()))
        field = (
            "gap_tool_calls_used_by_task" if corrective else "tool_calls_used_by_task"
        )
        total = sum(value for _, value in ordered)
        if corrective:
            total += sum(value for _, value in self.tool_calls_used_by_task)
        else:
            total += sum(value for _, value in self.gap_tool_calls_used_by_task)
        return self.model_copy(
            update={
                "tool_calls_used": total,
                field: ordered,
            }
        )


__all__ = [
    "CandidateEvidence",
    "EvidenceSynthesis",
    "EvidenceSynthesisDraft",
    "ResearchBudget",
    "ResearchGap",
    "ResearchPlan",
    "ResearchRouteDecision",
    "ResearcherStopReason",
    "ResearchTask",
    "ResearchTaskExecution",
    "SourceValidationResult",
    "StopReason",
    "TaskSufficiencyAssessment",
    "TaskStatus",
    "TaskExecutionStatus",
    "ToolCallRecord",
    "ResearchWorkerResult",
    "ValidatedEvidence",
]

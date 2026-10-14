"""Framework-independent contracts for technical research workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskStatus = Literal["pending", "running", "completed", "blocked", "failed"]
ResearcherStopReason = Literal[
    "submitted_for_validation",
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
    candidate_evidence: tuple[CandidateEvidence, ...] = ()
    tool_calls: tuple[ToolCallRecord, ...] = ()
    evidence_selections: tuple[ResearcherEvidenceSelectionRecord, ...] = ()
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

        return (
            self.initial_tool_calls_used_for(task_id)
            + self.gap_tool_calls_used_for(task_id)
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
            "gap_tool_calls_used_by_task"
            if corrective
            else "tool_calls_used_by_task"
        )
        total = sum(value for _, value in ordered)
        if corrective:
            total += sum(value for _, value in self.tool_calls_used_by_task)
        else:
            total += sum(value for _, value in self.gap_tool_calls_used_by_task)
        return self.model_copy(update={
            "tool_calls_used": total,
            field: ordered,
        })


__all__ = [
    "CandidateEvidence",
    "EvidenceBackedClaim",
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

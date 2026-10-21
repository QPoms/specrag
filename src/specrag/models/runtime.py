"""Durable research run and audit event contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .corpus import CorpusVersionRef
from .research import ResearchBudget


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


RunStatus = Literal[
    "queued", "running", "completed", "failed", "cancelled", "timed_out"
]
RunStopReason = Literal[
    "completed",
    "cancelled",
    "elapsed_time_budget",
    "iteration_budget",
    "tool_budget",
    "replan_budget",
    "insufficient_evidence",
    "failed",
]
RunFailureKind = Literal[
    "retryable_transient",
    "rate_limited",
    "timeout",
    "provider_unavailable",
    "authentication_or_configuration",
    "invalid_request",
    "structured_output",
    "model_refusal",
    "context_overflow",
    "internal_error",
    "unknown",
]

# Keep the persisted/API failure vocabulary in one dependency-free contract.
# Adapters and projections import this set instead of maintaining their own
# silently divergent allowlists.
RUN_FAILURE_KINDS: frozenset[str] = frozenset({
    "retryable_transient",
    "rate_limited",
    "timeout",
    "provider_unavailable",
    "authentication_or_configuration",
    "invalid_request",
    "structured_output",
    "model_refusal",
    "context_overflow",
    "internal_error",
    "unknown",
})


class ResearchRun(_StrictModel):
    schema_version: Literal[
        "research-run/v1", "research-run/v1.1", "research-run/v1.2",
        "research-run/v1.3", "research-run/v1.4", "research-run/v1.5",
    ] = (
        "research-run/v1.5"
    )
    run_id: str = Field(pattern=r"^RUN-[0-9A-F]{32}$")
    thread_id: str = Field(pattern=r"^THREAD-[0-9A-F]{32}$")
    workspace_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    release: int = Field(ge=1)
    answer_language: str = Field(default="English", min_length=1)
    detail_level: str = Field(default="standard", min_length=1)
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    status: RunStatus = "queued"
    stop_reason: RunStopReason | None = None
    failure_kind: RunFailureKind | None = None
    failure_retryable: bool | None = None
    failure_attempt_count: int | None = Field(default=None, ge=1)
    failure_status_code: int | None = Field(default=None, ge=100, le=599)
    development_question_id: str | None = None
    specfact_ids: tuple[str, ...] = ()
    specfact_proposal_ids: tuple[str, ...] = ()
    memory_ids: tuple[str, ...] = ()
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    deadline_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    retry_of_run_id: str | None = Field(
        default=None, pattern=r"^RUN-[0-9A-F]{32}$"
    )
    # A Run may be created by an older persisted schema without a corpus
    # binding. New Runs receive an exact binding before execution starts.
    corpus_version_policy: Literal[
        "latest_per_spec",
        "highest_version_within_release",
        "exact_version",
        "mixed_versions",
    ] | None = None
    corpus_release: int | None = Field(default=None, ge=1)
    corpus_versions: tuple[CorpusVersionRef, ...] = ()

    @model_validator(mode="after")
    def _coherent_lifecycle(self) -> "ResearchRun":
        terminal = self.status in {"completed", "failed", "cancelled", "timed_out"}
        if terminal and self.finished_at is None:
            raise ValueError("terminal run requires finished_at")
        if not terminal and self.finished_at is not None:
            raise ValueError("non-terminal run cannot have finished_at")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("run cannot start before creation")
        if self.finished_at is not None and self.started_at is not None:
            if self.finished_at < self.started_at:
                raise ValueError("run cannot finish before start")
        if self.retry_of_run_id == self.run_id:
            raise ValueError("Run cannot retry itself")
        if self.corpus_version_policy is None and (
            self.corpus_release is not None or self.corpus_versions
        ):
            raise ValueError("corpus binding policy is required with corpus data")
        if self.corpus_version_policy in {"latest_per_spec", "highest_version_within_release"}:
            if self.corpus_versions:
                raise ValueError("release corpus binding cannot list explicit versions")
            if self.corpus_version_policy == "highest_version_within_release" and self.corpus_release is None:
                raise ValueError("Release is required for a Release corpus binding")
            if self.corpus_version_policy == "latest_per_spec" and self.corpus_release is not None:
                raise ValueError("latest corpus binding cannot select a Release")
        elif self.corpus_version_policy == "exact_version":
            if len(self.corpus_versions) != 1 or self.corpus_release is not None:
                raise ValueError("exact corpus binding requires one version")
        elif self.corpus_version_policy == "mixed_versions":
            if len(self.corpus_versions) < 2 or self.corpus_release is not None:
                raise ValueError("mixed corpus binding requires two versions")
        if self.corpus_versions:
            ordered = tuple(sorted(
                self.corpus_versions,
                key=lambda item: (item.specification, item.version),
            ))
            if len({item.specification for item in ordered}) != len(ordered):
                raise ValueError("Run corpus binding selects a specification twice")
            object.__setattr__(self, "corpus_versions", ordered)
        return self


class ResearchRequest(_StrictModel):
    question: str = Field(min_length=1)
    release: int = Field(default=17, ge=1)
    answer_language: str = Field(default="English", min_length=1)
    detail_level: str = Field(default="standard", min_length=1)
    configuration_memory_ids: tuple[str, ...] = ()
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    elapsed_time_seconds: int = Field(default=900, ge=1, le=86400)
    development_question_id: str | None = None


class RunEvent(_StrictModel):
    schema_version: Literal["run-event/v1"] = "run-event/v1"
    event_id: str = Field(pattern=r"^EVT-[0-9A-F]{32}$")
    run_id: str = Field(pattern=r"^RUN-[0-9A-F]{32}$")
    workspace_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    event_type: Literal[
        "run_started",
        "node_transition",
        "task_dispatched",
        "tool_call",
        "evidence_selected",
        "source_validated",
        "judge_completed",
        "gap_created",
        "replanned",
        "synthesized",
        "resumed",
        "cancel_requested",
        "run_stopped",
        "model_receipt",
    ]
    occurred_at: datetime
    stage: str | None = None
    task_id: str | None = None
    operation_key: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ResearchRequest", "ResearchRun", "RunEvent", "RunStatus", "RunStopReason",
    "RunFailureKind", "RUN_FAILURE_KINDS",
]

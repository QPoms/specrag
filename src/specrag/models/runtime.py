"""Durable research run and audit event contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ResearchRun(_StrictModel):
    schema_version: Literal["research-run/v1"] = "research-run/v1"
    run_id: str = Field(pattern=r"^RUN-[0-9A-F]{32}$")
    thread_id: str = Field(pattern=r"^THREAD-[0-9A-F]{32}$")
    workspace_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    release: int = Field(ge=1)
    answer_language: str = Field(default="English", min_length=1)
    detail_level: str = Field(default="standard", min_length=1)
    status: RunStatus = "queued"
    stop_reason: RunStopReason | None = None
    development_question_id: str | None = None
    specfact_ids: tuple[str, ...] = ()
    specfact_proposal_ids: tuple[str, ...] = ()
    memory_ids: tuple[str, ...] = ()
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    deadline_at: datetime | None = None
    cancel_requested_at: datetime | None = None

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
        return self


class ResearchRequest(_StrictModel):
    question: str = Field(min_length=1)
    release: int = Field(default=17, ge=1)
    answer_language: str = Field(default="English", min_length=1)
    detail_level: str = Field(default="standard", min_length=1)
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
]

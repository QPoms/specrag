"""Dependency-free records for request budgeting and context compaction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelContextProfile(_StrictModel):
    model_id: str = Field(min_length=1)
    context_window_tokens: int = Field(ge=1)
    max_input_tokens: int = Field(ge=1)
    expected_output_tokens: int = Field(ge=1)


class ContextBudget(_StrictModel):
    model_id: str = Field(min_length=1)
    estimated_input_tokens: int = Field(ge=0)
    reserve_tokens: int = Field(ge=1)
    input_ceiling_tokens: int = Field(ge=0)
    fits: bool
    component_tokens: dict[str, int] = Field(default_factory=dict)


class ContextRecord(_StrictModel):
    context_id: str = Field(min_length=1)
    workspace_id: str | None = None
    owner_user_id: str | None = None
    run_id: str | None = None
    active_checkpoint_id: str | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def _scope_is_coherent(self) -> "ContextRecord":
        if (self.workspace_id is None) != (self.owner_user_id is None):
            raise ValueError("workspace and owner must be supplied together")
        return self


ContextMessageRole = Literal["system", "user", "assistant", "tool"]


class ContextMessage(_StrictModel):
    message_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    role: ContextMessageRole
    content: Any
    token_estimate: int = Field(ge=0)
    created_at: datetime
    turn_id: str | None = None


class StoredToolResult(_StrictModel):
    tool_result_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: Any
    token_estimate: int = Field(ge=0)
    created_at: datetime


class ToolResultPage(_StrictModel):
    tool_result_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: Any
    truncated: bool
    next_cursor: str | None = None
    token_estimate: int = Field(ge=0)


class CompactionSummary(_StrictModel):
    objective: str = Field(min_length=1)
    requirements: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    completed: tuple[str, ...] = ()
    progress: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    next_moves: tuple[str, ...] = ()
    references: tuple[str, ...] = ()


CompactionTrigger = Literal["auto", "manual", "overflow"]


class CompactionCheckpoint(_StrictModel):
    checkpoint_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    summary: CompactionSummary
    covers_through_message_id: str = Field(min_length=1)
    recent_start_message_id: str = Field(min_length=1)
    trigger: CompactionTrigger
    model_id: str | None = None
    before_tokens: int = Field(ge=0)
    after_tokens: int = Field(ge=0)
    created_at: datetime


class ReconstructedContext(_StrictModel):
    context_id: str
    summary: CompactionSummary | None = None
    messages: tuple[ContextMessage, ...] = ()


__all__ = [
    "CompactionCheckpoint", "CompactionSummary", "CompactionTrigger",
    "ContextBudget", "ContextMessage", "ContextMessageRole", "ContextRecord",
    "ModelContextProfile", "ReconstructedContext", "StoredToolResult",
    "ToolResultPage",
]

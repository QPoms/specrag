"""Provider-neutral contracts for the internal specification research tools."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .results import PriorResultRef, ResearchResult, SpecificationScope
from .runtime import RunFailureKind, RunStatus


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServiceDelegation(_StrictModel):
    schema_version: Literal["mcp-delegation/v1"] = "mcp-delegation/v1"
    audience: Literal["specrag-mcp"] = "specrag-mcp"
    user_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    conversation_id: str = Field(pattern=r"^CONV-[0-9A-F]{32}$")
    origin_message_id: str = Field(min_length=1)
    origin_turn_id: str = Field(pattern=r"^TURN-[0-9A-F]{32}$")
    submitted_user_sequence: int = Field(ge=1)
    tool_call_id: str = Field(pattern=r"^CALL-[0-9A-F]{32}$")
    operation_key: str = Field(min_length=1, max_length=300)
    issued_at: datetime
    expires_at: datetime
    nonce: str = Field(min_length=16, max_length=200)

    @model_validator(mode="after")
    def _valid_window(self) -> "ServiceDelegation":
        if self.expires_at <= self.issued_at:
            raise ValueError("delegation must expire after issuance")
        return self


class ResearchToolRequest(_StrictModel):
    question: str = Field(min_length=1, max_length=20000)
    objective: str | None = Field(default=None, min_length=1, max_length=20000)
    conditions: tuple[str, ...] = Field(default=(), max_length=50)
    requested_scope: SpecificationScope | None = None
    prior_results: tuple[PriorResultRef, ...] = Field(default=(), max_length=50)
    answer_language: str = Field(default="English", min_length=1, max_length=64)
    detail_level: str = Field(default="standard", min_length=1, max_length=64)

    @model_validator(mode="after")
    def _unique_context(self) -> "ResearchToolRequest":
        if any(not item.strip() or len(item) > 4000 for item in self.conditions):
            raise ValueError("research conditions must be 1 to 4000 characters")
        if len(self.conditions) != len(set(self.conditions)):
            raise ValueError("research conditions must be unique")
        if len(self.prior_results) != len(
            {item.result_id for item in self.prior_results}
        ):
            raise ValueError("prior Result IDs must be unique")
        return self


MCPResearchDisposition = Literal["completed", "accepted", "terminal"]
MCPRuntimeStatus = Literal["starting", "warming", "ready", "failed"]


class MCPWarmupStage(_StrictModel):
    name: Literal[
        "corpus_snapshot",
        "local_snapshot_discovery",
        "snapshot_consistency",
        "retriever_construction",
    ]
    duration_ms: float = Field(ge=0)


class MCPRuntimeReadiness(_StrictModel):
    status: MCPRuntimeStatus
    release: int = Field(default=17, ge=1)
    message: str = Field(min_length=1, max_length=300)
    stages: tuple[MCPWarmupStage, ...] = ()
    total_ms: float = Field(default=0, ge=0)


class MCPResearchStartResponse(_StrictModel):
    disposition: MCPResearchDisposition
    run_id: str = Field(pattern=r"^RUN-[0-9A-F]{32}$")
    status: RunStatus
    result_id: str | None = Field(
        default=None, pattern=r"^RESULT-[0-9A-F]{32}$"
    )
    result: ResearchResult | None = None

    @model_validator(mode="after")
    def _coherent_disposition(self) -> "MCPResearchStartResponse":
        if self.disposition == "completed" and (
            self.status != "completed"
            or self.result is None
            or self.result_id != self.result.result_id
        ):
            raise ValueError("completed research response requires its Result")
        if self.disposition == "terminal" and self.status not in {
            "failed", "cancelled", "timed_out"
        }:
            raise ValueError("terminal response requires abnormal terminal status")
        return self


class MCPResearchStatusResponse(_StrictModel):
    run_id: str = Field(pattern=r"^RUN-[0-9A-F]{32}$")
    status: RunStatus
    stop_reason: str | None = None
    result_id: str | None = Field(
        default=None, pattern=r"^RESULT-[0-9A-F]{32}$"
    )
    failure_kind: RunFailureKind | None = None
    failure_retryable: bool | None = None
    last_event_sequence: int = Field(default=0, ge=0)
    last_event_type: str | None = None


class MCPResearchResultResponse(_StrictModel):
    result: ResearchResult


__all__ = [
    "MCPResearchDisposition",
    "MCPResearchResultResponse",
    "MCPResearchStartResponse",
    "MCPResearchStatusResponse",
    "MCPRuntimeReadiness",
    "MCPRuntimeStatus",
    "MCPWarmupStage",
    "ResearchToolRequest",
    "ServiceDelegation",
]

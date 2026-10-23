"""Persistent Conversation aggregate and lifecycle contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .results import SpecificationScope
from .runtime import RunStatus


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


ConversationStatus = Literal["active", "archived"]
ConversationTurnStatus = Literal[
    "queued", "running", "completed", "failed", "cancelled"
]
ConversationTurnTrigger = Literal["user_message", "research_result_ready"]
ConversationToolCallStatus = Literal["proposed", "running", "completed", "failed"]
ResearchPresentationStatus = Literal["pending", "queued", "presented", "skipped"]
AssistantSourceKind = Literal[
    "direct_conversation", "research_result", "historical_result_readback"
]


class Conversation(_StrictModel):
    schema_version: Literal["conversation/v1"] = "conversation/v1"
    conversation_id: str = Field(pattern=r"^CONV-[0-9A-F]{32}$")
    workspace_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    status: ConversationStatus = "active"
    title: str | None = Field(default=None, min_length=1, max_length=200)
    revision: int = Field(default=0, ge=0)
    last_message_sequence: int = Field(default=0, ge=0)
    last_user_message_sequence: int = Field(default=0, ge=0)
    active_turn_id: str | None = Field(
        default=None, pattern=r"^TURN-[0-9A-F]{32}$"
    )
    active_scope_revision_id: str | None = Field(
        default=None, pattern=r"^SCOPE-[0-9A-F]{32}$"
    )
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def _coherent_state(self) -> "Conversation":
        if self.updated_at < self.created_at:
            raise ValueError("Conversation cannot update before creation")
        if self.last_user_message_sequence > self.last_message_sequence:
            raise ValueError("user message sequence exceeds message head")
        if self.status == "archived" and self.active_turn_id is not None:
            raise ValueError("archived Conversation cannot have an active Turn")
        return self


class ConversationMessageContent(_StrictModel):
    schema_version: Literal["conversation-message/v1"] = "conversation-message/v1"
    text: str = Field(min_length=1)
    source_kind: AssistantSourceKind | None = None
    research_result_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    status_kind: Literal[
        "queued", "running", "completed", "failed", "cancelled", "timed_out"
    ] | None = None

    @model_validator(mode="after")
    def _coherent_sources(self) -> "ConversationMessageContent":
        for name, values in (
            ("Research Result IDs", self.research_result_ids),
            ("Run IDs", self.run_ids),
            ("citations", self.citations),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"message {name} must be unique")
        if self.source_kind in {"research_result", "historical_result_readback"}:
            if not self.research_result_ids:
                raise ValueError("research-backed message requires a Result")
        if self.source_kind == "direct_conversation" and (
            self.research_result_ids or self.citations
        ):
            raise ValueError("direct conversation cannot claim research evidence")
        return self


class ConversationTurnBudget(_StrictModel):
    max_model_steps: int = Field(default=4, ge=1, le=20)
    max_tool_calls: int = Field(default=6, ge=0, le=50)
    max_research_submissions: int = Field(default=1, ge=0, le=10)
    elapsed_time_seconds: int = Field(default=60, ge=1, le=900)


class ConversationTurn(_StrictModel):
    schema_version: Literal["conversation-turn/v1"] = "conversation-turn/v1"
    turn_id: str = Field(pattern=r"^TURN-[0-9A-F]{32}$")
    conversation_id: str = Field(pattern=r"^CONV-[0-9A-F]{32}$")
    trigger: ConversationTurnTrigger
    trigger_message_id: str | None = None
    trigger_result_id: str | None = Field(
        default=None, pattern=r"^RESULT-[0-9A-F]{32}$"
    )
    trigger_run_id: str | None = Field(
        default=None, pattern=r"^RUN-[0-9A-F]{32}$"
    )
    submitted_user_sequence: int = Field(ge=0)
    status: ConversationTurnStatus = "queued"
    budget: ConversationTurnBudget = Field(default_factory=ConversationTurnBudget)
    operation_key: str = Field(min_length=1, max_length=200)
    reply_message_id: str | None = None
    stop_reason: str | None = None
    failure_kind: str | None = None
    failure_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def _coherent_lifecycle(self) -> "ConversationTurn":
        if self.trigger == "user_message":
            if self.trigger_message_id is None:
                raise ValueError("user Turn requires a trigger message")
            if self.trigger_result_id is not None or self.trigger_run_id is not None:
                raise ValueError("user Turn cannot have a Result trigger")
        else:
            if self.trigger_message_id is not None:
                raise ValueError("Result Turn cannot have a user trigger message")
            if self.trigger_result_id is None or self.trigger_run_id is None:
                raise ValueError("Result Turn requires Result and Run triggers")
            if self.budget.max_research_submissions != 0:
                raise ValueError("automatic Result Turn cannot submit research")
        terminal = self.status in {"completed", "failed", "cancelled"}
        if terminal != (self.finished_at is not None):
            raise ValueError("Turn terminal state and finished_at must agree")
        if self.status in {"running", "completed", "failed"}:
            if self.started_at is None:
                raise ValueError("started or terminal Turn requires started_at")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("Turn cannot start before creation")
        if (
            self.finished_at is not None
            and self.started_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("Turn cannot finish before start")
        if self.status == "completed" and self.reply_message_id is None:
            raise ValueError("completed Turn requires a reply message")
        if self.status == "failed" and self.failure_kind is None:
            raise ValueError("failed Turn requires a failure kind")
        return self


class ConversationToolCall(_StrictModel):
    schema_version: Literal["conversation-tool-call/v1"] = (
        "conversation-tool-call/v1"
    )
    call_id: str = Field(pattern=r"^CALL-[0-9A-F]{32}$")
    conversation_id: str = Field(pattern=r"^CONV-[0-9A-F]{32}$")
    turn_id: str = Field(pattern=r"^TURN-[0-9A-F]{32}$")
    tool_name: str = Field(min_length=1)
    arguments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arguments: dict[str, Any]
    status: ConversationToolCallStatus = "proposed"
    operation_key: str = Field(min_length=1, max_length=300)
    tool_result_id: str | None = None
    run_id: str | None = Field(default=None, pattern=r"^RUN-[0-9A-F]{32}$")
    result_id: str | None = Field(
        default=None, pattern=r"^RESULT-[0-9A-F]{32}$"
    )
    safe_error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def _coherent_lifecycle(self) -> "ConversationToolCall":
        terminal = self.status in {"completed", "failed"}
        if terminal != (self.finished_at is not None):
            raise ValueError("tool-call terminal state and finished_at must agree")
        if self.status in {"running", "completed", "failed"}:
            if self.started_at is None:
                raise ValueError("started tool call requires started_at")
        if self.status == "failed" and not self.safe_error:
            raise ValueError("failed tool call requires a safe error")
        return self


class ConversationScopeRevision(_StrictModel):
    schema_version: Literal["conversation-scope/v1"] = "conversation-scope/v1"
    scope_revision_id: str = Field(pattern=r"^SCOPE-[0-9A-F]{32}$")
    conversation_id: str = Field(pattern=r"^CONV-[0-9A-F]{32}$")
    revision: int = Field(ge=1)
    objective: str = Field(min_length=1)
    conditions: tuple[str, ...] = ()
    scope: SpecificationScope
    source_message_ids: tuple[str, ...] = Field(min_length=1)
    predecessor_id: str | None = Field(
        default=None, pattern=r"^SCOPE-[0-9A-F]{32}$"
    )
    created_at: datetime

    @model_validator(mode="after")
    def _unique_inputs(self) -> "ConversationScopeRevision":
        if len(self.conditions) != len(set(self.conditions)):
            raise ValueError("scope conditions must be unique")
        if len(self.source_message_ids) != len(set(self.source_message_ids)):
            raise ValueError("scope source messages must be unique")
        if self.predecessor_id == self.scope_revision_id:
            raise ValueError("scope revision cannot precede itself")
        return self


class ConversationResearchLink(_StrictModel):
    schema_version: Literal["conversation-research-link/v1"] = (
        "conversation-research-link/v1"
    )
    link_id: str = Field(pattern=r"^LINK-[0-9A-F]{32}$")
    conversation_id: str = Field(pattern=r"^CONV-[0-9A-F]{32}$")
    origin_message_id: str = Field(min_length=1)
    origin_turn_id: str = Field(pattern=r"^TURN-[0-9A-F]{32}$")
    tool_call_id: str = Field(pattern=r"^CALL-[0-9A-F]{32}$")
    run_id: str = Field(pattern=r"^RUN-[0-9A-F]{32}$")
    submitted_user_sequence: int = Field(ge=1)
    task_status: RunStatus
    latest_result_id: str | None = Field(
        default=None, pattern=r"^RESULT-[0-9A-F]{32}$"
    )
    presentation_status: ResearchPresentationStatus | None = None
    status_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    presented_at: datetime | None = None

    @model_validator(mode="after")
    def _coherent_projection(self) -> "ConversationResearchLink":
        if self.updated_at < self.created_at:
            raise ValueError("research link cannot update before creation")
        if self.latest_result_id is None and self.presentation_status is not None:
            raise ValueError("presentation state requires a Result")
        if self.latest_result_id is not None and self.presentation_status is None:
            raise ValueError("Result link requires presentation state")
        if self.presentation_status == "presented" and self.presented_at is None:
            raise ValueError("presented link requires presented_at")
        if self.presentation_status != "presented" and self.presented_at is not None:
            raise ValueError("only presented link may have presented_at")
        return self


ConversationEventType = Literal[
    "conversation_created",
    "message_appended",
    "turn_queued",
    "turn_started",
    "turn_stopped",
    "tool_call_started",
    "tool_call_completed",
    "tool_call_failed",
    "run_linked",
    "run_status_changed",
    "result_available",
    "auto_reply_deferred",
    "auto_reply_queued",
    "auto_reply_presented",
    "auto_reply_skipped",
    "conversation_archived",
    "conversation_unarchived",
]


class ConversationEvent(_StrictModel):
    schema_version: Literal["conversation-event/v1"] = "conversation-event/v1"
    event_id: str = Field(pattern=r"^CEVT-[0-9A-F]{32}$")
    conversation_id: str = Field(pattern=r"^CONV-[0-9A-F]{32}$")
    workspace_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    event_type: ConversationEventType
    operation_key: str = Field(min_length=1, max_length=300)
    occurred_at: datetime
    message_id: str | None = None
    turn_id: str | None = Field(default=None, pattern=r"^TURN-[0-9A-F]{32}$")
    call_id: str | None = Field(default=None, pattern=r"^CALL-[0-9A-F]{32}$")
    run_id: str | None = Field(default=None, pattern=r"^RUN-[0-9A-F]{32}$")
    result_id: str | None = Field(
        default=None, pattern=r"^RESULT-[0-9A-F]{32}$"
    )
    payload: dict[str, Any] = Field(default_factory=dict)


_Status = TypeVar("_Status", bound=str)


def _require_transition(
    current: _Status,
    target: _Status,
    allowed: dict[str, frozenset[str]],
    lifecycle: str,
) -> _Status:
    if target not in allowed.get(current, frozenset()):
        raise ValueError(f"invalid {lifecycle} transition: {current} -> {target}")
    return target


def require_conversation_status_transition(
    current: ConversationStatus, target: ConversationStatus
) -> ConversationStatus:
    return _require_transition(
        current,
        target,
        {"active": frozenset({"archived"}), "archived": frozenset({"active"})},
        "Conversation",
    )


def require_turn_status_transition(
    current: ConversationTurnStatus, target: ConversationTurnStatus
) -> ConversationTurnStatus:
    return _require_transition(
        current,
        target,
        {
            "queued": frozenset({"running", "cancelled"}),
            "running": frozenset({"completed", "failed", "cancelled", "queued"}),
        },
        "Conversation Turn",
    )


def require_tool_call_status_transition(
    current: ConversationToolCallStatus, target: ConversationToolCallStatus
) -> ConversationToolCallStatus:
    return _require_transition(
        current,
        target,
        {
            "proposed": frozenset({"running", "failed"}),
            "running": frozenset({"completed", "failed"}),
        },
        "Conversation tool call",
    )


def require_presentation_status_transition(
    current: ResearchPresentationStatus, target: ResearchPresentationStatus
) -> ResearchPresentationStatus:
    return _require_transition(
        current,
        target,
        {
            "pending": frozenset({"queued", "skipped"}),
            "queued": frozenset({"pending", "presented", "skipped"}),
        },
        "research presentation",
    )


__all__ = [
    "AssistantSourceKind",
    "Conversation",
    "ConversationEvent",
    "ConversationEventType",
    "ConversationMessageContent",
    "ConversationResearchLink",
    "ConversationScopeRevision",
    "ConversationStatus",
    "ConversationToolCall",
    "ConversationToolCallStatus",
    "ConversationTurn",
    "ConversationTurnBudget",
    "ConversationTurnStatus",
    "ConversationTurnTrigger",
    "ResearchPresentationStatus",
    "require_conversation_status_transition",
    "require_presentation_status_transition",
    "require_tool_call_status_transition",
    "require_turn_status_transition",
]

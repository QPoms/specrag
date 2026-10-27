"""Framework-independent HTTP transport contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .access import Membership, User, Workspace
from .context import ContextMessage
from .conversation import (
    Conversation,
    ConversationResearchLink,
    ConversationToolCall,
    ConversationTurn,
)
from .memory import MemoryEntry, MemoryProposal, MemoryTopic
from .research import (
    CandidateEvidence,
    ResearchGap,
    ResearchTask,
    TaskSufficiencyAssessment,
)
from .results import EvidenceStatus
from .runtime import (
    ResearchRun,
    RunEvent,
    RunFailureKind,
    RunStatus,
    RunStopReason,
)
from .specfacts import SpecFact, SpecFactProposal


class _StrictTransport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiErrorResponse(_StrictTransport):
    """Stable error envelope shared by HTTP clients."""

    code: str = Field(min_length=1, max_length=100)
    detail: str | tuple[dict[str, object], ...]


class LoginRequest(_StrictTransport):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class SessionResponse(_StrictTransport):
    access_token: str = Field(min_length=1)
    token_type: Literal["bearer"] = "bearer"
    expires_at: datetime


class WorkspaceCreateRequest(_StrictTransport):
    name: str = Field(min_length=1, max_length=200)


class ConversationCreateRequest(_StrictTransport):
    title: str | None = Field(default=None, min_length=1, max_length=200)


class ConversationMessageCreateRequest(_StrictTransport):
    text: str = Field(min_length=1, max_length=20000)
    operation_key: str = Field(min_length=1, max_length=200)
    expected_revision: int | None = Field(default=None, ge=0)


class ConversationStatusRequest(_StrictTransport):
    expected_revision: int | None = Field(default=None, ge=0)


class ConversationDetailResponse(_StrictTransport):
    conversation: Conversation
    messages: tuple[ContextMessage, ...] = ()
    turns: tuple[ConversationTurn, ...] = ()
    tool_calls: tuple[ConversationToolCall, ...] = ()
    research_links: tuple[ConversationResearchLink, ...] = ()
    event_cursor: int = Field(ge=0)


class ConversationMessageAcceptedResponse(_StrictTransport):
    message: ContextMessage
    turn: ConversationTurn
    conversation_revision: int = Field(ge=0)
    event_cursor: int = Field(ge=1)


class ConversationStatusResponse(_StrictTransport):
    conversation: Conversation
    event_cursor: int = Field(ge=1)


class RunCreateRequest(_StrictTransport):
    question: str = Field(min_length=1, max_length=20000)
    release: int = Field(default=17, ge=1)
    answer_language: str = Field(default="English", min_length=1, max_length=64)
    detail_level: str = Field(default="standard", min_length=1, max_length=64)
    budget: "RunBudgetRequest" = Field(default_factory=lambda: RunBudgetRequest())
    elapsed_time_seconds: int | None = Field(default=None, ge=1, le=86400)
    development_question_id: str | None = Field(default=None, max_length=128)


class RunBudgetRequest(_StrictTransport):
    max_iterations: int = Field(default=4, ge=1)
    max_tool_calls: int = Field(default=12, ge=1)
    max_gap_tool_calls: int = Field(default=6, ge=0)
    max_replans: int = Field(default=3, ge=0)


class RunRestartRequest(_StrictTransport):
    elapsed_time_seconds: int | None = Field(default=None, ge=1, le=86400)


class WorkspaceContextResponse(_StrictTransport):
    user_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    role: Literal["member", "reviewer", "admin"]
    session_expires_at: datetime


class WorkspaceMembershipResponse(_StrictTransport):
    workspace: Workspace
    membership: Membership


class WorkspaceMemberResponse(_StrictTransport):
    user: User
    membership: Membership


class MemberCreateRequest(_StrictTransport):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=10, max_length=1024)
    role: Literal["member", "reviewer", "admin"] = "member"


class MemberRoleRequest(_StrictTransport):
    role: Literal["member", "reviewer", "admin"]


class MemberRemovedResponse(_StrictTransport):
    status: Literal["removed"] = "removed"
    username: str = Field(min_length=1)


class RunInspectionResponse(_StrictTransport):
    run: ResearchRun
    events: tuple[RunEvent, ...]
    stage_latency_ms: dict[str, float]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    checkpoint_available: bool
    result_id: str | None = Field(default=None, pattern=r"^RESULT-[0-9A-F]{32}$")
    answer: str | None = None
    evidence_ids: tuple[str, ...] = ()
    evidence: tuple[CandidateEvidence, ...] = ()
    evidence_status: EvidenceStatus | None = None
    research_tasks: tuple[ResearchTask, ...] = ()
    task_assessments: tuple[TaskSufficiencyAssessment, ...] = ()
    gaps: tuple[ResearchGap, ...] = ()
    failure: "RunFailureResponse | None" = None


class RunMetadataResponse(_StrictTransport):
    """Content-free Run lifecycle projection safe for workspace operators."""

    run_id: str = Field(pattern=r"^RUN-[0-9A-F]{32}$")
    workspace_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    status: RunStatus
    stop_reason: RunStopReason | None = None
    failure_kind: RunFailureKind | None = None
    failure_retryable: bool | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    research_deadline_at: datetime | None = None
    finalization_deadline_at: datetime | None = None
    deadline_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    retry_of_run_id: str | None = Field(default=None, pattern=r"^RUN-[0-9A-F]{32}$")
    run_attempt: int = Field(default=0, ge=0)


class RunFailureResponse(_StrictTransport):
    summary: str = Field(min_length=1, max_length=500)
    stop_reason: str = Field(min_length=1, max_length=100)
    failure_kind: RunFailureKind | None = None
    retryable: bool = False
    attempt_count: int | None = Field(default=None, ge=1)
    status_code: int | None = Field(default=None, ge=100, le=599)
    last_stage: str | None = Field(default=None, max_length=100)
    last_event_type: str | None = Field(default=None, max_length=100)
    last_event_at: datetime | None = None


class ReviewDecisionRequest(_StrictTransport):
    approve: bool
    reason: str | None = Field(default=None, max_length=1000)


class MemoryTopicResponse(_StrictTransport):
    topic: MemoryTopic
    entries: tuple[MemoryEntry, ...] = ()


class MemoryDashboardResponse(_StrictTransport):
    user_core: tuple[MemoryEntry, ...] = ()
    project_core: tuple[MemoryEntry, ...] = ()
    topics: tuple[MemoryTopicResponse, ...] = ()
    proposals: tuple[MemoryProposal, ...] = ()


class SpecFactDashboardResponse(_StrictTransport):
    facts: tuple[SpecFact, ...] = ()
    proposals: tuple[SpecFactProposal, ...] = ()


class LogoutResponse(_StrictTransport):
    status: Literal["logged_out"] = "logged_out"


__all__ = [
    "ApiErrorResponse",
    "ConversationCreateRequest",
    "ConversationDetailResponse",
    "ConversationMessageAcceptedResponse",
    "ConversationMessageCreateRequest",
    "ConversationStatusRequest",
    "ConversationStatusResponse",
    "LoginRequest",
    "LogoutResponse",
    "MemberCreateRequest",
    "MemberRemovedResponse",
    "MemberRoleRequest",
    "MemoryDashboardResponse",
    "MemoryTopicResponse",
    "ReviewDecisionRequest",
    "RunInspectionResponse",
    "RunMetadataResponse",
    "RunFailureResponse",
    "RunCreateRequest",
    "RunBudgetRequest",
    "RunRestartRequest",
    "SessionResponse",
    "SpecFactDashboardResponse",
    "WorkspaceContextResponse",
    "WorkspaceCreateRequest",
    "WorkspaceMemberResponse",
    "WorkspaceMembershipResponse",
]

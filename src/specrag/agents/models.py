"""Immutable state and audit contracts for bounded Researcher execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from specrag.infrastructure.structured_agent import CodexAgentReceipt
from specrag.models import ResearcherStopReason
from specrag.retrieval.planner import PlannedRetrievalExecution
from specrag.tools import RetrievalToolName


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


AgentToolName = RetrievalToolName | Literal["finish"]


class AgentRetrievalAction(_StrictModel):
    """One requested tool call; unused fields must be empty values."""

    action_id: str = Field(min_length=1)
    tool: AgentToolName
    query_id: str
    lexical_query: str
    semantic_query: str
    clause: str = ""
    target_specifications: tuple[str, ...] = Field(max_length=6)
    seed_evidence_ids: tuple[str, ...] = Field(max_length=4)
    max_results: int = Field(ge=1, le=12)
    rationale: str = Field(min_length=1)
    expected_evidence: str = Field(min_length=1)


class AgentRetrievalDecision(_StrictModel):
    assessment: str = Field(min_length=1)
    actions: tuple[AgentRetrievalAction, ...] = Field(min_length=1, max_length=3)


class AgentActionValidation(_StrictModel):
    action_id: str = Field(min_length=1)
    accepted: bool
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _coherent_acceptance(self) -> "AgentActionValidation":
        if self.accepted == bool(self.errors):
            raise ValueError("accepted must be true exactly when errors is empty")
        return self


class ResearcherActionPlan(_StrictModel):
    """One model-proposed turn plus trusted pre-execution validation."""

    schema_version: Literal["researcher-action-plan/v1"] = (
        "researcher-action-plan/v1"
    )
    input_payload: dict[str, Any]
    decision: AgentRetrievalDecision
    receipt: CodexAgentReceipt
    validations: tuple[AgentActionValidation, ...]
    accepted: bool

    @model_validator(mode="after")
    def _coherent_validation(self) -> "ResearcherActionPlan":
        action_ids = tuple(action.action_id for action in self.decision.actions)
        validation_ids = tuple(item.action_id for item in self.validations)
        if validation_ids != action_ids:
            raise ValueError("action validations must preserve decision order")
        expected = bool(self.validations) and all(
            item.accepted for item in self.validations
        )
        if self.accepted != expected:
            raise ValueError("plan acceptance must match action validations")
        return self


class AgentActionOutcome(_StrictModel):
    round_number: int = Field(ge=1)
    action: AgentRetrievalAction
    status: Literal[
        "completed",
        "partial",
        "no_hits",
        "rejected",
        "unavailable",
    ]
    evidence_ids: tuple[str, ...] = ()
    new_evidence_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class AgentRetrievalTurn(_StrictModel):
    round_number: int = Field(ge=1)
    assessment: str = Field(min_length=1)
    receipt: CodexAgentReceipt
    outcomes: tuple[AgentActionOutcome, ...]


class AgenticRetrievalExecution(_StrictModel):
    schema_version: Literal["agentic-retrieval-execution/v0.1"] = (
        "agentic-retrieval-execution/v0.1"
    )
    status: Literal["completed", "partial", "unavailable"]
    stop_reason: ResearcherStopReason
    rounds_used: int = Field(ge=0)
    search_actions_used: int = Field(ge=0)
    expansion_actions_used: int = Field(ge=0)
    turns: tuple[AgentRetrievalTurn, ...] = ()
    evidence_execution: PlannedRetrievalExecution
    errors: tuple[str, ...] = ()


__all__ = [
    "AgentActionValidation",
    "AgentActionOutcome",
    "AgentRetrievalAction",
    "AgentRetrievalDecision",
    "AgentRetrievalTurn",
    "AgentToolName",
    "AgenticRetrievalExecution",
    "ResearcherActionPlan",
]

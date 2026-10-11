"""Immutable state and audit contracts for bounded Researcher execution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    "AgentActionOutcome",
    "AgentRetrievalAction",
    "AgentRetrievalDecision",
    "AgentRetrievalTurn",
    "AgentToolName",
    "AgenticRetrievalExecution",
]

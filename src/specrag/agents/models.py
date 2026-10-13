"""Immutable state and audit contracts for bounded Researcher execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from specrag.infrastructure.structured_agent import CodexAgentReceipt
from specrag.models import (
    NativeFunctionCall,
    NativeFunctionDefinition,
    ResearcherStopReason,
)
from specrag.retrieval.planner import PlannedRetrievalExecution
from specrag.tools import RetrievalToolName


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentEvidenceSelection(_StrictModel):
    """Candidate evidence retained as the active set for one research task."""

    query_id: str = Field(min_length=1)
    retained_evidence_ids: tuple[str, ...]


class AgentUnresolvedGap(_StrictModel):
    """One explicit evidence gap remaining for a task answer unit."""

    query_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class AgentResearchReview(_StrictModel):
    """Evidence review committed before retrieval or downstream submission."""

    assessment: str = Field(min_length=1)
    evidence_selections: tuple[AgentEvidenceSelection, ...]
    unresolved_gaps: tuple[AgentUnresolvedGap, ...]


AgentControlName = Literal[
    "update_research_state",
    "submit_for_validation",
]


class AgentControlCall(_StrictModel):
    call_id: str = Field(min_length=1)
    name: AgentControlName
    review: AgentResearchReview


class AgentRetrievalCall(_StrictModel):
    """One native retrieval function call after provider JSON parsing."""

    call_id: str = Field(min_length=1)
    tool: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any]


class AgentToolCallValidation(_StrictModel):
    call_id: str = Field(min_length=1)
    accepted: bool
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _coherent_acceptance(self) -> "AgentToolCallValidation":
        if self.accepted == bool(self.errors):
            raise ValueError("accepted must be true exactly when errors is empty")
        return self


class AgentEvidenceSelectionValidation(_StrictModel):
    query_id: str = Field(min_length=1)
    accepted: bool
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _coherent_acceptance(self) -> "AgentEvidenceSelectionValidation":
        if self.accepted == bool(self.errors):
            raise ValueError("accepted must be true exactly when errors is empty")
        return self


class ResearcherActionPlan(_StrictModel):
    """One native model turn plus trusted pre-execution validation."""

    schema_version: Literal["researcher-action-plan/v3"] = (
        "researcher-action-plan/v3"
    )
    input_payload: dict[str, Any]
    native_calls: tuple[NativeFunctionCall, ...]
    control_call: AgentControlCall | None
    retrieval_calls: tuple[AgentRetrievalCall, ...]
    receipt: CodexAgentReceipt
    selection_validations: tuple[AgentEvidenceSelectionValidation, ...] = ()
    validations: tuple[AgentToolCallValidation, ...]
    protocol_errors: tuple[str, ...] = ()
    accepted: bool
    replay_items: tuple[dict[str, Any], ...] = Field(
        default=(),
        exclude=True,
        repr=False,
    )

    @model_validator(mode="after")
    def _coherent_validation(self) -> "ResearcherActionPlan":
        call_ids = tuple(call.call_id for call in self.retrieval_calls)
        validation_ids = tuple(item.call_id for item in self.validations)
        if validation_ids != call_ids:
            raise ValueError("tool validations must preserve retrieval call order")
        expected = not self.protocol_errors and all(
            item.accepted for item in self.validations
        ) and all(
            item.accepted for item in self.selection_validations
        )
        if self.accepted != expected:
            raise ValueError("plan acceptance must match call validations")
        return self


class AgentToolOutcome(_StrictModel):
    round_number: int = Field(ge=1)
    call: AgentRetrievalCall
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
    phase: Literal["retrieval", "finalization"] = "retrieval"
    control_call: AgentControlCall | None
    selection_validations: tuple[AgentEvidenceSelectionValidation, ...] = ()
    receipt: CodexAgentReceipt
    outcomes: tuple[AgentToolOutcome, ...]
    protocol_errors: tuple[str, ...] = ()


class AgenticRetrievalExecution(_StrictModel):
    schema_version: Literal["agentic-retrieval-execution/v0.3"] = (
        "agentic-retrieval-execution/v0.3"
    )
    status: Literal["completed", "partial", "unavailable"]
    stop_reason: ResearcherStopReason
    rounds_used: int = Field(ge=0)
    search_actions_used: int = Field(ge=0)
    expansion_actions_used: int = Field(ge=0)
    turns: tuple[AgentRetrievalTurn, ...] = ()
    evidence_execution: PlannedRetrievalExecution
    candidate_evidence_execution: PlannedRetrievalExecution | None = None
    errors: tuple[str, ...] = ()


def researcher_control_definitions() -> tuple[NativeFunctionDefinition, ...]:
    """Return the two agent-owned control functions for native turns."""

    parameters = AgentResearchReview.model_json_schema()
    return (
        NativeFunctionDefinition(
            name="update_research_state",
            description=(
                "Commit the current evidence review, retained evidence, and "
                "remaining answer-unit gaps before requesting retrieval tools."
            ),
            parameters=parameters,
        ),
        NativeFunctionDefinition(
            name="submit_for_validation",
            description=(
                "Submit the reviewed active evidence to downstream source and "
                "sufficiency validation without claiming that it is complete."
            ),
            parameters=parameters,
        ),
    )


__all__ = [
    "AgentControlCall",
    "AgentControlName",
    "AgentEvidenceSelection",
    "AgentEvidenceSelectionValidation",
    "AgentResearchReview",
    "AgentRetrievalCall",
    "AgentRetrievalTurn",
    "AgentToolCallValidation",
    "AgentToolOutcome",
    "AgentUnresolvedGap",
    "AgenticRetrievalExecution",
    "ResearcherActionPlan",
    "researcher_control_definitions",
]

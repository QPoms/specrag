"""Unified runtime state for LangGraph research workflows."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from specrag.models import (
    CandidateEvidence,
    EvidenceConflict,
    EvidenceSynthesis,
    ResearchBudget,
    ResearchGap,
    ResearchPlan,
    ResearchRouteDecision,
    ResearcherStopReason,
    ResearchTask,
    ResearchWorkerResult,
    SourceValidationResult,
    StopReason,
    TaskSufficiencyAssessment,
    ToolCallRecord,
    ValidatedEvidence,
)


class ResearchState(TypedDict):
    schema_version: Literal["research-state/v1.0"]
    research_id: str
    user_question: str
    route: Literal["plan", "finish"]
    route_decision: ResearchRouteDecision | None
    research_plan: ResearchPlan | None
    research_tasks: tuple[ResearchTask, ...]
    candidate_evidence: tuple[CandidateEvidence, ...]
    source_validations: tuple[SourceValidationResult, ...]
    evidence_conflicts: tuple[EvidenceConflict, ...]
    validated_evidence: tuple[ValidatedEvidence, ...]
    sufficiency_assessments: tuple[TaskSufficiencyAssessment, ...]
    gaps: tuple[ResearchGap, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    budget: ResearchBudget
    iteration: int
    evidence_sufficient: bool | None
    final_answer: str | None
    synthesis: EvidenceSynthesis | None
    memory_knowledge_ids: tuple[str, ...]
    stop_reason: StopReason | None
    last_researcher_stop_reason: ResearcherStopReason | None
    errors: tuple[str, ...]
    researcher_results: Annotated[
        tuple[ResearchWorkerResult, ...], operator.add
    ]
    researcher_results_processed: int


def new_research_state(
    *,
    research_id: str,
    user_question: str,
    budget: ResearchBudget | None = None,
) -> ResearchState:
    if not research_id.strip():
        raise ValueError("research_id must not be blank")
    if not user_question.strip():
        raise ValueError("user_question must not be blank")
    return ResearchState(
        schema_version="research-state/v1.0",
        research_id=research_id,
        user_question=user_question,
        route="plan",
        route_decision=None,
        research_plan=None,
        research_tasks=(),
        candidate_evidence=(),
        source_validations=(),
        evidence_conflicts=(),
        validated_evidence=(),
        sufficiency_assessments=(),
        gaps=(),
        tool_calls=(),
        budget=budget or ResearchBudget(),
        iteration=0,
        evidence_sufficient=None,
        final_answer=None,
        synthesis=None,
        memory_knowledge_ids=(),
        stop_reason=None,
        last_researcher_stop_reason=None,
        errors=(),
        researcher_results=(),
        researcher_results_processed=0,
    )


__all__ = ["ResearchState", "new_research_state"]

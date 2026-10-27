"""Unified runtime state for LangGraph research workflows."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from specrag.models import (
    CandidateEvidence,
    EvidenceSynthesis,
    JudgeReviewRecord,
    ResearchBudget,
    ResearchGap,
    ResearchPlan,
    ResearchRouteDecision,
    ResearcherStopReason,
    ResearcherEvidenceSelectionRecord,
    ResearchTask,
    ResearchTaskExecution,
    ResearchWorkerResult,
    SourceValidationResult,
    StopReason,
    TaskSufficiencyAssessment,
    TaskEvidenceSet,
    ToolCallRecord,
    ValidatedEvidence,
)


class ResearchState(TypedDict):
    schema_version: Literal[
        "research-state/v1.7", "research-state/v1.8", "research-state/v1.9",
        "research-state/v1.10"
    ]
    research_id: str
    user_question: str
    route: Literal["plan", "finish"]
    route_decision: ResearchRouteDecision | None
    research_plan: ResearchPlan | None
    research_tasks: tuple[ResearchTask, ...]
    task_executions: tuple[ResearchTaskExecution, ...]
    candidate_evidence: tuple[CandidateEvidence, ...]
    source_validations: tuple[SourceValidationResult, ...]
    validated_evidence: tuple[ValidatedEvidence, ...]
    sufficiency_assessments: tuple[TaskSufficiencyAssessment, ...]
    gaps: tuple[ResearchGap, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    researcher_evidence_selections: tuple[
        ResearcherEvidenceSelectionRecord, ...
    ]
    task_evidence_sets: tuple[TaskEvidenceSet, ...]
    judge_review_records: tuple[JudgeReviewRecord, ...]
    judge_skipped_task_ids: tuple[str, ...]
    budget: ResearchBudget
    iteration: int
    evidence_sufficient: bool | None
    final_answer: str | None
    synthesis: EvidenceSynthesis | None
    publication_path: Literal[
        "full_synthesis", "qualified_synthesis", "status_only"
    ] | None
    specfact_proposal_ids: tuple[str, ...]
    specfact_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]
    model_receipts: tuple[dict[str, object], ...]
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
    specfact_ids: tuple[str, ...] = (),
    memory_ids: tuple[str, ...] = (),
) -> ResearchState:
    if not research_id.strip():
        raise ValueError("research_id must not be blank")
    if not user_question.strip():
        raise ValueError("user_question must not be blank")
    return ResearchState(
        schema_version="research-state/v1.10",
        research_id=research_id,
        user_question=user_question,
        route="plan",
        route_decision=None,
        research_plan=None,
        research_tasks=(),
        task_executions=(),
        candidate_evidence=(),
        source_validations=(),
        validated_evidence=(),
        sufficiency_assessments=(),
        gaps=(),
        tool_calls=(),
        researcher_evidence_selections=(),
        task_evidence_sets=(),
        judge_review_records=(),
        judge_skipped_task_ids=(),
        budget=budget or ResearchBudget(),
        iteration=0,
        evidence_sufficient=None,
        final_answer=None,
        synthesis=None,
        publication_path=None,
        specfact_proposal_ids=(),
        specfact_ids=specfact_ids,
        memory_ids=memory_ids,
        model_receipts=(),
        stop_reason=None,
        last_researcher_stop_reason=None,
        errors=(),
        researcher_results=(),
        researcher_results_processed=0,
    )


__all__ = ["ResearchState", "new_research_state"]

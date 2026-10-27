"""Deterministic corrective Researcher context from Judge-owned gaps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from specrag.models import CandidateEvidence, EvidenceRequirement, ResearchTask

from .state import ResearchState


@dataclass(frozen=True)
class CorrectiveResearchInput:
    """Trusted state restored before a Judge-gap Researcher execution."""

    shared_context: dict[str, object]
    initial_evidence: tuple[CandidateEvidence, ...]
    initial_active_evidence_ids: tuple[str, ...]
    initial_evidence_requirements: tuple[EvidenceRequirement, ...]


def requirements_for_task(
    requirements: Sequence[EvidenceRequirement], task_id: str
) -> tuple[EvidenceRequirement, ...]:
    return tuple(item for item in requirements if item.task_id == task_id)


def requirements_in_task_order(
    tasks: Sequence[ResearchTask],
    requirements_by_task: Mapping[str, tuple[EvidenceRequirement, ...]],
) -> tuple[EvidenceRequirement, ...]:
    return tuple(
        requirement
        for task in tasks
        for requirement in requirements_by_task[task.task_id]
    )


def build_corrective_research_input(
    state: ResearchState,
    task: ResearchTask,
) -> CorrectiveResearchInput:
    """Expose gaps, valid candidates, and prior attempts without model planning."""

    gaps = tuple(
        gap for gap in state["gaps"] if gap.task_id == task.task_id and not gap.resolved
    )
    if not gaps:
        raise ValueError("corrective research requires an unresolved Judge gap")

    valid_ids = {item.evidence_id for item in state["source_validations"] if item.valid}
    candidates = tuple(
        item
        for item in state["candidate_evidence"]
        if item.task_id == task.task_id and item.evidence_id in valid_ids
    )
    selected_ids = {
        evidence_id for gap in gaps for evidence_id in gap.supporting_evidence_ids
    }
    selected_ids.update(
        item.evidence_id
        for item in state["validated_evidence"]
        if (item.task_id == task.task_id and item.source_valid and item.supports_target)
    )
    active_ids = tuple(
        item.evidence_id for item in candidates if item.evidence_id in selected_ids
    )

    prior_calls = tuple(
        {
            "call_id": call.call_id,
            "tool_name": call.tool_name,
            "lexical_query": call.lexical_query,
            "semantic_query": call.semantic_query,
            "hypothetical_document": call.hypothetical_document,
            "clause": call.clause,
            "target_specifications": list(call.target_specifications),
            "seed_evidence_ids": list(call.seed_evidence_ids),
            "status": call.status,
            "evidence_ids": list(call.evidence_ids),
            "new_evidence_ids": list(call.new_evidence_ids),
            "error": call.error,
        }
        for call in state["tool_calls"]
        if call.task_id == task.task_id
    )
    return CorrectiveResearchInput(
        shared_context={
            "gap_retrieval_mode": "evidence_gap",
            "unresolved_gaps": tuple(gap.model_dump(mode="json") for gap in gaps),
            "judge_supported_evidence_ids": active_ids,
            "source_valid_candidate_count": len(candidates),
            "prior_tool_calls": prior_calls,
        },
        initial_evidence=candidates,
        initial_active_evidence_ids=active_ids,
        initial_evidence_requirements=requirements_for_task(
            state.get("evidence_requirements", ()), task.task_id
        ),
    )


__all__ = [
    "CorrectiveResearchInput",
    "build_corrective_research_input",
    "requirements_for_task",
    "requirements_in_task_order",
]

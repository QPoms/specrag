"""Deterministic regression metrics for Conversation Agent traces."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from specrag.models import ConversationMessageContent


class ConversationEvaluationCase(BaseModel):
    """Reviewed expectation plus one captured Conversation Turn trace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    expected_action: Literal["direct", "start_research", "historical_readback"]
    invoked_tools: tuple[str, ...] = ()
    response: ConversationMessageContent
    expected_conditions: tuple[str, ...] = ()
    submitted_conditions: tuple[str, ...] = ()
    expected_result_ids: tuple[str, ...] = ()
    expected_citations: tuple[str, ...] = ()
    available_evidence_ids: tuple[str, ...] = ()
    required_context_refs: tuple[str, ...] = ()
    recovered_context_refs: tuple[str, ...] = ()
    compacted: bool = False


class ConversationEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    routing_correct: bool
    normative_source_valid: bool
    conditions_preserved: bool
    result_selection_correct: bool
    citation_fidelity: bool
    context_carry_over: bool
    compaction_recovery: bool

    @property
    def passed(self) -> bool:
        return all((
            self.routing_correct,
            self.normative_source_valid,
            self.conditions_preserved,
            self.result_selection_correct,
            self.citation_fidelity,
            self.context_carry_over,
            self.compaction_recovery,
        ))


def evaluate_conversation_case(
    case: ConversationEvaluationCase,
) -> ConversationEvaluation:
    response = case.response
    tools = set(case.invoked_tools)
    if case.expected_action == "direct":
        routing = (
            "start_research" not in tools
            and response.source_kind == "direct_conversation"
        )
    elif case.expected_action == "start_research":
        routing = (
            "start_research" in tools
            and response.source_kind == "research_result"
        )
    else:
        routing = (
            "start_research" not in tools
            and response.source_kind == "historical_result_readback"
        )

    evidence_required = case.expected_action != "direct"
    normative_source_valid = (
        response.source_kind == "direct_conversation"
        if not evidence_required
        else bool(response.research_result_ids)
        and response.source_kind in {"research_result", "historical_result_readback"}
    )
    expected_refs = set(case.required_context_refs)
    recovered_refs = set(case.recovered_context_refs)
    context_carry_over = expected_refs <= recovered_refs
    return ConversationEvaluation(
        case_id=case.case_id,
        routing_correct=routing,
        normative_source_valid=normative_source_valid,
        conditions_preserved=(
            case.submitted_conditions == case.expected_conditions
        ),
        result_selection_correct=(
            response.research_result_ids == case.expected_result_ids
        ),
        citation_fidelity=(
            response.citations == case.expected_citations
            and set(response.citations) <= set(case.available_evidence_ids)
        ),
        context_carry_over=context_carry_over,
        compaction_recovery=(not case.compacted or context_carry_over),
    )


def summarize_conversation_evaluations(
    evaluations: tuple[ConversationEvaluation, ...],
) -> dict[str, float | int]:
    total = len(evaluations)
    passed = sum(item.passed for item in evaluations)
    return {
        "cases": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
    }


__all__ = [
    "ConversationEvaluation",
    "ConversationEvaluationCase",
    "evaluate_conversation_case",
    "summarize_conversation_evaluations",
]

"""Convert trusted retrieval-tool results into Researcher evidence candidates."""

from __future__ import annotations

from typing import Any

from specrag.models import NativeFunctionCallOutput, ResearchTask
from specrag.retrieval.planner import PlannedRetrievalHit
from specrag.tools import RetrievalToolResult, ToolEvidence
from .models import AgentRetrievalTurn, AgentToolOutcome


def _evidence_score(evidence: ToolEvidence) -> float:
    if evidence.score is None:
        return 1.0 / evidence.rank
    if evidence.score_kind == "semantic_distance":
        return 1.0 / (1.0 + max(evidence.score, 0.0))
    return evidence.score


def planned_hits_from_tool(
    result: RetrievalToolResult,
    *,
    task: ResearchTask,
) -> tuple[PlannedRetrievalHit, ...]:
    """Preserve source provenance while adapting tool evidence to the ledger."""

    hits: list[PlannedRetrievalHit] = []
    for evidence in result.evidence:
        score = _evidence_score(evidence)
        hits.append(PlannedRetrievalHit(
            hit_id=evidence.evidence_id,
            chunk_id=evidence.chunk_id,
            collection_rank=evidence.rank,
            specification=evidence.specification,
            release=evidence.release,
            version=evidence.version,
            source_path=evidence.source_path,
            document_sha256=evidence.document_sha256,
            heading_path=evidence.heading_path,
            line_start=evidence.line_start,
            line_end=evidence.line_end,
            excerpt=evidence.excerpt,
            evidence_roles=(task.objective,),
            probe_ids=(result.request_id,),
            retrieval_sources=(f"tool:{result.tool_name}",),
            retrieval_hop=(
                2
                if result.tool_name in {
                    "expand_source_context",
                    "navigate_clause",
                    "cross_reference_lookup",
                }
                else 1
            ),
            semantic_rank=(
                evidence.rank
                if result.tool_name in {"semantic_search", "hybrid_search"}
                else None
            ),
            lexical_rank=(
                evidence.rank
                if result.tool_name in {"keyword_search", "hybrid_search"}
                else None
            ),
            rrf_score=(
                max(evidence.score or 0.0, 0.0)
                if evidence.score_kind == "rrf"
                else 0.0
            ),
            route_score=0.0,
            anchor_score=0.0,
            heading_score=0.0,
            agreement_score=0.0,
            noise_penalty=0.0,
            reranker_score=(
                evidence.score
                if evidence.score_kind == "cross_encoder"
                else None
            ),
            final_score=score,
        ))
    return tuple(hits)


def research_history(turns: list[AgentRetrievalTurn]) -> list[dict[str, Any]]:
    return [{
        "round": outcome.round_number,
        "call_id": outcome.call.call_id,
        "tool": outcome.call.tool,
        "query_id": outcome.call.arguments.get("query_id", ""),
        "status": outcome.status,
        "new_evidence_ids": list(outcome.new_evidence_ids),
        "errors": list(outcome.errors),
        "next_cursor": outcome.next_cursor,
    } for turn in turns for outcome in turn.outcomes]


def native_tool_outputs(
    proposal: Any,
    outcomes: tuple[AgentToolOutcome, ...],
    *,
    control_errors: tuple[str, ...] = (),
) -> tuple[NativeFunctionCallOutput, ...]:
    outcome_by_id = {outcome.call.call_id: outcome for outcome in outcomes}
    control_id = proposal.control_call.call_id if proposal.control_call is not None else None
    outputs: list[NativeFunctionCallOutput] = []
    for native_call in proposal.native_calls:
        if native_call.call_id == control_id:
            content = {
                "status": "rejected" if control_errors else "completed",
                "errors": list(control_errors),
            }
        elif (outcome := outcome_by_id.get(native_call.call_id)) is None:
            content = {
                "status": "rejected",
                "errors": list(proposal.protocol_errors) or [
                    "call was not recognized by the Researcher protocol"
                ],
            }
        else:
            content = {
                "status": outcome.status,
                "tool": outcome.call.tool,
                "query_id": outcome.call.arguments.get("query_id", ""),
                "evidence_ids": list(outcome.evidence_ids),
                "new_evidence_ids": list(outcome.new_evidence_ids),
                "errors": list(outcome.errors),
                "tool_result_id": outcome.tool_result_id,
                "next_cursor": outcome.next_cursor,
            }
        outputs.append(NativeFunctionCallOutput(call_id=native_call.call_id, output=content))
    return tuple(outputs)


__all__ = ["native_tool_outputs", "planned_hits_from_tool", "research_history"]

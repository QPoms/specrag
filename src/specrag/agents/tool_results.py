"""Convert trusted retrieval-tool results into Researcher evidence candidates."""

from __future__ import annotations

from specrag.models import ResearchTask
from specrag.retrieval.planner import PlannedRetrievalHit
from specrag.tools import RetrievalToolResult, ToolEvidence


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
                2 if result.tool_name == "expand_source_context" else 1
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


__all__ = ["planned_hits_from_tool"]

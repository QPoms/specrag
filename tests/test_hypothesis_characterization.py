from __future__ import annotations

from specrag.retrieval.hypothesis import merge_parallel_retrieval_channels
from specrag.retrieval.planner import (
    PlannedRetrievalExecution,
    PlannedRetrievalHit,
    PlannedRetrievalTaskResult,
)


def _hit(*, hit_id: str, probe_id: str) -> PlannedRetrievalHit:
    return PlannedRetrievalHit(
        hit_id=hit_id,
        chunk_id="shared-chunk",
        collection_rank=1,
        specification="24.501",
        version="17.1.0",
        source_path="TS_24.501.md",
        document_sha256="A" * 64,
        heading_path=("5 Procedure",),
        line_start=10,
        line_end=12,
        excerpt="The Registration Request carries the requested information.",
        evidence_roles=("carrier",),
        probe_ids=(probe_id,),
        matched_anchors=("Registration Request",),
        rrf_score=0.02,
        route_score=0.004,
        anchor_score=0.004,
        heading_score=0.0,
        agreement_score=0.0,
        noise_penalty=0.0,
        final_score=0.028,
    )


def _execution(hit: PlannedRetrievalHit) -> PlannedRetrievalExecution:
    return PlannedRetrievalExecution(
        status="completed",
        embedding_model="test-embedding",
        corpus_version_policy="highest_version_within_release",
        corpus_release=17,
        tasks=(
            PlannedRetrievalTaskResult(
                retrieval_id="RET-Q1",
                query_id="Q1",
                planner_action="planned",
                status="completed",
                hits=(hit,),
            ),
        ),
    )


def test_parallel_channels_deduplicate_chunk_and_preserve_provenance() -> None:
    literal = _execution(_hit(hit_id="A:1", probe_id="Q1-R1"))
    hypothesis = _execution(_hit(hit_id="B:1", probe_id="H1-R1-Q1"))

    merged = merge_parallel_retrieval_channels(
        literal,
        hypothesis,
        route_a_quota=1,
        route_b_quota=1,
        max_hits_per_question=2,
    )

    assert len(merged.tasks[0].hits) == 1
    assert merged.tasks[0].hits[0].chunk_id == "shared-chunk"
    assert merged.tasks[0].hits[0].retrieval_sources == (
        "route_a",
        "route_b:H1",
        "route_b_requirement:H1:R1",
        "route_b_subquery:H1:R1:Q1",
        "selection:subquery_base",
        "selection:requirement_base",
    )

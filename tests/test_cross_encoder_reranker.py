from __future__ import annotations

from types import SimpleNamespace

from specrag.retrieval.hybrid import MarkdownChunk
from specrag.retrieval.planner import (
    PlannedDiscoveryRetrieval,
    ReferenceExpansionPolicy,
    RetrievalProbe,
    SpecificationRoute,
    _execute_task,
)


class _Bm25:
    def search(self, *args, **kwargs):
        return [("C1", 2.0), ("C2", 1.0)]


class _Retriever:
    def __init__(self) -> None:
        self.chunks = (
            _chunk("C1", "lower reranker relevance"),
            _chunk("C2", "higher reranker relevance"),
        )
        self.documents = self.chunks
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.cfg = SimpleNamespace(
            rrf_k=60,
            lexical_candidates_per_collection=20,
        )
        self.bm25 = _Bm25()

    def _dense(self, *args, **kwargs):
        return [("C1", 0.1), ("C2", 0.2)]


class _Reranker:
    model_name = "fake-minilm"
    deterministic_weight = 0.0

    def predict(self, query, passages):
        return tuple(1.0 if "higher" in passage else 0.0 for passage in passages)


class _FailingReranker(_Reranker):
    def predict(self, query, passages):
        raise RuntimeError("synthetic failure")


def _chunk(chunk_id: str, text: str) -> MarkdownChunk:
    return MarkdownChunk(
        chunk_id=chunk_id,
        source_collection="3gpp_markdown",
        specification="24.501",
        version="17.1.0",
        source_path="TS_24.501.md",
        document_sha256="A" * 64,
        heading_path=("5 Procedure",),
        line_start=1,
        line_end=2,
        text=text,
    )


def _item() -> PlannedDiscoveryRetrieval:
    return PlannedDiscoveryRetrieval(
        retrieval_id="RET-D1",
        query_id="D1",
        claim_id="C1",
        angle="relation",
        original_question="What relation is defined?",
        planner_action="planned",
        reference_expansion=ReferenceExpansionPolicy(enabled=False),
        retrieval_probes=(
            RetrievalProbe(
                probe_id="D1-H1-R1-Q1",
                evidence_role="relation",
                lexical_query="relation",
                semantic_query="What relation is defined?",
                routes=(
                    SpecificationRoute(
                        specification="24.501",
                        weight=1.0,
                        reason="test",
                        source="agent_hypothesis",
                    ),
                ),
                anchors=("relation",),
            ),
        ),
    )


def test_optional_reranker_reorders_and_records_scores() -> None:
    result = _execute_task(
        _Retriever(),
        _item(),
        final_hits=2,
        ranking_mode="rrf_anchor_single_if_full",
        reranker=_Reranker(),
    )

    assert result.reranker_status == "applied"
    assert [hit.chunk_id for hit in result.hits] == ["C2", "C1"]
    assert result.hits[0].reranker_score == 1.0
    assert result.hits[0].deterministic_score is not None
    assert result.hits[0].reranker_model == "fake-minilm"


def test_reranker_failure_falls_back_without_failing_task() -> None:
    result = _execute_task(
        _Retriever(),
        _item(),
        final_hits=2,
        ranking_mode="rrf_anchor_single_if_full",
        reranker=_FailingReranker(),
    )

    assert result.status == "completed"
    assert result.reranker_status == "fallback"
    assert "synthetic failure" in (result.reranker_error or "")
    assert [hit.chunk_id for hit in result.hits] == ["C1", "C2"]


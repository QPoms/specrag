from __future__ import annotations

from types import SimpleNamespace

from specrag.config import AppConfig, RetrievalCfg
from specrag.retrieval.hybrid import CorpusScope, MarkdownChunk
from specrag.retrieval.planner import (
    DiscoveryRetrievalPlanV3,
    PlannedDiscoveryRetrieval,
    ReferenceExpansionPolicy,
    RetrievalProbe,
    SpecificationRoute,
    _execute_task,
    execute_planned_discovery_retrieval,
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
        self.fingerprint = "F" * 64
        self.index_manifest_path = ".local/test/manifest.json"
        self.corpus_scope = CorpusScope(
            version_policy="highest_version_within_release",
            release=16,
        )

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


def _explicit_anchor_item() -> PlannedDiscoveryRetrieval:
    return PlannedDiscoveryRetrieval(
        retrieval_id="RET-D2",
        query_id="D2",
        claim_id="C2",
        angle="carrier",
        original_question="Which message carries Registration Request?",
        planner_action="planned",
        reference_expansion=ReferenceExpansionPolicy(enabled=False),
        retrieval_probes=(
            RetrievalProbe(
                probe_id="D2-R1-Q1",
                evidence_role="carrier",
                lexical_query="Registration Request message",
                semantic_query="Which message carries Registration Request?",
                routes=(
                    SpecificationRoute(
                        specification="24.501",
                        weight=1.0,
                        reason="explicit question anchor",
                        source="explicit_query",
                    ),
                ),
                anchors=("Registration Request",),
            ),
        ),
    )


def test_explicit_anchor_recall_preserves_match_and_provenance() -> None:
    retriever = _Retriever()
    retriever.chunks = (
        _chunk("C1", "The Registration Request message contains the requested IE."),
        _chunk("C2", "Unrelated procedure text."),
    )
    retriever.documents = retriever.chunks
    retriever.chunk_by_id = {chunk.chunk_id: chunk for chunk in retriever.chunks}

    result = _execute_task(
        retriever,
        _explicit_anchor_item(),
        final_hits=2,
    )

    assert result.status == "completed"
    assert result.hits[0].chunk_id == "C1"
    assert result.hits[0].matched_anchors == ("Registration Request",)
    assert result.hits[0].source_path == "TS_24.501.md"
    assert result.hits[0].version == "17.1.0"


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


def test_planned_execution_records_the_retrievers_actual_release_scope() -> None:
    retriever = _Retriever()
    result = execute_planned_discovery_retrieval(
        DiscoveryRetrievalPlanV3(retrievals=(_item(),)),
        app_config=AppConfig(retrieval=RetrievalCfg()),
        retriever=retriever,
        final_hits=2,
    )

    assert result.status == "completed"
    assert result.corpus_version_policy == "highest_version_within_release"
    assert result.corpus_release == 16

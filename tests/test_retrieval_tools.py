from __future__ import annotations

from types import SimpleNamespace

from specrag.retrieval.models import CorpusScope, MarkdownChunk, RetrievalHit
from specrag.tools import RetrievalToolRequest, build_retrieval_tool_registry


def _chunk(chunk_id: str, line_start: int, text: str) -> MarkdownChunk:
    return MarkdownChunk(
        chunk_id=chunk_id,
        source_collection="3gpp_markdown",
        specification="24.501",
        version="17.1.0",
        source_path="data/3gpp/24.501/v17.1.0/source.md",
        document_sha256="A" * 64,
        heading_path=("5.1Registration procedure",),
        line_start=line_start,
        line_end=line_start + 1,
        text=text,
    )


def _hit(chunk: MarkdownChunk) -> RetrievalHit:
    return RetrievalHit(
        hit_id="H1",
        chunk_id=chunk.chunk_id,
        collection_rank=1,
        source_collection="3gpp_markdown",
        specification=chunk.specification,
        version=chunk.version,
        source_path=chunk.source_path,
        document_sha256=chunk.document_sha256,
        heading_path=chunk.heading_path,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        excerpt=chunk.text,
        semantic_rank=1,
        semantic_distance=0.1,
        lexical_rank=1,
        lexical_score=3.0,
        rrf_score=0.03,
    )


class _Reranker:
    def predict(self, query, passages):
        assert query == "registration procedure"
        return tuple(float(index) for index, _ in enumerate(passages, 1))


def _registry():
    first = _chunk("C1", 10, "The UE shall initiate registration.")
    second = _chunk("C2", 12, "The network shall process registration.")
    chunks = (first, second)
    retriever = SimpleNamespace(
        corpus_scope=CorpusScope(
            version_policy="highest_version_within_release",
            release=17,
        ),
        chunks=chunks,
        chunk_by_id={chunk.chunk_id: chunk for chunk in chunks},
        keyword_candidates=lambda query, **kwargs: [("C1", 3.0)],
        semantic_candidates=lambda query, **kwargs: [("C2", 0.1)],
        hybrid_hits=lambda request_id, query, **kwargs: ((_hit(first),), ()),
    )
    return build_retrieval_tool_registry(retriever, reranker=_Reranker())


def _request(**updates) -> RetrievalToolRequest:
    values = {
        "request_id": "REQ1",
        "query_id": "Q1",
        "query": "registration",
        "limit": 4,
    }
    values.update(updates)
    return RetrievalToolRequest(**values)


def test_registry_exposes_all_retrieval_capabilities_and_schemas() -> None:
    registry = _registry()

    assert registry.names == (
        "clause_lookup",
        "expand_same_section",
        "hybrid_search",
        "keyword_search",
        "rerank",
        "semantic_search",
    )
    bundle = registry.schema_bundle()
    assert bundle["request"]["title"] == "RetrievalToolRequest"
    assert len(bundle["capabilities"]) == 6


def test_keyword_tool_returns_source_grounded_evidence() -> None:
    result = _registry().invoke("keyword_search", _request())

    assert result.status == "completed"
    assert result.evidence[0].chunk_id == "C1"
    assert result.evidence[0].score_kind == "bm25"
    assert result.evidence[0].release == 17


def test_semantic_tool_uses_semantic_query() -> None:
    result = _registry().invoke(
        "semantic_search",
        _request(query="", semantic_query="registration procedure"),
    )

    assert result.status == "completed"
    assert result.evidence[0].chunk_id == "C2"
    assert result.evidence[0].score_kind == "semantic_distance"


def test_hybrid_tool_preserves_rrf_score() -> None:
    result = _registry().invoke(
        "hybrid_search",
        _request(semantic_query="registration procedure"),
    )

    assert result.status == "completed"
    assert result.evidence[0].chunk_id == "C1"
    assert result.evidence[0].score_kind == "rrf"
    assert result.evidence[0].score == 0.03


def test_rerank_tool_accepts_only_trusted_seed_chunks() -> None:
    result = _registry().invoke(
        "rerank",
        _request(
            query="registration procedure",
            seed_chunk_ids=("C1", "C2"),
            limit=1,
        ),
    )

    assert result.status == "completed"
    assert result.evidence[0].chunk_id == "C2"
    assert result.evidence[0].score_kind == "cross_encoder"


def test_clause_lookup_tool_matches_exact_heading_clause() -> None:
    result = _registry().invoke(
        "clause_lookup",
        _request(query="", clause="5.1"),
    )

    assert result.status == "completed"
    assert [item.chunk_id for item in result.evidence] == ["C1", "C2"]


def test_clause_lookup_tool_matches_title_and_normalizes_heading_spacing() -> None:
    registry = _registry()

    title_result = registry.invoke(
        "clause_lookup",
        _request(query="", clause="Registration procedure"),
    )
    combined_result = registry.invoke(
        "clause_lookup",
        _request(query="", clause="5.1 Registration procedure"),
    )

    assert [item.chunk_id for item in title_result.evidence] == ["C1", "C2"]
    assert [item.chunk_id for item in combined_result.evidence] == ["C1", "C2"]


def test_same_section_tool_expands_from_trusted_seed() -> None:
    result = _registry().invoke(
        "expand_same_section",
        _request(query="", seed_chunk_ids=("C1",)),
    )

    assert result.status == "completed"
    assert [item.chunk_id for item in result.evidence] == ["C2"]

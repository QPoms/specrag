from __future__ import annotations

from types import SimpleNamespace

from specrag.retrieval.clauses import ClauseIndex
from specrag.retrieval.models import (
    ClauseRecord,
    ClauseReference,
    CorpusScope,
    MarkdownChunk,
    RetrievalHit,
)
from specrag.tools import RetrievalToolRequest, build_retrieval_tool_registry


def _chunk(
    chunk_id: str,
    line_start: int,
    text: str,
    *,
    heading_path: tuple[str, ...] = ("5.1Registration procedure",),
) -> MarkdownChunk:
    return MarkdownChunk(
        chunk_id=chunk_id,
        source_collection="3gpp_markdown",
        specification="24.501",
        version="17.1.0",
        source_path="data/3gpp/24.501/v17.1.0/source.md",
        document_sha256="A" * 64,
        heading_path=heading_path,
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


def _record(
    clause_id: str,
    chunks: tuple[MarkdownChunk, ...],
    *,
    heading_path: tuple[str, ...],
    parent_clause_id: str | None = None,
    child_clause_ids: tuple[str, ...] = (),
    direct_chunk_ids: tuple[str, ...] | None = None,
    cross_references: tuple[ClauseReference, ...] = (),
) -> ClauseRecord:
    direct = (
        tuple(chunk.chunk_id for chunk in chunks)
        if direct_chunk_ids is None
        else direct_chunk_ids
    )
    return ClauseRecord(
        specification="24.501",
        version="17.1.0",
        release=17,
        source_path="data/3gpp/24.501/v17.1.0/source.md",
        document_sha256="A" * 64,
        clause_id=clause_id,
        title=heading_path[-1].removeprefix(clause_id),
        heading_path=heading_path,
        heading_level=len(heading_path),
        line_start=min((chunk.line_start for chunk in chunks), default=1),
        direct_line_end=max((chunk.line_end for chunk in chunks), default=1),
        line_end=max((chunk.line_end for chunk in chunks), default=1),
        parent_clause_id=parent_clause_id,
        child_clause_ids=child_clause_ids,
        direct_chunk_ids=direct,
        chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        cross_references=cross_references,
    )


class _Reranker:
    def predict(self, query, passages):
        assert query == "registration procedure"
        return tuple(float(index) for index, _ in enumerate(passages, 1))


def _registry(
    semantic_queries: list[str] | None = None,
    hybrid_queries: list[tuple[str, str | None]] | None = None,
):
    first = _chunk("C1", 10, "The UE shall initiate registration.")
    second = _chunk("C2", 12, "The network shall process registration.")
    chunks = (first, second)
    clause_index = ClauseIndex((
        _record(
            "5.1",
            chunks,
            heading_path=first.heading_path,
            cross_references=(ClauseReference(
                raw_text="clause 5.1",
                line_number=10,
                target_clause_id="5.1",
                target_specification="24.501",
            ),),
        ),
    ), chunks)
    retriever = SimpleNamespace(
        corpus_scope=CorpusScope(
            version_policy="highest_version_within_release",
            release=17,
        ),
        chunks=chunks,
        chunk_by_id={chunk.chunk_id: chunk for chunk in chunks},
        clause_index=clause_index,
        keyword_candidates=lambda query, **kwargs: [("C1", 3.0)],
        semantic_candidates=lambda query, **kwargs: (
            semantic_queries.append(query) if semantic_queries is not None else None,
            [("C2", 0.1)],
        )[1],
        hybrid_hits=lambda request_id, query, **kwargs: (
            hybrid_queries.append((query, kwargs.get("semantic_query")))
            if hybrid_queries is not None
            else None,
            ((_hit(first),), ()),
        )[1],
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
        "cross_reference_lookup",
        "expand_source_context",
        "hybrid_search",
        "hyde_search",
        "keyword_search",
        "navigate_clause",
        "rerank",
        "semantic_search",
    )
    bundle = registry.schema_bundle()
    assert len(bundle["tools"]) == 9
    assert len(bundle["capabilities"]) == 9
    definitions = {item["name"]: item for item in bundle["tools"]}
    hybrid_properties = definitions["hybrid_search"]["parameters"]["properties"]
    assert set(hybrid_properties) == {
        "query_id",
        "max_results",
        "target_specifications",
        "lexical_query",
        "semantic_query",
    }
    assert "hypothetical_document" not in hybrid_properties


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


def test_hyde_tool_fuses_lexical_and_hypothetical_queries() -> None:
    hybrid_queries: list[tuple[str, str | None]] = []
    result = _registry(hybrid_queries=hybrid_queries).invoke(
        "hyde_search",
        _request(
            query="registration procedure",
            hypothetical_document=(
                "The UE shall initiate the registration procedure."
            ),
        ),
    )

    assert hybrid_queries == [(
        "registration procedure",
        "The UE shall initiate the registration procedure.",
    )]
    assert result.status == "completed"
    assert result.evidence[0].chunk_id == "C1"
    assert result.evidence[0].match_kind == "hyde_search"
    assert result.evidence[0].score_kind == "rrf"


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


def test_clause_lookup_paginates_long_direct_clause() -> None:
    registry = _registry()

    first = registry.invoke(
        "clause_lookup",
        _request(query="", clause="5.1", limit=1, cursor=0),
    )
    second = registry.invoke(
        "clause_lookup",
        _request(query="", clause="5.1", limit=1, cursor=1),
    )

    assert [item.chunk_id for item in first.evidence] == ["C1"]
    assert first.next_cursor == 1
    assert [item.chunk_id for item in second.evidence] == ["C2"]
    assert second.next_cursor is None


def test_cross_reference_tool_requires_observed_index_edge() -> None:
    registry = _registry()

    result = registry.invoke(
        "cross_reference_lookup",
        _request(
            query="",
            seed_chunk_ids=("C1",),
            reference_specification="24.501",
            reference_clause="5.1",
            limit=1,
        ),
    )
    rejected = registry.invoke(
        "cross_reference_lookup",
        _request(
            query="",
            seed_chunk_ids=("C1",),
            reference_specification="24.501",
            reference_clause="5.2",
        ),
    )

    assert [item.chunk_id for item in result.evidence] == ["C1"]
    assert result.next_cursor == 1
    assert rejected.status == "rejected"


def test_clause_navigation_pages_direct_children_from_observed_parent() -> None:
    parent_chunk = _chunk(
        "PARENT", 10, "Parent content.", heading_path=("8.4.9Parent",)
    )
    child_a = _chunk(
        "CHILD-A", 20, "First child.",
        heading_path=("8.4.9Parent", "8.4.9.1First child"),
    )
    child_b = _chunk(
        "CHILD-B", 30, "Second child.",
        heading_path=("8.4.9Parent", "8.4.9.2Second child"),
    )
    chunks = (parent_chunk, child_a, child_b)
    clause_index = ClauseIndex((
        _record(
            "8.4.9", chunks, heading_path=parent_chunk.heading_path,
            child_clause_ids=("8.4.9.1", "8.4.9.2"),
            direct_chunk_ids=("PARENT",),
        ),
        _record(
            "8.4.9.1", (child_a,), heading_path=child_a.heading_path,
            parent_clause_id="8.4.9",
        ),
        _record(
            "8.4.9.2", (child_b,), heading_path=child_b.heading_path,
            parent_clause_id="8.4.9",
        ),
    ), chunks)
    retriever = SimpleNamespace(
        corpus_scope=CorpusScope(release=17),
        chunks=chunks,
        chunk_by_id={chunk.chunk_id: chunk for chunk in chunks},
        clause_index=clause_index,
        keyword_candidates=lambda query, **kwargs: [],
        semantic_candidates=lambda query, **kwargs: [],
        hybrid_hits=lambda request_id, query, **kwargs: ((), ()),
    )
    registry = build_retrieval_tool_registry(retriever, reranker=_Reranker())

    first = registry.invoke(
        "navigate_clause",
        _request(
            query="", seed_chunk_ids=("PARENT",),
            clause_relation="children", limit=1,
        ),
    )
    second = registry.invoke(
        "navigate_clause",
        _request(
            query="", seed_chunk_ids=("PARENT",),
            clause_relation="children", cursor=1, limit=1,
        ),
    )

    assert [item.chunk_id for item in first.evidence] == ["CHILD-A"]
    assert first.next_cursor == 1
    assert [item.chunk_id for item in second.evidence] == ["CHILD-B"]
    assert second.next_cursor is None


def test_source_context_tool_expands_from_trusted_seed() -> None:
    result = _registry().invoke(
        "expand_source_context",
        _request(query="", seed_chunk_ids=("C1",)),
    )

    assert result.status == "completed"
    assert [item.chunk_id for item in result.evidence] == ["C2"]


def test_source_context_tool_includes_adjacent_sibling_subsection() -> None:
    successful = _chunk(
        "SUCCESS",
        20,
        "The peer shall acknowledge the accepted change.",
        heading_path=("8.4.9Mobility Settings Change", "8.4.9.2Successful Operation"),
    )
    unsuccessful = _chunk(
        "FAILURE",
        30,
        "The peer shall send MOBILITY CHANGE FAILURE when it refuses.",
        heading_path=("8.4.9Mobility Settings Change", "8.4.9.3Unsuccessful Operation"),
    )
    retriever = SimpleNamespace(
        corpus_scope=CorpusScope(release=17),
        chunks=(successful, unsuccessful),
        chunk_by_id={
            successful.chunk_id: successful,
            unsuccessful.chunk_id: unsuccessful,
        },
        clause_index=ClauseIndex((
            _record(
                "8.4.9",
                (successful, unsuccessful),
                heading_path=("8.4.9Mobility Settings Change",),
                child_clause_ids=("8.4.9.2", "8.4.9.3"),
                direct_chunk_ids=(),
            ),
            _record(
                "8.4.9.2",
                (successful,),
                heading_path=successful.heading_path,
                parent_clause_id="8.4.9",
            ),
            _record(
                "8.4.9.3",
                (unsuccessful,),
                heading_path=unsuccessful.heading_path,
                parent_clause_id="8.4.9",
            ),
        ), (successful, unsuccessful)),
        keyword_candidates=lambda query, **kwargs: [],
        semantic_candidates=lambda query, **kwargs: [],
        hybrid_hits=lambda request_id, query, **kwargs: ((), ()),
    )
    registry = build_retrieval_tool_registry(retriever, reranker=_Reranker())

    result = registry.invoke(
        "expand_source_context",
        _request(query="", seed_chunk_ids=("SUCCESS",)),
    )

    assert result.status == "completed"
    assert [item.chunk_id for item in result.evidence] == ["FAILURE"]

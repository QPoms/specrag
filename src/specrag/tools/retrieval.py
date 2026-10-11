"""Thin tool adapters over the active specification retrieval implementation."""

from __future__ import annotations

from typing import Iterable

from specrag.retrieval.hybrid import HybridMarkdownRetriever
from specrag.retrieval.index import searchable_text
from specrag.retrieval.models import MarkdownChunk, RetrievalHit
from specrag.retrieval.navigation import (
    expand_same_section_chunks,
    lookup_clause_chunks,
)
from specrag.retrieval.reranker import CrossEncoderReranker

from .models import (
    RetrievalToolName,
    RetrievalToolRequest,
    RetrievalToolResult,
    ScoreKind,
    ToolEvidence,
    ToolCapability,
    ToolStatus,
)
from .registry import ToolRegistry


class _RetrieverTool:
    name: RetrievalToolName

    def __init__(self, retriever: HybridMarkdownRetriever) -> None:
        self.retriever = retriever

    def _result(
        self,
        request: RetrievalToolRequest,
        evidence: Iterable[ToolEvidence] = (),
        *,
        errors: tuple[str, ...] = (),
        status: ToolStatus | None = None,
    ) -> RetrievalToolResult:
        materialized = tuple(evidence)
        resolved_status: ToolStatus = status or (
            "partial"
            if materialized and errors
            else "completed"
            if materialized
            else "unavailable"
            if errors
            else "no_hits"
        )
        return RetrievalToolResult(
            request_id=request.request_id,
            query_id=request.query_id,
            tool_name=self.name,
            status=resolved_status,
            corpus_scope=self.retriever.corpus_scope,
            evidence=materialized,
            errors=errors,
        )

    def _chunk_evidence(
        self,
        chunk: MarkdownChunk,
        *,
        request: RetrievalToolRequest,
        rank: int,
        score: float | None = None,
        score_kind: ScoreKind = "source_order",
    ) -> ToolEvidence:
        assert chunk.specification is not None and chunk.version is not None
        return ToolEvidence(
            evidence_id=f"{request.request_id}:{rank}",
            chunk_id=chunk.chunk_id,
            rank=rank,
            match_kind=self.name,
            score=score,
            score_kind=score_kind,
            specification=chunk.specification,
            release=int(chunk.version.split(".", 1)[0]),
            version=chunk.version,
            source_path=chunk.source_path,
            document_sha256=chunk.document_sha256,
            heading_path=chunk.heading_path,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            excerpt=chunk.text,
        )

    def _hit_evidence(
        self,
        hit: RetrievalHit,
        *,
        request: RetrievalToolRequest,
        rank: int,
    ) -> ToolEvidence:
        assert hit.specification is not None and hit.version is not None
        return ToolEvidence(
            evidence_id=f"{request.request_id}:{rank}",
            chunk_id=hit.chunk_id,
            rank=rank,
            match_kind=self.name,
            score=hit.rrf_score,
            score_kind="rrf",
            specification=hit.specification,
            release=int(hit.version.split(".", 1)[0]),
            version=hit.version,
            source_path=hit.source_path,
            document_sha256=hit.document_sha256,
            heading_path=hit.heading_path,
            line_start=hit.line_start,
            line_end=hit.line_end,
            excerpt=hit.excerpt,
        )

    def _reject(
        self,
        request: RetrievalToolRequest,
        message: str,
    ) -> RetrievalToolResult:
        return self._result(request, errors=(message,), status="rejected")


class KeywordSearchTool(_RetrieverTool):
    name: RetrievalToolName = "keyword_search"
    capability = ToolCapability(
        name=name,
        description="Deterministic BM25 search over the active 3GPP corpus scope.",
        required_fields=("query",),
        optional_fields=("target_specifications", "limit"),
        execution_kind="deterministic",
    )

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        if not request.query.strip():
            return self._reject(request, "keyword_search requires a query")
        try:
            candidates = self.retriever.keyword_candidates(
                request.query,
                allowed_specifications=request.target_specifications,
                limit=request.limit,
            )
        except Exception as exc:
            return self._result(
                request,
                errors=(f"{type(exc).__name__}: {exc}",),
            )
        return self._result(
            request,
            (
                self._chunk_evidence(
                    self.retriever.chunk_by_id[chunk_id],
                    request=request,
                    rank=rank,
                    score=score,
                    score_kind="bm25",
                )
                for rank, (chunk_id, score) in enumerate(candidates, 1)
            ),
        )


class SemanticSearchTool(_RetrieverTool):
    name: RetrievalToolName = "semantic_search"
    capability = ToolCapability(
        name=name,
        description="Embedding search over the active 3GPP corpus scope.",
        required_fields=("query",),
        optional_fields=("target_specifications", "limit"),
        execution_kind="model_scored",
    )

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        query = request.semantic_query or request.query
        if not query.strip():
            return self._reject(request, "semantic_search requires a query")
        try:
            candidates = self.retriever.semantic_candidates(
                query,
                allowed_specifications=request.target_specifications,
                limit=request.limit,
            )
        except Exception as exc:
            return self._result(
                request,
                errors=(f"{type(exc).__name__}: {exc}",),
            )
        return self._result(
            request,
            (
                self._chunk_evidence(
                    self.retriever.chunk_by_id[chunk_id],
                    request=request,
                    rank=rank,
                    score=distance,
                    score_kind="semantic_distance",
                )
                for rank, (chunk_id, distance) in enumerate(candidates, 1)
            ),
        )


class HybridSearchTool(_RetrieverTool):
    name: RetrievalToolName = "hybrid_search"
    capability = ToolCapability(
        name=name,
        description="BM25 and semantic retrieval fused with reciprocal rank fusion.",
        required_fields=("query",),
        optional_fields=("semantic_query", "target_specifications", "limit"),
        execution_kind="model_scored",
    )

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        if not request.query.strip():
            return self._reject(request, "hybrid_search requires a query")
        hits, errors = self.retriever.hybrid_hits(
            request.request_id,
            request.query,
            semantic_query=request.semantic_query,
            allowed_specifications=request.target_specifications,
            limit=request.limit,
        )
        return self._result(
            request,
            (
                self._hit_evidence(hit, request=request, rank=rank)
                for rank, hit in enumerate(hits, 1)
            ),
            errors=errors,
        )


class ClauseLookupTool(_RetrieverTool):
    name: RetrievalToolName = "clause_lookup"
    capability = ToolCapability(
        name=name,
        description="Exact clause-heading lookup in indexed 3GPP source chunks.",
        required_fields=("clause",),
        optional_fields=("target_specifications", "limit"),
        execution_kind="deterministic",
    )

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        if request.clause is None:
            return self._reject(request, "clause_lookup requires a clause")
        try:
            chunks = lookup_clause_chunks(
                self.retriever.chunks,
                request.clause,
                specifications=request.target_specifications,
                limit=request.limit,
            )
        except ValueError as exc:
            return self._reject(request, str(exc))
        return self._result(
            request,
            (
                self._chunk_evidence(chunk, request=request, rank=rank)
                for rank, chunk in enumerate(chunks, 1)
            ),
        )


class ExpandSameSectionTool(_RetrieverTool):
    name: RetrievalToolName = "expand_same_section"
    capability = ToolCapability(
        name=name,
        description="Bounded source-order expansion around trusted seed chunks.",
        required_fields=("seed_chunk_ids",),
        optional_fields=("limit",),
        execution_kind="deterministic",
    )

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        try:
            chunks = expand_same_section_chunks(
                self.retriever.chunks,
                request.seed_chunk_ids,
                limit=request.limit,
            )
        except ValueError as exc:
            return self._reject(request, str(exc))
        return self._result(
            request,
            (
                self._chunk_evidence(chunk, request=request, rank=rank)
                for rank, chunk in enumerate(chunks, 1)
            ),
        )


class RerankTool(_RetrieverTool):
    name: RetrievalToolName = "rerank"
    capability = ToolCapability(
        name=name,
        description="Cross-encoder reranking of trusted candidate source chunks.",
        required_fields=("query", "seed_chunk_ids"),
        optional_fields=("limit",),
        execution_kind="model_scored",
    )

    def __init__(
        self,
        retriever: HybridMarkdownRetriever,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        super().__init__(retriever)
        self.reranker = reranker or CrossEncoderReranker()

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        query = request.semantic_query or request.query
        if not query.strip():
            return self._reject(request, "rerank requires a query")
        if not request.seed_chunk_ids:
            return self._reject(request, "rerank requires seed chunk IDs")
        unknown = tuple(
            chunk_id
            for chunk_id in request.seed_chunk_ids
            if chunk_id not in self.retriever.chunk_by_id
        )
        if unknown:
            return self._reject(
                request,
                "unknown seed chunks: " + ", ".join(unknown),
            )
        chunks = tuple(
            self.retriever.chunk_by_id[chunk_id]
            for chunk_id in request.seed_chunk_ids
        )
        try:
            scores = self.reranker.predict(
                query,
                tuple(searchable_text(chunk) for chunk in chunks),
            )
        except Exception as exc:
            return self._result(
                request,
                errors=(f"{type(exc).__name__}: {exc}",),
            )
        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: (-item[1], item[0].chunk_id),
        )[:request.limit]
        return self._result(
            request,
            (
                self._chunk_evidence(
                    chunk,
                    request=request,
                    rank=rank,
                    score=score,
                    score_kind="cross_encoder",
                )
                for rank, (chunk, score) in enumerate(ranked, 1)
            ),
        )


def build_retrieval_tool_registry(
    retriever: HybridMarkdownRetriever,
    *,
    reranker: CrossEncoderReranker | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        KeywordSearchTool(retriever),
        SemanticSearchTool(retriever),
        HybridSearchTool(retriever),
        RerankTool(retriever, reranker),
        ClauseLookupTool(retriever),
        ExpandSameSectionTool(retriever),
    ):
        registry.register(tool)
    return registry


__all__ = [
    "ClauseLookupTool",
    "ExpandSameSectionTool",
    "HybridSearchTool",
    "KeywordSearchTool",
    "RerankTool",
    "SemanticSearchTool",
    "build_retrieval_tool_registry",
]

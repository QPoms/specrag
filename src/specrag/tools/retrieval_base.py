"""Shared result and provenance adapter for trusted retrieval tools."""

from __future__ import annotations

from collections.abc import Iterable

from specrag.retrieval.hybrid import HybridMarkdownRetriever
from specrag.retrieval.models import MarkdownChunk, RetrievalHit

from .models import (
    RetrievalToolName,
    RetrievalToolRequest,
    RetrievalToolResult,
    ScoreKind,
    ToolEvidence,
    ToolStatus,
)


class RetrieverTool:
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
        next_cursor: int | None = None,
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
            next_cursor=next_cursor,
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


__all__ = ["RetrieverTool"]

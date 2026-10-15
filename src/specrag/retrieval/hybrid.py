"""Auditable orchestration of lexical and semantic specification retrieval.

This module owns the hybrid workflow only. Corpus discovery, immutable models,
snapshot persistence, lexical search, semantic search, and rank fusion live in
their single-responsibility modules and are composed here.
"""

from __future__ import annotations

from typing import Any, Literal

from ..config import AppConfig, RetrievalCfg, load_config
from ..infrastructure.vector_store import open_chroma_collection
from .corpus import (
    chunk_source_documents,
    corpus_scope_from_config,
    discover_source_documents,
    evidence_body_text,
    relative_repository_path,
    repository_path,
)
from .fusion import reciprocal_rank_fusion
from .index import (
    index_fingerprint,
    load_chunks,
    searchable_text,
    vector_cache_fingerprint,
    write_snapshot,
)
from .lexical import BM25Index
from .models import (
    CorpusVersionPolicy,
    CorpusScope,
    MarkdownChunk,
    RetrievalExecution,
    RetrievalHit,
    RetrievalIndexManifest,
    RetrievalTaskResult,
    SourceDocument,
    SpecificationVersion,
)
from .semantic import dense_search


class HybridMarkdownRetriever:
    """Local semantic recall plus deterministic BM25/RRF with safe fallback."""

    def __init__(
        self,
        cfg: RetrievalCfg,
        *,
        corpus_scope: CorpusScope | None = None,
    ):
        self.cfg = cfg
        self.corpus_scope = corpus_scope or corpus_scope_from_config(cfg)
        self.documents = discover_source_documents(cfg, self.corpus_scope)
        self.fingerprint = index_fingerprint(
            self.documents,
            cfg,
            self.corpus_scope,
        )
        index_root = repository_path(cfg.index_dir, "retrieval index directory")
        snapshot = index_root / self.fingerprint
        self.index_manifest_path = relative_repository_path(snapshot / "manifest.json")
        chunks_path = snapshot / "chunks.jsonl"
        manifest_path = snapshot / "manifest.json"
        cached_chunks: tuple[MarkdownChunk, ...] | None = None
        if chunks_path.is_file() and manifest_path.is_file():
            try:
                cached_manifest = RetrievalIndexManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                loaded_chunks = load_chunks(chunks_path)
                if (
                    cached_manifest.index_fingerprint == self.fingerprint
                    and cached_manifest.chunk_count == len(loaded_chunks)
                ):
                    cached_chunks = loaded_chunks
            except (OSError, ValueError):
                cached_chunks = None
        self.chunks = cached_chunks or chunk_source_documents(self.documents, cfg)
        manifest = RetrievalIndexManifest(
            index_fingerprint=self.fingerprint,
            embedding_model=cfg.embedding_model,
            corpus_version_policy=self.corpus_scope.version_policy,
            corpus_release=self.corpus_scope.release,
            corpus_versions=self.corpus_scope.versions,
            chunk_characters=cfg.chunk_characters,
            chunk_overlap_characters=cfg.chunk_overlap_characters,
            collection_name="specification_evidence",
            documents=self.documents,
            chunk_count=len(self.chunks),
        )
        if cached_chunks is None:
            write_snapshot(snapshot, self.chunks, manifest)
        self.chunk_by_id = {item.chunk_id: item for item in self.chunks}
        self.bm25 = BM25Index(self.chunks)
        self.active_document_hashes = tuple(sorted({
            document.document_sha256 for document in self.documents
        }))
        self.collection = None
        self.vector_error: str | None = None
        try:
            self.collection = open_chroma_collection(
                index_root / "vector-cache" / vector_cache_fingerprint(cfg),
                self.chunks,
                embedding_model=cfg.embedding_model,
                document_text=searchable_text,
                retain_extra=True,
            )
        except Exception as exc:
            self.vector_error = f"{type(exc).__name__}: {exc}"

    def _dense(
        self,
        text: str,
        source_collection: str,
        allowed_specifications: tuple[str, ...] = (),
    ) -> list[tuple[str, float]]:
        if self.collection is None:
            raise RuntimeError(self.vector_error or "vector index is unavailable")
        return dense_search(
            self.collection,
            self.cfg,
            text,
            source_collection,
            allowed_specifications,
            getattr(self, "active_document_hashes", ()),
        )

    def _collection_hits(
        self,
        retrieval_id: str,
        text: str,
        source_collection: str,
        allowed_specifications: tuple[str, ...] = (),
        *,
        semantic_text: str | None = None,
        lexical_text: str | None = None,
    ) -> tuple[list[RetrievalHit], list[str]]:
        errors: list[str] = []
        try:
            semantic = self._dense(
                semantic_text or text,
                source_collection,
                allowed_specifications,
            )
        except Exception as exc:
            semantic = []
            errors.append(f"semantic: {type(exc).__name__}: {exc}")
        try:
            lexical = self.bm25.search(
                lexical_text or text,
                source_collection=source_collection,
                limit=self.cfg.lexical_candidates_per_collection,
                allowed_specifications=(
                    frozenset(allowed_specifications)
                    if allowed_specifications
                    else None
                ),
            )
        except Exception as exc:
            lexical = []
            errors.append(f"lexical: {type(exc).__name__}: {exc}")

        ranking = reciprocal_rank_fusion(
            semantic,
            lexical,
            rrf_k=self.cfg.rrf_k,
            limit=self.cfg.final_hits_per_collection,
        )
        hits: list[RetrievalHit] = []
        for rank, chunk_id in enumerate(ranking.ordered_chunk_ids, 1):
            chunk = self.chunk_by_id[chunk_id]
            hits.append(RetrievalHit(
                hit_id=f"{retrieval_id}:{source_collection}:{rank}",
                chunk_id=chunk.chunk_id,
                collection_rank=rank,
                source_collection=chunk.source_collection,
                specification=chunk.specification,
                version=chunk.version,
                source_path=chunk.source_path,
                document_sha256=chunk.document_sha256,
                heading_path=chunk.heading_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                character_start=chunk.character_start,
                character_end=chunk.character_end,
                excerpt=chunk.text,
                semantic_rank=ranking.semantic_rank.get(chunk_id),
                semantic_distance=ranking.semantic_distance.get(chunk_id),
                lexical_rank=ranking.lexical_rank.get(chunk_id),
                lexical_score=ranking.lexical_score.get(chunk_id),
                rrf_score=ranking.rrf_score[chunk_id],
            ))
        return hits, errors

    def keyword_candidates(
        self,
        text: str,
        *,
        allowed_specifications: tuple[str, ...] = (),
        limit: int,
    ) -> list[tuple[str, float]]:
        return self.bm25.search(
            text,
            source_collection="3gpp_markdown",
            limit=limit,
            allowed_specifications=(
                frozenset(allowed_specifications)
                if allowed_specifications
                else None
            ),
        )

    def semantic_candidates(
        self,
        text: str,
        *,
        allowed_specifications: tuple[str, ...] = (),
        limit: int,
    ) -> list[tuple[str, float]]:
        return self._dense(
            text,
            "3gpp_markdown",
            allowed_specifications,
        )[:limit]

    def hybrid_hits(
        self,
        request_id: str,
        text: str,
        *,
        semantic_query: str | None = None,
        allowed_specifications: tuple[str, ...] = (),
        limit: int,
    ) -> tuple[tuple[RetrievalHit, ...], tuple[str, ...]]:
        hits, errors = self._collection_hits(
            request_id,
            text,
            "3gpp_markdown",
            allowed_specifications,
            semantic_text=semantic_query,
            lexical_text=text,
        )
        return tuple(hits[:limit]), tuple(errors)

    def retrieve(self, item: Any) -> RetrievalTaskResult:
        hits: list[RetrievalHit] = []
        errors: list[str] = []
        for source_collection in item.source_collections:
            try:
                collection_hits, collection_errors = self._collection_hits(
                    item.retrieval_id,
                    item.retrieval_text,
                    source_collection,
                )
                hits.extend(collection_hits)
                errors.extend(
                    f"{source_collection}: {error}"
                    for error in collection_errors
                )
            except Exception as exc:
                errors.append(f"{source_collection}: {type(exc).__name__}: {exc}")
        if hits and errors:
            status = "partial"
        elif hits:
            status = "completed"
        elif errors:
            status = "unavailable"
        else:
            status = "no_hits"
        return RetrievalTaskResult(
            retrieval_id=item.retrieval_id,
            query_id=item.query_id,
            status=status,
            hits=tuple(hits),
            errors=tuple(errors),
        )


def effective_corpus_scope(
    cfg: RetrievalCfg,
    retriever: object | None = None,
) -> CorpusScope:
    """Return the scope actually used by a retriever, or the configured scope."""

    scope = getattr(retriever, "corpus_scope", None)
    if scope is None:
        return corpus_scope_from_config(cfg)
    if not isinstance(scope, CorpusScope):
        raise TypeError("retriever.corpus_scope must be a CorpusScope")
    return scope


def execute_retrieval_plan(
    plan: Any,
    *,
    app_config: AppConfig | None = None,
    corpus_scope: CorpusScope | None = None,
) -> RetrievalExecution:
    """Execute all Retrieval Plan items and return a serializable audit artifact."""

    cfg = (app_config or load_config()).retrieval
    active_scope = corpus_scope or corpus_scope_from_config(cfg)
    try:
        retriever = HybridMarkdownRetriever(cfg, corpus_scope=active_scope)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return RetrievalExecution(
            status="unavailable",
            embedding_model=cfg.embedding_model,
            corpus_version_policy=active_scope.version_policy,
            corpus_release=active_scope.release,
            corpus_versions=active_scope.versions,
            tasks=tuple(
                RetrievalTaskResult(
                    retrieval_id=item.retrieval_id,
                    query_id=item.query_id,
                    status="unavailable",
                    errors=(error,),
                )
                for item in plan.retrievals
            ),
            errors=(error,),
        )

    tasks = tuple(retriever.retrieve(item) for item in plan.retrievals)
    status: Literal["completed", "partial", "unavailable"]
    if tasks and all(item.status == "unavailable" for item in tasks):
        status = "unavailable"
    elif any(item.status in {"partial", "unavailable"} for item in tasks):
        status = "partial"
    else:
        status = "completed"
    return RetrievalExecution(
        status=status,
        index_fingerprint=retriever.fingerprint,
        index_manifest_path=retriever.index_manifest_path,
        embedding_model=cfg.embedding_model,
        corpus_version_policy=retriever.corpus_scope.version_policy,
        corpus_release=retriever.corpus_scope.release,
        corpus_versions=retriever.corpus_scope.versions,
        tasks=tasks,
        errors=tuple(error for task in tasks for error in task.errors),
    )


__all__ = [
    "CorpusVersionPolicy",
    "CorpusScope",
    "HybridMarkdownRetriever",
    "MarkdownChunk",
    "RetrievalExecution",
    "RetrievalHit",
    "RetrievalIndexManifest",
    "RetrievalTaskResult",
    "SourceDocument",
    "SpecificationVersion",
    "chunk_source_documents",
    "discover_source_documents",
    "evidence_body_text",
    "effective_corpus_scope",
    "execute_retrieval_plan",
]

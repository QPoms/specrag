"""Thin tool adapters over the active specification retrieval implementation."""

from __future__ import annotations

from specrag.models import NativeFunctionDefinition
from specrag.retrieval.hybrid import HybridMarkdownRetriever
from specrag.retrieval.index import searchable_text
from specrag.retrieval.navigation import expand_source_context_chunks
from specrag.retrieval.reranker import CrossEncoderReranker

from .models import (
    ClauseLookupArguments,
    ClauseNavigationArguments,
    CrossReferenceLookupArguments,
    ExpandSourceContextArguments,
    HybridSearchArguments,
    HyDESearchArguments,
    KeywordSearchArguments,
    RerankArguments,
    RetrievalToolName,
    RetrievalToolRequest,
    RetrievalToolResult,
    SemanticSearchArguments,
    ToolCapability,
)
from .registry import ToolRegistry
from .retrieval_base import RetrieverTool


class KeywordSearchTool(RetrieverTool):
    name: RetrievalToolName = "keyword_search"
    argument_type = KeywordSearchArguments
    capability = ToolCapability(
        name=name,
        description=(
            "BM25 search for short exact protocol identifiers, message names, "
            "information elements, acronyms, or distinctive source wording; "
            "supply a concise lexical_query and avoid long concatenations of "
            "conceptual terms."
        ),
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


class SemanticSearchTool(RetrieverTool):
    name: RetrievalToolName = "semantic_search"
    argument_type = SemanticSearchArguments
    capability = ToolCapability(
        name=name,
        description=(
            "Embedding search for behavior, conditions, responsibilities, or "
            "relationships whose source wording may differ from the question; "
            "describe the required behavior in semantic_query."
        ),
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


class HybridSearchTool(RetrieverTool):
    name: RetrievalToolName = "hybrid_search"
    argument_type = HybridSearchArguments
    capability = ToolCapability(
        name=name,
        description=(
            "BM25 and embedding retrieval fused with reciprocal rank fusion; "
            "use when exact protocol anchors and conceptual meaning both matter. "
            "Supply concise exact anchors in lexical_query and the required "
            "behavior or relation in semantic_query."
        ),
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


class HyDESearchTool(RetrieverTool):
    name: RetrievalToolName = "hyde_search"
    argument_type = HyDESearchArguments
    capability = ToolCapability(
        name=name,
        description=(
            "Fuse BM25 retrieval over concise exact protocol anchors with "
            "embedding retrieval over an untrusted model-generated hypothetical "
            "normative passage. Do not invent clause numbers, citations, or "
            "unsupported protocol names; the passage is never evidence."
        ),
        execution_kind="model_scored",
    )

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        hypothetical_document = request.hypothetical_document or ""
        if not request.query.strip():
            return self._reject(
                request,
                "hyde_search requires a lexical query",
            )
        if not hypothetical_document.strip():
            return self._reject(
                request,
                "hyde_search requires a hypothetical document",
            )
        try:
            hits, errors = self.retriever.hybrid_hits(
                request.request_id,
                request.query,
                semantic_query=hypothetical_document,
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
                self._hit_evidence(hit, request=request, rank=rank)
                for rank, hit in enumerate(hits, 1)
            ),
            errors=errors,
        )


class ClauseLookupTool(RetrieverTool):
    name: RetrievalToolName = "clause_lookup"
    argument_type = ClauseLookupArguments
    capability = ToolCapability(
        name=name,
        description=(
            "Lookup by a clause identifier, exact title, or combined heading. "
            "A model-proposed value is an untrusted retrieval hypothesis; only "
            "source content returned by the tool can become evidence. Do not "
            "manufacture neighboring clauses by incrementing an observed value."
        ),
        execution_kind="deterministic",
    )

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        if request.clause is None:
            return self._reject(request, "clause_lookup requires a clause")
        try:
            records = self.retriever.clause_index.lookup(
                request.clause,
                specifications=request.target_specifications,
                limit=12,
            )
        except ValueError as exc:
            return self._reject(request, str(exc))
        all_chunks = tuple(dict.fromkeys(
            chunk
            for record in records
            for chunk in self.retriever.clause_index.chunks_for(
                record,
                include_descendants=request.include_descendants,
            )
        ))
        page = all_chunks[request.cursor:request.cursor + request.limit]
        next_cursor = (
            request.cursor + len(page)
            if request.cursor + len(page) < len(all_chunks)
            else None
        )
        return self._result(
            request,
            (
                self._chunk_evidence(chunk, request=request, rank=rank)
                for rank, chunk in enumerate(page, 1)
            ),
            next_cursor=next_cursor,
        )


class ClauseNavigationTool(RetrieverTool):
    name: RetrievalToolName = "navigate_clause"
    argument_type = ClauseNavigationArguments
    capability = ToolCapability(
        name=name,
        description=(
            "Follow a parent, direct-child, or parsed cross-reference edge from "
            "one to four already observed evidence items. Navigation is limited "
            "to clauses present in the active corpus snapshot."
        ),
        execution_kind="deterministic",
    )

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        if request.clause_relation is None:
            return self._reject(request, "navigate_clause requires a relation")
        unknown = tuple(
            item
            for item in request.seed_chunk_ids
            if item not in self.retriever.chunk_by_id
        )
        if unknown:
            return self._reject(
                request, "unknown seed chunks: " + ", ".join(unknown)
            )
        records = []
        for chunk_id in request.seed_chunk_ids:
            owner = self.retriever.clause_index.owner(chunk_id)
            if owner is None:
                continue
            records.extend(
                self.retriever.clause_index.related(owner, request.clause_relation)
            )
        unique_records = tuple(dict.fromkeys(records))
        chunks = tuple(dict.fromkeys(
            chunk
            for record in unique_records
            for chunk in self.retriever.clause_index.chunks_for(record)
        ))
        page = chunks[request.cursor:request.cursor + request.limit]
        next_cursor = (
            request.cursor + len(page)
            if request.cursor + len(page) < len(chunks)
            else None
        )
        return self._result(
            request,
            (
                self._chunk_evidence(chunk, request=request, rank=rank)
                for rank, chunk in enumerate(page, 1)
            ),
            next_cursor=next_cursor,
        )


class CrossReferenceLookupTool(RetrieverTool):
    name: RetrievalToolName = "cross_reference_lookup"
    argument_type = CrossReferenceLookupArguments
    capability = ToolCapability(
        name=name,
        description=(
            "Read a clause explicitly referenced by previously observed "
            "evidence. Supply the cited target specification (or null for the "
            "same specification), target clause, cited version when present, "
            "and a page cursor. The tool "
            "rejects unobserved, ambiguous, absent, or out-of-scope targets."
        ),
        execution_kind="deterministic",
    )

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        if request.reference_clause is None:
            return self._reject(
                request, "cross_reference_lookup requires a target clause"
            )
        unknown = tuple(
            item
            for item in request.seed_chunk_ids
            if item not in self.retriever.chunk_by_id
        )
        if unknown:
            return self._reject(
                request, "unknown seed chunks: " + ", ".join(unknown)
            )
        targets = []
        for chunk_id in request.seed_chunk_ids:
            owner = self.retriever.clause_index.owner(chunk_id)
            if owner is None:
                continue
            target = self.retriever.clause_index.resolve_reference(
                owner,
                target_specification=request.reference_specification,
                target_clause_id=request.reference_clause,
                target_version=request.reference_version,
            )
            if target is not None:
                targets.append(target)
        unique_targets = tuple(dict.fromkeys(targets))
        if len(unique_targets) != 1:
            return self._reject(
                request,
                "the observed source does not identify one in-scope target clause",
            )
        chunks = self.retriever.clause_index.chunks_for(unique_targets[0])
        page = chunks[request.cursor:request.cursor + request.limit]
        next_cursor = (
            request.cursor + len(page)
            if request.cursor + len(page) < len(chunks)
            else None
        )
        return self._result(
            request,
            (
                self._chunk_evidence(chunk, request=request, rank=rank)
                for rank, chunk in enumerate(page, 1)
            ),
            next_cursor=next_cursor,
        )


class ExpandSourceContextTool(RetrieverTool):
    name: RetrievalToolName = "expand_source_context"
    argument_type = ExpandSourceContextArguments
    capability = ToolCapability(
        name=name,
        description=(
            "Bounded source-order expansion around trusted seed chunks, "
            "including the current subsection and its immediately adjacent "
            "sibling subsections within the same parent section. Supply one to "
            "four already observed seed_evidence_ids."
        ),
        execution_kind="deterministic",
    )

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        try:
            chunks = expand_source_context_chunks(
                self.retriever.chunks,
                request.seed_chunk_ids,
                limit=request.limit,
                clause_index=self.retriever.clause_index,
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


class RerankTool(RetrieverTool):
    name: RetrievalToolName = "rerank"
    argument_type = RerankArguments
    capability = ToolCapability(
        name=name,
        description=(
            "Cross-encoder reranking of already observed trusted candidate "
            "chunks using semantic_query and one to four seed_evidence_ids; "
            "this does not discover new source candidates."
        ),
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


def retrieval_tool_capabilities() -> tuple[ToolCapability, ...]:
    """Describe the executable retrieval tools without constructing indexes."""

    capabilities = (
        KeywordSearchTool.capability,
        SemanticSearchTool.capability,
        HybridSearchTool.capability,
        HyDESearchTool.capability,
        RerankTool.capability,
        ClauseLookupTool.capability,
        ClauseNavigationTool.capability,
        CrossReferenceLookupTool.capability,
        ExpandSourceContextTool.capability,
    )
    return tuple(sorted(capabilities, key=lambda item: item.name))


def retrieval_tool_definitions() -> tuple[NativeFunctionDefinition, ...]:
    """Describe strict native tools without constructing retrieval indexes."""

    tool_types = (
        KeywordSearchTool,
        SemanticSearchTool,
        HybridSearchTool,
        HyDESearchTool,
        RerankTool,
        ClauseLookupTool,
        ClauseNavigationTool,
        CrossReferenceLookupTool,
        ExpandSourceContextTool,
    )
    return tuple(sorted(
        (
            NativeFunctionDefinition(
                name=tool_type.name,
                description=tool_type.capability.description,
                parameters=tool_type.argument_type.model_json_schema(),
            )
            for tool_type in tool_types
        ),
        key=lambda item: item.name,
    ))


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
        HyDESearchTool(retriever),
        RerankTool(retriever, reranker),
        ClauseLookupTool(retriever),
        ClauseNavigationTool(retriever),
        CrossReferenceLookupTool(retriever),
        ExpandSourceContextTool(retriever),
    ):
        registry.register(tool)
    return registry


__all__ = [
    "ClauseLookupTool",
    "ClauseNavigationTool",
    "CrossReferenceLookupTool",
    "ExpandSourceContextTool",
    "HybridSearchTool",
    "HyDESearchTool",
    "KeywordSearchTool",
    "RerankTool",
    "SemanticSearchTool",
    "build_retrieval_tool_registry",
    "retrieval_tool_capabilities",
    "retrieval_tool_definitions",
]

"""Stable request and evidence contracts for agent-callable retrieval tools."""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from specrag.retrieval.models import CorpusScope


RetrievalToolName = Literal[
    "keyword_search",
    "semantic_search",
    "hybrid_search",
    "hyde_search",
    "rerank",
    "clause_lookup",
    "navigate_clause",
    "cross_reference_lookup",
    "expand_source_context",
]
ToolStatus = Literal["completed", "partial", "no_hits", "unavailable", "rejected"]
ScoreKind = Literal[
    "bm25",
    "semantic_distance",
    "rrf",
    "cross_encoder",
    "source_order",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCapability(_StrictModel):
    name: RetrievalToolName
    description: str = Field(min_length=1)
    execution_kind: Literal["deterministic", "model_scored"]
    source_constraint: Literal["active_3gpp_corpus_scope"] = (
        "active_3gpp_corpus_scope"
    )


class _ToolArguments(_StrictModel):
    """Common strict arguments exposed to the model for retrieval calls."""

    query_id: str = Field(min_length=1)
    max_results: int = Field(ge=1, le=12)


class _ScopedToolArguments(_ToolArguments):
    target_specifications: tuple[str, ...] = Field(max_length=6)

    @model_validator(mode="after")
    def _valid_specifications(self) -> "_ScopedToolArguments":
        if len(self.target_specifications) != len(set(self.target_specifications)):
            raise ValueError("target specifications must be unique")
        invalid = [
            value
            for value in self.target_specifications
            if re.fullmatch(r"\d{2}\.\d{3}", value) is None
        ]
        if invalid:
            raise ValueError(
                "invalid target specifications: " + ", ".join(invalid)
            )
        return self


class KeywordSearchArguments(_ScopedToolArguments):
    lexical_query: str = Field(min_length=1, max_length=4000)


class SemanticSearchArguments(_ScopedToolArguments):
    semantic_query: str = Field(min_length=1, max_length=4000)


class HybridSearchArguments(_ScopedToolArguments):
    lexical_query: str = Field(min_length=1, max_length=4000)
    semantic_query: str = Field(min_length=1, max_length=4000)


class HyDESearchArguments(_ScopedToolArguments):
    lexical_query: str = Field(min_length=1, max_length=4000)
    hypothetical_document: str = Field(min_length=1, max_length=4000)


class RerankArguments(_ToolArguments):
    semantic_query: str = Field(min_length=1, max_length=4000)
    seed_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=4)


class ClauseLookupArguments(_ScopedToolArguments):
    clause: str = Field(min_length=1, max_length=64)
    include_descendants: bool
    cursor: int = Field(ge=0)


class ClauseNavigationArguments(_ToolArguments):
    seed_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    relation: Literal["parent", "children"]
    cursor: int = Field(ge=0)


class CrossReferenceLookupArguments(_ToolArguments):
    seed_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    target_specification: str | None = Field(
        pattern=r"^\d{2}\.\d{3}$"
    )
    target_version: str | None = Field(pattern=r"^\d+\.\d+\.\d+$")
    target_clause: str = Field(min_length=1, max_length=64)
    cursor: int = Field(ge=0)


class ExpandSourceContextArguments(_ToolArguments):
    seed_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=4)


RetrievalToolArguments: TypeAlias = (
    KeywordSearchArguments
    | SemanticSearchArguments
    | HybridSearchArguments
    | HyDESearchArguments
    | RerankArguments
    | ClauseLookupArguments
    | ClauseNavigationArguments
    | CrossReferenceLookupArguments
    | ExpandSourceContextArguments
)


RETRIEVAL_TOOL_ARGUMENT_TYPES: dict[
    RetrievalToolName,
    type[RetrievalToolArguments],
] = {
    "keyword_search": KeywordSearchArguments,
    "semantic_search": SemanticSearchArguments,
    "hybrid_search": HybridSearchArguments,
    "hyde_search": HyDESearchArguments,
    "rerank": RerankArguments,
    "clause_lookup": ClauseLookupArguments,
    "navigate_clause": ClauseNavigationArguments,
    "cross_reference_lookup": CrossReferenceLookupArguments,
    "expand_source_context": ExpandSourceContextArguments,
}


def parse_retrieval_tool_arguments(
    name: str,
    arguments: dict[str, Any],
) -> RetrievalToolArguments:
    try:
        argument_type = RETRIEVAL_TOOL_ARGUMENT_TYPES[name]  # type: ignore[index]
    except KeyError as exc:
        raise KeyError(f"unknown retrieval tool: {name}") from exc
    return argument_type.model_validate(arguments)


class RetrievalToolRequest(_StrictModel):
    request_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    query: str = Field(default="", max_length=4000)
    semantic_query: str | None = Field(default=None, max_length=4000)
    hypothetical_document: str | None = Field(default=None, max_length=4000)
    target_specifications: tuple[str, ...] = Field(default=(), max_length=12)
    clause: str | None = Field(default=None, max_length=64)
    clause_relation: Literal["parent", "children"] | None = None
    reference_specification: str | None = Field(
        default=None, pattern=r"^\d{2}\.\d{3}$"
    )
    reference_version: str | None = Field(
        default=None, pattern=r"^\d+\.\d+\.\d+$"
    )
    reference_clause: str | None = Field(default=None, max_length=64)
    include_descendants: bool = False
    cursor: int = Field(default=0, ge=0)
    seed_chunk_ids: tuple[str, ...] = Field(default=(), max_length=50)
    limit: int = Field(default=8, ge=1, le=50)

    @model_validator(mode="after")
    def _unique_inputs(self) -> "RetrievalToolRequest":
        if len(self.target_specifications) != len(set(self.target_specifications)):
            raise ValueError("target specifications must be unique")
        invalid = [
            value
            for value in self.target_specifications
            if re.fullmatch(r"\d{2}\.\d{3}", value) is None
        ]
        if invalid:
            raise ValueError(
                "invalid target specifications: " + ", ".join(invalid)
            )
        if len(self.seed_chunk_ids) != len(set(self.seed_chunk_ids)):
            raise ValueError("seed chunk IDs must be unique")
        return self


class ToolEvidence(_StrictModel):
    evidence_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    match_kind: RetrievalToolName
    score: float | None = None
    score_kind: ScoreKind
    source_collection: Literal["3gpp_markdown"] = "3gpp_markdown"
    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    release: int = Field(ge=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    source_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")
    heading_path: tuple[str, ...] = ()
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    excerpt: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_locator(self) -> "ToolEvidence":
        if self.line_end < self.line_start:
            raise ValueError("evidence line_end precedes line_start")
        if int(self.version.split(".", 1)[0]) != self.release:
            raise ValueError("evidence Release does not match its exact version")
        return self


class RetrievalToolResult(_StrictModel):
    request_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    tool_name: RetrievalToolName
    status: ToolStatus
    corpus_scope: CorpusScope
    evidence: tuple[ToolEvidence, ...] = ()
    errors: tuple[str, ...] = ()
    next_cursor: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _coherent_status(self) -> "RetrievalToolResult":
        if self.status == "completed" and not self.evidence:
            raise ValueError("completed tool result requires evidence")
        if self.status == "partial" and (not self.evidence or not self.errors):
            raise ValueError("partial tool result requires evidence and errors")
        if self.status == "no_hits" and (self.evidence or self.errors):
            raise ValueError("no_hits tool result cannot contain evidence or errors")
        if self.status in {"unavailable", "rejected"} and (
            self.evidence or not self.errors
        ):
            raise ValueError(
                f"{self.status} tool result requires errors and no evidence"
            )
        return self


class RetrievalTool(Protocol):
    name: RetrievalToolName
    capability: ToolCapability
    argument_type: ClassVar[type[RetrievalToolArguments]]

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult: ...


__all__ = [
    "ClauseLookupArguments",
    "ClauseNavigationArguments",
    "CrossReferenceLookupArguments",
    "ExpandSourceContextArguments",
    "HybridSearchArguments",
    "HyDESearchArguments",
    "KeywordSearchArguments",
    "RETRIEVAL_TOOL_ARGUMENT_TYPES",
    "RetrievalTool",
    "RetrievalToolArguments",
    "parse_retrieval_tool_arguments",
    "RetrievalToolName",
    "RetrievalToolRequest",
    "RetrievalToolResult",
    "ScoreKind",
    "SemanticSearchArguments",
    "RerankArguments",
    "ToolEvidence",
    "ToolCapability",
    "ToolStatus",
]

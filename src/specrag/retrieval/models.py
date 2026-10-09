"""Immutable contracts emitted by the specification retrieval subsystem."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


CorpusVersionPolicy = Literal[
    "latest_per_spec",
    "highest_version_within_release",
]


def validate_corpus_scope(
    version_policy: CorpusVersionPolicy,
    release: int | None,
) -> None:
    if version_policy == "highest_version_within_release" and release is None:
        raise ValueError(
            "corpus_release is required for highest_version_within_release"
        )
    if version_policy == "latest_per_spec" and release is not None:
        raise ValueError(
            "corpus_release may only be set for highest_version_within_release"
        )


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceDocument(_StrictModel):
    source_collection: Literal["3gpp_markdown"]
    specification: str | None = Field(default=None, pattern=r"^\d{2}\.\d{3}$")
    version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    source_path: str
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")

    @model_validator(mode="after")
    def _coherent_source_identity(self):
        if self.specification is None or self.version is None:
            raise ValueError("3GPP source requires specification and version")
        return self


class MarkdownChunk(_StrictModel):
    chunk_id: str = Field(min_length=1)
    source_collection: Literal["3gpp_markdown"]
    specification: str | None = Field(default=None, pattern=r"^\d{2}\.\d{3}$")
    version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    source_path: str
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")
    heading_path: tuple[str, ...] = ()
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    character_start: int | None = Field(default=None, ge=0)
    character_end: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_locator(self):
        if self.line_end < self.line_start:
            raise ValueError("chunk line_end precedes line_start")
        if (self.character_start is None) != (self.character_end is None):
            raise ValueError("chunk character bounds must be supplied together")
        if self.character_start is not None and (
            self.line_start != self.line_end
            or (self.character_end or 0) <= self.character_start
        ):
            raise ValueError("chunk character bounds must locate one non-empty line")
        return self


class RetrievalHit(_StrictModel):
    hit_id: str
    chunk_id: str
    collection_rank: int = Field(ge=1)
    source_collection: Literal["3gpp_markdown"]
    specification: str | None = Field(default=None, pattern=r"^\d{2}\.\d{3}$")
    version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    source_path: str
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")
    heading_path: tuple[str, ...] = ()
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    character_start: int | None = Field(default=None, ge=0)
    character_end: int | None = Field(default=None, ge=1)
    excerpt: str = Field(min_length=1)
    semantic_rank: int | None = Field(default=None, ge=1)
    semantic_distance: float | None = None
    lexical_rank: int | None = Field(default=None, ge=1)
    lexical_score: float | None = None
    rrf_score: float = Field(ge=0)

    @model_validator(mode="after")
    def _valid_locator(self):
        if self.line_end < self.line_start:
            raise ValueError("hit line_end precedes line_start")
        if (self.character_start is None) != (self.character_end is None):
            raise ValueError("hit character bounds must be supplied together")
        if self.character_start is not None and (
            self.line_start != self.line_end
            or (self.character_end or 0) <= self.character_start
        ):
            raise ValueError("hit character bounds must locate one non-empty line")
        return self


class RetrievalTaskResult(_StrictModel):
    retrieval_id: str
    query_id: str
    status: Literal["completed", "partial", "no_hits", "unavailable"]
    hits: tuple[RetrievalHit, ...] = ()
    errors: tuple[str, ...] = ()


class RetrievalExecution(_StrictModel):
    schema_version: Literal["retrieval-execution/v1.0"] = (
        "retrieval-execution/v1.0"
    )
    status: Literal["completed", "partial", "unavailable"]
    index_fingerprint: str | None = None
    index_manifest_path: str | None = None
    embedding_model: str
    corpus_version_policy: CorpusVersionPolicy
    corpus_release: int | None = Field(default=None, ge=1)
    tasks: tuple[RetrievalTaskResult, ...] = ()
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _coherent_corpus_scope(self):
        validate_corpus_scope(self.corpus_version_policy, self.corpus_release)
        return self


class RetrievalIndexManifest(_StrictModel):
    schema_version: Literal["specification-retrieval-index/v1.0"] = (
        "specification-retrieval-index/v1.0"
    )
    index_fingerprint: str
    embedding_model: str
    corpus_version_policy: CorpusVersionPolicy
    corpus_release: int | None = Field(default=None, ge=1)
    chunk_characters: int
    chunk_overlap_characters: int
    collection_name: str
    documents: tuple[SourceDocument, ...]
    chunk_count: int

    @model_validator(mode="after")
    def _coherent_corpus_scope(self):
        validate_corpus_scope(self.corpus_version_policy, self.corpus_release)
        return self


__all__ = [
    "CorpusVersionPolicy",
    "MarkdownChunk",
    "RetrievalExecution",
    "RetrievalHit",
    "RetrievalIndexManifest",
    "RetrievalTaskResult",
    "SourceDocument",
]

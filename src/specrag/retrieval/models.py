"""Immutable contracts emitted by the specification retrieval subsystem."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


CorpusVersionPolicy = Literal[
    "latest_per_spec",
    "highest_version_within_release",
    "exact_version",
    "mixed_versions",
]


def release_from_version(version: str) -> int:
    return int(version.split(".", 1)[0])


def validate_corpus_scope(
    version_policy: CorpusVersionPolicy,
    release: int | None,
    versions: tuple["SpecificationVersion", ...] = (),
) -> None:
    if version_policy == "highest_version_within_release" and release is None:
        raise ValueError(
            "corpus_release is required for highest_version_within_release"
        )
    if version_policy != "highest_version_within_release" and release is not None:
        raise ValueError("corpus_release is only valid for a Release scope")
    if version_policy in {"latest_per_spec", "highest_version_within_release"}:
        if versions:
            raise ValueError("explicit versions are not valid for this corpus scope")
    elif version_policy == "exact_version" and len(versions) != 1:
        raise ValueError("exact_version requires exactly one specification version")
    elif version_policy == "mixed_versions" and len(versions) < 2:
        raise ValueError("mixed_versions requires at least two specification versions")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SpecificationVersion(_StrictModel):
    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    release: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _coherent_release(self) -> "SpecificationVersion":
        expected = release_from_version(self.version)
        if self.release is not None and self.release != expected:
            raise ValueError("Release does not match the exact version")
        object.__setattr__(self, "release", expected)
        return self


class CorpusScope(_StrictModel):
    """One auditable corpus snapshot requested by a retrieval run."""

    version_policy: CorpusVersionPolicy = "highest_version_within_release"
    release: int | None = Field(default=None, ge=1)
    versions: tuple[SpecificationVersion, ...] = ()

    @model_validator(mode="after")
    def _coherent_scope(self) -> "CorpusScope":
        ordered = tuple(sorted(
            self.versions,
            key=lambda item: (item.specification, item.version),
        ))
        if len({item.specification for item in ordered}) != len(ordered):
            raise ValueError("a corpus scope cannot select a specification twice")
        object.__setattr__(self, "versions", ordered)
        validate_corpus_scope(self.version_policy, self.release, ordered)
        return self


class SourceDocument(_StrictModel):
    source_collection: Literal["3gpp_markdown"]
    specification: str | None = Field(default=None, pattern=r"^\d{2}\.\d{3}$")
    version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    release: int | None = Field(default=None, ge=1)
    source_path: str
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")

    @model_validator(mode="after")
    def _coherent_source_identity(self):
        if self.specification is None or self.version is None:
            raise ValueError("3GPP source requires specification and version")
        expected = release_from_version(self.version)
        if self.release is not None and self.release != expected:
            raise ValueError("source Release does not match its exact version")
        object.__setattr__(self, "release", expected)
        return self


class MarkdownChunk(_StrictModel):
    chunk_id: str = Field(min_length=1)
    source_collection: Literal["3gpp_markdown"]
    specification: str | None = Field(default=None, pattern=r"^\d{2}\.\d{3}$")
    version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    release: int | None = Field(default=None, ge=1)
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
        if self.specification is None or self.version is None:
            raise ValueError("3GPP chunk requires specification and version")
        expected = release_from_version(self.version)
        if self.release is not None and self.release != expected:
            raise ValueError("chunk Release does not match its exact version")
        object.__setattr__(self, "release", expected)
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
    release: int | None = Field(default=None, ge=1)
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
        if self.specification is None or self.version is None:
            raise ValueError("3GPP hit requires specification and version")
        expected = release_from_version(self.version)
        if self.release is not None and self.release != expected:
            raise ValueError("hit Release does not match its exact version")
        object.__setattr__(self, "release", expected)
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
    schema_version: Literal["retrieval-execution/v1.1"] = (
        "retrieval-execution/v1.1"
    )
    status: Literal["completed", "partial", "unavailable"]
    index_fingerprint: str | None = None
    index_manifest_path: str | None = None
    embedding_model: str
    corpus_version_policy: CorpusVersionPolicy
    corpus_release: int | None = Field(default=None, ge=1)
    corpus_versions: tuple[SpecificationVersion, ...] = ()
    tasks: tuple[RetrievalTaskResult, ...] = ()
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _coherent_corpus_scope(self):
        validate_corpus_scope(
            self.corpus_version_policy,
            self.corpus_release,
            self.corpus_versions,
        )
        return self


class RetrievalIndexManifest(_StrictModel):
    schema_version: Literal["specification-retrieval-index/v1.1"] = (
        "specification-retrieval-index/v1.1"
    )
    index_fingerprint: str
    embedding_model: str
    corpus_version_policy: CorpusVersionPolicy
    corpus_release: int | None = Field(default=None, ge=1)
    corpus_versions: tuple[SpecificationVersion, ...] = ()
    chunk_characters: int
    chunk_overlap_characters: int
    collection_name: str
    documents: tuple[SourceDocument, ...]
    chunk_count: int

    @model_validator(mode="after")
    def _coherent_corpus_scope(self):
        validate_corpus_scope(
            self.corpus_version_policy,
            self.corpus_release,
            self.corpus_versions,
        )
        return self


__all__ = [
    "CorpusScope",
    "CorpusVersionPolicy",
    "MarkdownChunk",
    "RetrievalExecution",
    "RetrievalHit",
    "RetrievalIndexManifest",
    "RetrievalTaskResult",
    "SourceDocument",
    "SpecificationVersion",
    "release_from_version",
    "validate_corpus_scope",
]

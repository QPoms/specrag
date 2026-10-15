"""Framework-independent contracts for versioned corpus maintenance."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


CorpusVersionStatus = Literal[
    "discovered", "indexing", "ready", "active", "draining", "retired", "failed"
]


class CorpusSourceDocument(_StrictModel):
    """Materialized public-corpus source returned by an ingestion adapter."""

    source_collection: Literal["3gpp_markdown"]
    specification: str | None = Field(default=None, pattern=r"^\d{2}\.\d{3}$")
    version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    release: int | None = Field(default=None, ge=1)
    source_path: str
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")

    @model_validator(mode="after")
    def _coherent_source_identity(self) -> "CorpusSourceDocument":
        if self.specification is None or self.version is None:
            raise ValueError("3GPP source requires specification and version")
        expected = int(self.version.split(".", 1)[0])
        if self.release is not None and self.release != expected:
            raise ValueError("source Release does not match its exact version")
        object.__setattr__(self, "release", expected)
        return self


class CorpusVersionRef(_StrictModel):
    """A stable exact specification version used by a Run snapshot."""

    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    release: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _coherent_release(self) -> "CorpusVersionRef":
        expected = int(self.version.split(".", 1)[0])
        if self.release is not None and self.release != expected:
            raise ValueError("Release does not match the exact version")
        object.__setattr__(self, "release", expected)
        return self


class CorpusVersionRecord(_StrictModel):
    """Persisted discovery/build state for one immutable source document."""

    version_id: str = Field(min_length=1)
    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    release: int = Field(ge=1)
    source_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")
    status: CorpusVersionStatus = "discovered"
    discovered_at: datetime
    indexed_at: datetime | None = None
    activated_at: datetime | None = None
    index_fingerprint: str | None = None
    chunk_count: int | None = Field(default=None, ge=1)
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _coherent_identity(self) -> "CorpusVersionRecord":
        expected = int(self.version.split(".", 1)[0])
        if self.release != expected:
            raise ValueError("record Release does not match exact version")
        if self.status == "ready" and self.indexed_at is None:
            raise ValueError("ready version requires indexed_at")
        if self.status == "active" and self.activated_at is None:
            raise ValueError("active version requires activated_at")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed version requires failure_reason")
        return self


class ActiveCorpusVersion(_StrictModel):
    """The active pointer for one Release + Specification pair."""

    release: int = Field(ge=1)
    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    version_id: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")

    @model_validator(mode="after")
    def _coherent_release(self) -> "ActiveCorpusVersion":
        if self.release != int(self.version.split(".", 1)[0]):
            raise ValueError("active Release does not match exact version")
        return self


class CorpusSyncItem(_StrictModel):
    """One planned or executed document-level sync action."""

    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    source_path: str = Field(min_length=1)
    action: Literal["unchanged", "new", "changed", "failed", "activated"]
    status: CorpusVersionStatus
    message: str | None = None


class CorpusBuildResult(_StrictModel):
    """Validated artifacts produced for exactly one source document."""

    index_fingerprint: str = Field(min_length=1)
    chunk_count: int = Field(ge=1)
    lexical_record_count: int = Field(ge=1)
    vector_record_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _complete_indexes(self) -> "CorpusBuildResult":
        if self.lexical_record_count != self.chunk_count:
            raise ValueError("lexical index does not cover every chunk")
        if self.vector_record_count != self.chunk_count:
            raise ValueError("vector index does not cover every chunk")
        return self


class SpecificationSourceCandidate(_StrictModel):
    """Provider discovery result before download/conversion materializes Markdown."""

    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    release: int = Field(ge=1)
    source_uri: str = Field(min_length=1)
    target_source_path: str = Field(min_length=1)
    source_format: Literal["markdown", "docx", "zip"]
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9A-F]{64}$")

    @model_validator(mode="after")
    def _matching_release(self) -> "SpecificationSourceCandidate":
        if self.release != int(self.version.split(".", 1)[0]):
            raise ValueError("candidate Release does not match exact version")
        return self


__all__ = [
    "ActiveCorpusVersion",
    "CorpusSourceDocument",
    "CorpusBuildResult",
    "CorpusSyncItem",
    "CorpusVersionRecord",
    "CorpusVersionRef",
    "CorpusVersionStatus",
    "SpecificationSourceCandidate",
]

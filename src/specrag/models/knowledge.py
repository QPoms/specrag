"""Framework-independent contracts for validated long-term knowledge."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class KnowledgeSource(_StrictModel):
    evidence_id: str = Field(min_length=1)
    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    release: int = Field(ge=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    source_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")
    heading_path: tuple[str, ...] = ()
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @model_validator(mode="after")
    def _coherent_source(self) -> "KnowledgeSource":
        if self.line_end < self.line_start:
            raise ValueError("knowledge source line range is invalid")
        if int(self.version.split(".", 1)[0]) != self.release:
            raise ValueError("knowledge source Release and version disagree")
        return self


class KnowledgeProposal(_StrictModel):
    claim_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    conditions: tuple[str, ...] = ()
    expires_at: datetime | None = None


class ValidatedKnowledge(_StrictModel):
    schema_version: Literal["validated-knowledge/v1.0"] = (
        "validated-knowledge/v1.0"
    )
    knowledge_id: str = Field(pattern=r"^K-[0-9A-F]{24}$")
    claim_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    conditions: tuple[str, ...] = ()
    sources: tuple[KnowledgeSource, ...] = Field(min_length=1)
    validated_at: datetime
    expires_at: datetime | None = None
    validation_status: Literal["validated"] = "validated"
    memory_status: Literal[
        "active",
        "superseded",
        "conflicted",
        "invalidated",
    ] = "active"
    superseded_by: str | None = None
    invalidation_reason: str | None = None

    @model_validator(mode="after")
    def _coherent_status(self) -> "ValidatedKnowledge":
        if self.expires_at is not None and self.expires_at <= self.validated_at:
            raise ValueError("knowledge expiry must be after validation time")
        if self.memory_status == "superseded" and self.superseded_by is None:
            raise ValueError("superseded knowledge requires superseded_by")
        if self.memory_status != "superseded" and self.superseded_by is not None:
            raise ValueError("only superseded knowledge may set superseded_by")
        if self.memory_status == "invalidated" and not self.invalidation_reason:
            raise ValueError("invalidated knowledge requires a reason")
        if self.memory_status != "invalidated" and self.invalidation_reason:
            raise ValueError("only invalidated knowledge may set a reason")
        return self


class MemorySearchResult(_StrictModel):
    knowledge: ValidatedKnowledge
    score: float = Field(ge=0, le=1)


__all__ = [
    "KnowledgeProposal",
    "KnowledgeSource",
    "MemorySearchResult",
    "ValidatedKnowledge",
]

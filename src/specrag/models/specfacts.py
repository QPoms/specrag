"""Workspace-scoped reviewed specification fact contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SpecFactSource(_StrictModel):
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
    def _coherent_source(self) -> "SpecFactSource":
        if self.line_end < self.line_start:
            raise ValueError("SpecFact source line range is invalid")
        if int(self.version.split(".", 1)[0]) != self.release:
            raise ValueError("SpecFact source Release and version disagree")
        return self


class SpecFactDraft(_StrictModel):
    claim_id: str = Field(min_length=1)
    assertion: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    conditions: tuple[str, ...] = ()
    expires_at: datetime | None = None


class SpecFactProposal(_StrictModel):
    schema_version: Literal["specfact-proposal/v1"] = "specfact-proposal/v1"
    proposal_id: str = Field(pattern=r"^SFP-[0-9A-F]{24}$")
    operation_key: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    run_id: str = Field(pattern=r"^RUN-[0-9A-F]{32}$")
    claim_id: str = Field(min_length=1)
    assertion: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    conditions: tuple[str, ...] = ()
    sources: tuple[SpecFactSource, ...] = Field(min_length=1)
    proposed_by: str = Field(min_length=1)
    proposed_at: datetime
    expires_at: datetime | None = None
    status: Literal["pending", "approved", "rejected"] = "pending"
    reviewer_user_id: str | None = None
    reviewed_at: datetime | None = None
    review_reason: str | None = None

    @model_validator(mode="after")
    def _coherent_review(self) -> "SpecFactProposal":
        if self.expires_at is not None and self.expires_at <= self.proposed_at:
            raise ValueError("SpecFact proposal expiry must follow creation")
        reviewed = self.status in {"approved", "rejected"}
        if reviewed != bool(self.reviewer_user_id and self.reviewed_at):
            raise ValueError("reviewed proposal requires reviewer and review time")
        if self.status == "rejected" and not self.review_reason:
            raise ValueError("rejected proposal requires a reason")
        return self


class SpecFact(_StrictModel):
    schema_version: Literal["specfact/v1"] = "specfact/v1"
    fact_id: str = Field(pattern=r"^SF-[0-9A-F]{24}$")
    proposal_id: str = Field(pattern=r"^SFP-[0-9A-F]{24}$")
    workspace_id: str = Field(min_length=1)
    assertion: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    conditions: tuple[str, ...] = ()
    sources: tuple[SpecFactSource, ...] = Field(min_length=1)
    reviewer_user_id: str = Field(min_length=1)
    approved_at: datetime
    expires_at: datetime | None = None
    status: Literal["approved", "superseded", "conflicted", "invalidated"] = (
        "approved"
    )
    superseded_by: str | None = None
    invalidation_reason: str | None = None

    @model_validator(mode="after")
    def _coherent_status(self) -> "SpecFact":
        if self.expires_at is not None and self.expires_at <= self.approved_at:
            raise ValueError("SpecFact expiry must follow approval")
        if (self.status == "superseded") != (self.superseded_by is not None):
            raise ValueError("superseded SpecFact requires superseded_by only")
        if (self.status == "invalidated") != bool(self.invalidation_reason):
            raise ValueError("invalidated SpecFact requires a reason only")
        return self


class SpecFactSearchResult(_StrictModel):
    fact: SpecFact
    score: float = Field(ge=0, le=1)


__all__ = [
    "SpecFact",
    "SpecFactDraft",
    "SpecFactProposal",
    "SpecFactSearchResult",
    "SpecFactSource",
]

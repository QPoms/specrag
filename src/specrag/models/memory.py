"""Auditable Core and Topic Memory contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MemoryScope = Literal["user", "workspace"]
MemoryLayer = Literal["core", "topic"]
MemoryKind = Literal[
    "default_release",
    "answer_language",
    "detail_level",
    "preference",
    "research_habit",
    "followed_topic",
    "research_lead",
    "project_decision",
    "evaluation_convention",
    "experiment_lesson",
]
MemoryOrigin = Literal["explicit", "extractor", "consolidation", "migrated"]
MemoryStatus = Literal["active", "superseded", "retracted", "archived"]
MemoryTopicStatus = Literal["active", "archived"]
MemoryOperation = Literal[
    "add", "supersede", "retract", "create_topic", "merge", "archive", "restore"
]
MemoryProposalStatus = Literal["pending", "approved", "rejected"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryTopic(_StrictModel):
    schema_version: Literal["memory-topic/v1"] = "memory-topic/v1"
    topic_id: str = Field(pattern=r"^MT-[0-9A-F]{32}$")
    workspace_id: str = Field(min_length=1)
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=100)
    description: str = Field(min_length=1, max_length=1000)
    pinned: bool = False
    status: MemoryTopicStatus = "active"
    created_by: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    operation_key: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def _coherent_dates(self) -> "MemoryTopic":
        if self.updated_at < self.created_at:
            raise ValueError("topic update cannot precede creation")
        return self


class MemoryEntry(_StrictModel):
    schema_version: Literal["memory-entry/v1"] = "memory-entry/v1"
    memory_id: str = Field(pattern=r"^(?:MEM|UM)-[0-9A-F]{32}$")
    scope: MemoryScope
    layer: MemoryLayer
    workspace_id: str = Field(min_length=1)
    owner_user_id: str | None = None
    topic_id: str | None = Field(default=None, pattern=r"^MT-[0-9A-F]{32}$")
    kind: MemoryKind
    key: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=8000)
    created_by: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    status: MemoryStatus = "active"
    origin: MemoryOrigin = "explicit"
    revision: int = Field(default=1, ge=1)
    supersedes_id: str | None = Field(
        default=None, pattern=r"^(?:MEM|UM)-[0-9A-F]{32}$"
    )
    merged_from_ids: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    operation_key: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def _coherent_entry(self) -> "MemoryEntry":
        if self.scope == "user" and not self.owner_user_id:
            raise ValueError("user Memory requires an owner")
        if self.scope == "workspace" and self.owner_user_id is not None:
            raise ValueError("workspace Memory cannot have a user owner")
        if self.layer == "core" and self.topic_id is not None:
            raise ValueError("Core Memory cannot belong to a Topic")
        if self.layer == "topic" and self.topic_id is None:
            raise ValueError("Topic Memory requires a Topic")
        if self.layer == "topic" and self.scope != "workspace":
            raise ValueError("Topic Memory is workspace-scoped")
        if self.kind in {"default_release", "answer_language", "detail_level"}:
            if self.scope != "user" or self.layer != "core":
                raise ValueError("configuration Memory must be User Core")
            if self.key.casefold() != "default":
                raise ValueError(f"{self.kind} must use the default key")
        if self.updated_at < self.created_at:
            raise ValueError("Memory update cannot precede creation")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("Memory expiry must follow creation")
        if self.revision == 1 and self.supersedes_id is not None:
            raise ValueError("first Memory revision cannot supersede another item")
        if self.revision > 1 and self.supersedes_id is None:
            raise ValueError("later Memory revision must name its predecessor")
        if len(self.merged_from_ids) != len(set(self.merged_from_ids)):
            raise ValueError("merged Memory IDs must be unique")
        if len(self.source_refs) != len(set(self.source_refs)):
            raise ValueError("Memory source references must be unique")
        if self.memory_id in self.merged_from_ids:
            raise ValueError("Memory cannot merge itself")
        if self.origin == "consolidation" and len(self.merged_from_ids) < 2:
            raise ValueError("consolidated Memory requires at least two inputs")
        return self


class MemoryMutationDraft(_StrictModel):
    operation: MemoryOperation
    scope: MemoryScope
    kind: MemoryKind | None = None
    key: str | None = Field(default=None, max_length=200)
    content: str | None = Field(default=None, max_length=8000)
    topic_id: str | None = Field(default=None, pattern=r"^MT-[0-9A-F]{32}$")
    target_memory_ids: tuple[str, ...] = ()
    topic_slug: str | None = Field(
        default=None, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=100
    )
    topic_description: str | None = Field(default=None, max_length=1000)
    topic_pinned: bool = False
    source_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _valid_operation_shape(self) -> "MemoryMutationDraft":
        user_kinds = {
            "default_release",
            "answer_language",
            "detail_level",
            "preference",
            "research_habit",
            "followed_topic",
            "research_lead",
        }
        workspace_kinds = {
            "project_decision", "evaluation_convention", "experiment_lesson"
        }
        if self.kind is not None:
            if self.scope == "user" and self.kind not in user_kinds:
                raise ValueError(f"{self.kind} is not valid User Memory")
            if self.scope == "workspace" and self.kind not in workspace_kinds:
                raise ValueError(f"{self.kind} is not valid Project Memory")
        if self.scope == "user" and self.topic_id is not None:
            raise ValueError("User Memory cannot target a workspace Topic")
        if self.operation == "create_topic":
            if self.scope != "workspace" or not self.topic_slug or not self.topic_description:
                raise ValueError("create_topic requires workspace scope, slug, and description")
        elif self.operation == "add":
            if self.kind is None or not self.key or not self.content:
                raise ValueError("add requires kind, key, and content")
        elif self.operation == "supersede":
            if len(self.target_memory_ids) != 1 or not self.content:
                raise ValueError("supersede requires one target and replacement content")
        elif self.operation == "merge":
            if len(self.target_memory_ids) < 2 or not self.content:
                raise ValueError("merge requires at least two targets and merged content")
        elif self.operation in {"retract", "archive", "restore"}:
            if not self.target_memory_ids:
                raise ValueError(f"{self.operation} requires at least one target")
        if len(self.target_memory_ids) != len(set(self.target_memory_ids)):
            raise ValueError("proposal target IDs must be unique")
        if len(self.source_refs) != len(set(self.source_refs)):
            raise ValueError("proposal source references must be unique")
        return self


class MemoryExtractionResult(_StrictModel):
    drafts: tuple[MemoryMutationDraft, ...] = ()


class MemoryExtractionSource(_StrictModel):
    source_ref: str = Field(min_length=1, max_length=300)
    user_text: str = Field(min_length=1, max_length=20000)
    assistant_text: str = Field(default="", max_length=40000)


class MemoryConsolidationResult(_StrictModel):
    drafts: tuple[MemoryMutationDraft, ...] = ()

    @model_validator(mode="after")
    def _merge_only(self) -> "MemoryConsolidationResult":
        if any(draft.operation != "merge" for draft in self.drafts):
            raise ValueError("consolidation may only propose merge operations")
        return self


class MemoryProposal(_StrictModel):
    schema_version: Literal["memory-proposal/v1"] = "memory-proposal/v1"
    proposal_id: str = Field(pattern=r"^MP-[0-9A-F]{32}$")
    operation_key: str = Field(min_length=1, max_length=300)
    workspace_id: str = Field(min_length=1)
    owner_user_id: str | None = None
    draft: MemoryMutationDraft
    proposed_by: str = Field(min_length=1)
    proposed_at: datetime
    status: MemoryProposalStatus = "pending"
    reviewer_user_id: str | None = None
    reviewed_at: datetime | None = None
    review_reason: str | None = Field(default=None, max_length=1000)
    applied_memory_ids: tuple[str, ...] = ()
    applied_topic_id: str | None = Field(default=None, pattern=r"^MT-[0-9A-F]{32}$")

    @model_validator(mode="after")
    def _coherent_review(self) -> "MemoryProposal":
        if self.draft.scope == "user" and not self.owner_user_id:
            raise ValueError("user Memory proposal requires an owner")
        if self.draft.scope == "workspace" and self.owner_user_id is not None:
            raise ValueError("workspace Memory proposal cannot have a user owner")
        reviewed = self.status != "pending"
        if reviewed != bool(self.reviewer_user_id and self.reviewed_at):
            raise ValueError("reviewed proposal requires reviewer and timestamp")
        if self.status == "rejected" and not self.review_reason:
            raise ValueError("rejected proposal requires a reason")
        return self


class MemoryMaintenanceState(_StrictModel):
    schema_version: Literal["memory-maintenance/v1"] = "memory-maintenance/v1"
    maintenance_id: str = Field(pattern=r"^MM-[0-9A-F]{32}$")
    scope: MemoryScope
    workspace_id: str = Field(min_length=1)
    owner_user_id: str | None = None
    last_consolidated_at: datetime | None = None
    changes_since_consolidation: int = Field(default=0, ge=0)
    last_pruned_at: datetime | None = None

    @model_validator(mode="after")
    def _coherent_scope(self) -> "MemoryMaintenanceState":
        if self.scope == "user" and not self.owner_user_id:
            raise ValueError("user maintenance state requires an owner")
        if self.scope == "workspace" and self.owner_user_id is not None:
            raise ValueError("workspace maintenance state cannot have a user owner")
        return self


__all__ = [
    "MemoryEntry",
    "MemoryExtractionResult",
    "MemoryExtractionSource",
    "MemoryConsolidationResult",
    "MemoryKind",
    "MemoryLayer",
    "MemoryMaintenanceState",
    "MemoryMutationDraft",
    "MemoryOperation",
    "MemoryOrigin",
    "MemoryProposal",
    "MemoryProposalStatus",
    "MemoryScope",
    "MemoryStatus",
    "MemoryTopic",
    "MemoryTopicStatus",
]

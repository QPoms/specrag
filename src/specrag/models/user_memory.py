"""Private user memory contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


UserMemoryKind = Literal[
    "default_release",
    "answer_language",
    "detail_level",
    "followed_topic",
    "research_lead",
]
MemoryOrigin = Literal["explicit_user", "trusted_import"]
MemoryStatus = Literal["active", "deleted", "expired"]


class UserMemoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["user-memory/v1"] = "user-memory/v1"
    memory_id: str = Field(pattern=r"^UM-[0-9A-F]{32}$")
    owner_user_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    kind: UserMemoryKind
    key: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=4000)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    status: MemoryStatus = "active"
    origin: MemoryOrigin = "explicit_user"
    revision: int = Field(default=1, ge=1)
    supersedes_id: str | None = Field(default=None, pattern=r"^UM-[0-9A-F]{32}$")
    operation_key: str = ""

    @model_validator(mode="after")
    def _coherent_dates(self) -> "UserMemoryItem":
        if self.updated_at < self.created_at:
            raise ValueError("memory update cannot precede creation")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("memory expiry must follow creation")
        if self.revision == 1 and self.supersedes_id is not None:
            raise ValueError("first memory revision cannot supersede another item")
        if self.revision > 1 and self.supersedes_id is None:
            raise ValueError("later memory revision must name its predecessor")
        return self


__all__ = ["MemoryOrigin", "MemoryStatus", "UserMemoryItem", "UserMemoryKind"]

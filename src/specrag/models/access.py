"""Identity and trusted authorization context contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class User(_StrictModel):
    user_id: str = Field(min_length=1)
    username: str = Field(min_length=1, max_length=128)
    status: Literal["active", "disabled"] = "active"
    created_at: datetime


class Workspace(_StrictModel):
    workspace_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)
    created_by: str = Field(min_length=1)
    created_at: datetime


class Membership(_StrictModel):
    workspace_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    role: Literal["member", "reviewer", "admin"]
    created_at: datetime


class ActorContext(_StrictModel):
    """Authenticated context created by trusted application code."""

    user_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    role: Literal["member", "reviewer", "admin"]
    session_id: str = Field(min_length=1)
    authenticated_at: datetime
    expires_at: datetime

    @model_validator(mode="after")
    def _valid_session_window(self) -> "ActorContext":
        if self.expires_at <= self.authenticated_at:
            raise ValueError("actor context must expire after authentication")
        return self


__all__ = ["ActorContext", "Membership", "User", "Workspace"]

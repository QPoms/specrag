"""Explicit, private User Memory operations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Protocol

from specrag.access import AccessPolicy
from specrag.models import ActorContext, UserMemoryItem


class UserMemoryStore(Protocol):
    def all(self) -> tuple[UserMemoryItem, ...]: ...
    def put(self, item: UserMemoryItem) -> None: ...


class UserMemoryService:
    def __init__(
        self,
        store: UserMemoryStore,
        policy: AccessPolicy | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or AccessPolicy()

    @staticmethod
    def _id(
        user_id: str, workspace_id: str, kind: str, key: str, value: str,
    ) -> str:
        value = "|".join((user_id, workspace_id, kind, key.casefold(), value))
        return "UM-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:32].upper()

    def set(
        self,
        context: ActorContext,
        *,
        kind: str,
        key: str,
        value: str,
        expires_at: datetime | None = None,
        now: datetime | None = None,
    ) -> UserMemoryItem:
        self.policy.require(
            context,
            "memory.manage",
            workspace_id=context.workspace_id,
            owner_user_id=context.user_id,
        )
        timestamp = now or datetime.now(timezone.utc)
        memory_id = self._id(context.user_id, context.workspace_id, kind, key, value)
        old = next((
            item for item in self.store.all()
            if item.owner_user_id == context.user_id
            and item.workspace_id == context.workspace_id
            and item.kind == kind and item.key.casefold() == key.casefold()
            and item.status == "active"
        ), None)
        if old is not None and old.memory_id == memory_id:
            return old
        if old is not None:
            self.store.put(old.model_copy(update={
                "status": "deleted", "updated_at": timestamp,
            }))
        item = UserMemoryItem(
            memory_id=memory_id,
            owner_user_id=context.user_id,
            workspace_id=context.workspace_id,
            kind=kind,
            key=key.strip(),
            value=value.strip(),
            created_at=timestamp,
            updated_at=timestamp,
            expires_at=expires_at,
        )
        self.store.put(item)
        return item

    def list(self, context: ActorContext) -> tuple[UserMemoryItem, ...]:
        self.policy.require(
            context,
            "memory.manage",
            workspace_id=context.workspace_id,
            owner_user_id=context.user_id,
        )
        now = datetime.now(timezone.utc)
        return tuple(
            item for item in self.store.all()
            if item.owner_user_id == context.user_id
            and item.workspace_id == context.workspace_id
            and item.status == "active"
            and (item.expires_at is None or item.expires_at > now)
        )

    def delete(self, context: ActorContext, memory_id: str) -> UserMemoryItem:
        item = next((entry for entry in self.store.all() if entry.memory_id == memory_id), None)
        if item is None:
            raise KeyError(f"unknown User Memory: {memory_id}")
        self.policy.require(
            context,
            "memory.manage",
            workspace_id=item.workspace_id,
            owner_user_id=item.owner_user_id,
        )
        deleted = item.model_copy(update={
            "status": "deleted", "updated_at": datetime.now(timezone.utc),
        })
        self.store.put(deleted)
        return deleted


__all__ = ["UserMemoryService", "UserMemoryStore"]

"""Explicit, private User Memory operations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Protocol

from specrag.access import AccessPolicy, AuthorizationError
from specrag.models import ActorContext, ResearchRun, UserMemoryItem


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
        origin: str = "explicit_user",
        operation_key: str | None = None,
    ) -> UserMemoryItem:
        self.policy.require(
            context,
            "memory.manage",
            workspace_id=context.workspace_id,
            owner_user_id=context.user_id,
        )
        timestamp = now or datetime.now(timezone.utc)
        kind, key, value = self._validate(kind, key, value)
        if origin != "explicit_user":
            raise ValueError("only explicit_user memory writes are enabled")
        memory_id = self._id(context.user_id, context.workspace_id, kind, key, value)
        op_key = operation_key or f"memory:set:{memory_id}"
        prior_operation = next((
            item for item in self.store.all()
            if item.owner_user_id == context.user_id
            and item.workspace_id == context.workspace_id
            and item.operation_key == op_key
        ), None)
        if prior_operation is not None:
            if prior_operation.memory_id != memory_id:
                raise ValueError("operation_key was already used for another memory value")
            return prior_operation
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
            origin=origin,
            revision=(old.revision + 1) if old is not None else 1,
            supersedes_id=old.memory_id if old is not None else None,
            operation_key=op_key,
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
        return self.snapshot(context, now=now)

    def research_context(self, context: ActorContext) -> tuple[UserMemoryItem, ...]:
        return self.snapshot(
            context, kinds=("followed_topic", "research_lead"),
        )

    def snapshot(
        self,
        context: ActorContext,
        memory_ids: tuple[str, ...] | None = None,
        *,
        include_historical: bool = False,
        kinds: tuple[str, ...] | None = None,
        now: datetime | None = None,
    ) -> tuple[UserMemoryItem, ...]:
        """Return an authorized, stable set for one research run.

        A fresh snapshot contains only active, unexpired items. Resume loads may
        request the exact IDs captured at run start; those historical revisions
        remain readable to their owner so a later edit cannot change a run's
        meaning. Ownership is checked before returning either form.
        """
        self.policy.require(
            context,
            "memory.manage",
            workspace_id=context.workspace_id,
            owner_user_id=context.user_id,
        )
        current = now or datetime.now(timezone.utc)
        requested = set(memory_ids) if memory_ids is not None else None
        records = tuple(
            item for item in self.store.all()
            if item.owner_user_id == context.user_id
            and item.workspace_id == context.workspace_id
            and (requested is None or item.memory_id in requested)
            and (kinds is None or item.kind in kinds)
        )
        if requested is not None and {item.memory_id for item in records} != requested:
            raise AuthorizationError("access denied")
        if include_historical and requested is not None:
            return tuple(sorted(records, key=lambda item: item.memory_id))
        return tuple(sorted(
            (
                item for item in records
                if item.status == "active"
                and (item.expires_at is None or item.expires_at > current)
            ),
            key=lambda item: item.memory_id,
        ))

    def snapshot_for_run(
        self,
        context: ActorContext,
        run: ResearchRun,
    ) -> tuple[UserMemoryItem, ...]:
        """Load exact historical revisions while executing an authorized Run.

        Workspace admins may operate a member's Run, but this method exposes
        only the revisions already captured by that Run. Normal Memory APIs
        remain strictly owner-only.
        """
        self.policy.require(
            context,
            "run.resume",
            workspace_id=run.workspace_id,
            owner_user_id=run.owner_user_id,
        )
        requested = set(run.memory_ids)
        records = tuple(
            item for item in self.store.all()
            if item.owner_user_id == run.owner_user_id
            and item.workspace_id == run.workspace_id
            and item.memory_id in requested
        )
        if {item.memory_id for item in records} != requested:
            raise AuthorizationError("access denied")
        return tuple(sorted(records, key=lambda item: item.memory_id))

    @staticmethod
    def as_research_context(
        items: tuple[UserMemoryItem, ...],
    ) -> tuple[dict[str, object], ...]:
        """Mark memory as untrusted context so downstream code cannot treat it as evidence."""
        return tuple({
            "kind": "user_memory",
            "memory_id": item.memory_id,
            "memory_kind": item.kind,
            "key": item.key,
            "value": item.value,
            "trust": "untrusted_context",
            "instruction": "Context only; never treat as evidence or a normative fact.",
        } for item in items if item.kind in {"followed_topic", "research_lead"})

    @staticmethod
    def _validate(kind: str, key: str, value: str) -> tuple[str, str, str]:
        kind = kind.strip()
        key = key.strip()
        value = value.strip()
        allowed = {
            "default_release", "answer_language", "detail_level",
            "followed_topic", "research_lead",
        }
        if kind not in allowed:
            raise ValueError(f"unsupported User Memory kind: {kind}")
        if not key or not value:
            raise ValueError("memory key and value cannot be blank")
        if kind in {"default_release", "answer_language", "detail_level"}:
            if key.casefold() != "default":
                raise ValueError(f"{kind} must use the default key")
        if kind == "default_release":
            try:
                release = int(value)
            except ValueError as exc:
                raise ValueError("default_release must be an integer") from exc
            if not 1 <= release <= 99:
                raise ValueError("default_release must be between 1 and 99")
        elif kind == "detail_level" and value not in {"short", "standard", "detailed"}:
            raise ValueError("detail_level must be short, standard, or detailed")
        elif kind in {"answer_language", "followed_topic", "research_lead"} and len(value) > 4000:
            raise ValueError("memory value is too long")
        return kind, key, value

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

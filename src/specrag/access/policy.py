"""Default-deny workspace authorization decisions."""

from __future__ import annotations

from typing import Literal

from specrag.models.access import ActorContext


Permission = Literal[
    "research.create",
    "run.metadata.read",
    "run.content.read",
    "run.lifecycle.manage",
    "conversation.create",
    "conversation.read",
    "conversation.write",
    "conversation.archive",
    "conversation.events.read",
    "result.read",
    "memory.user.manage",
    "memory.project.read",
    "memory.project.propose",
    "memory.project.manage",
    "memory.project.review",
    "specfact.read",
    "specfact.propose",
    "specfact.review",
    "workspace.members.manage",
]


class AuthorizationError(PermissionError):
    """Raised when trusted context does not authorize an operation."""


class AccessPolicy:
    _ROLE_PERMISSIONS: dict[str, frozenset[Permission]] = {
        "member": frozenset({
            "research.create",
            "run.metadata.read",
            "run.content.read",
            "run.lifecycle.manage",
            "conversation.create",
            "conversation.read",
            "conversation.write",
            "conversation.archive",
            "conversation.events.read",
            "result.read",
            "memory.user.manage",
            "memory.project.read",
            "memory.project.propose",
            "specfact.read",
            "specfact.propose",
        }),
        "reviewer": frozenset({
            "research.create",
            "run.metadata.read",
            "run.content.read",
            "run.lifecycle.manage",
            "conversation.create",
            "conversation.read",
            "conversation.write",
            "conversation.archive",
            "conversation.events.read",
            "result.read",
            "memory.user.manage",
            "memory.project.read",
            "memory.project.propose",
            "memory.project.manage",
            "memory.project.review",
            "specfact.read",
            "specfact.propose",
            "specfact.review",
        }),
        "admin": frozenset({
            "research.create",
            "run.metadata.read",
            "run.content.read",
            "run.lifecycle.manage",
            "conversation.create",
            "conversation.read",
            "conversation.write",
            "conversation.archive",
            "conversation.events.read",
            "result.read",
            "memory.user.manage",
            "memory.project.read",
            "memory.project.propose",
            "memory.project.manage",
            "memory.project.review",
            "specfact.read",
            "specfact.propose",
            "specfact.review",
            "workspace.members.manage",
        }),
    }

    def require(
        self,
        context: ActorContext | None,
        permission: Permission,
        *,
        workspace_id: str,
        owner_user_id: str | None = None,
    ) -> None:
        if context is None or context.workspace_id != workspace_id:
            raise AuthorizationError("access denied")
        if permission not in self._ROLE_PERMISSIONS.get(context.role, frozenset()):
            raise AuthorizationError("access denied")
        if permission == "memory.user.manage" and owner_user_id is not None:
            if context.user_id != owner_user_id:
                raise AuthorizationError("access denied")
        private_content = permission in {
            "conversation.read",
            "conversation.write",
            "conversation.archive",
            "conversation.events.read",
            "result.read",
            "run.content.read",
        }
        if private_content and owner_user_id is not None:
            if context.user_id != owner_user_id:
                raise AuthorizationError("access denied")
        lifecycle_scoped = permission in {
            "run.metadata.read",
            "run.lifecycle.manage",
        }
        if lifecycle_scoped and owner_user_id is not None:
            if context.role != "admin" and context.user_id != owner_user_id:
                raise AuthorizationError("access denied")


__all__ = ["AccessPolicy", "AuthorizationError", "Permission"]

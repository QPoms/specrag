"""Local account, session, and workspace application service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
from typing import Protocol
from uuid import uuid4

from specrag.models import ActorContext, Membership, User, Workspace

from .policy import AccessPolicy, AuthorizationError


class IdentityStore(Protocol):
    def user_count(self) -> int: ...
    def user_by_name(self, username: str) -> User | None: ...
    def user_by_id(self, user_id: str) -> User | None: ...
    def put_user(self, user: User, salt: bytes, password_hash: bytes) -> None: ...
    def credential(self, user_id: str) -> tuple[bytes, bytes] | None: ...
    def put_session(
        self, session_id: str, token_hash: str, user_id: str,
        created_at: datetime, expires_at: datetime,
    ) -> None: ...
    def session_user(
        self, token_hash: str, now: datetime,
    ) -> tuple[str, str, datetime, datetime] | None: ...
    def revoke_session(self, token_hash: str, now: datetime) -> bool: ...
    def put_workspace(self, workspace: Workspace, membership: Membership) -> None: ...
    def workspace_by_name_or_id(self, value: str) -> Workspace | None: ...
    def workspaces_for_user(
        self, user_id: str,
    ) -> tuple[tuple[Workspace, Membership], ...]: ...
    def membership(self, workspace_id: str, user_id: str) -> Membership | None: ...
    def memberships_for_workspace(
        self, workspace_id: str,
    ) -> tuple[tuple[User, Membership], ...]: ...
    def put_membership(self, membership: Membership) -> None: ...
    def remove_membership(self, workspace_id: str, user_id: str) -> bool: ...


class LocalIdentityService:
    def __init__(
        self,
        store: IdentityStore,
        policy: AccessPolicy | None = None,
        session_duration: timedelta = timedelta(hours=12),
    ) -> None:
        self.store = store
        self.policy = policy or AccessPolicy()
        self.session_duration = session_duration

    @staticmethod
    def _password_hash(password: str, salt: bytes) -> bytes:
        return hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32
        )

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def bootstrap(self, username: str, password: str) -> User:
        if self.store.user_count() != 0:
            raise AuthorizationError("bootstrap is available only before the first account")
        return self._create_user(username, password)

    def _create_user(self, username: str, password: str) -> User:
        username = username.strip()
        if not username:
            raise ValueError("username must not be blank")
        if self.store.user_by_name(username) is not None:
            raise ValueError("username already exists")
        if len(password) < 10:
            raise ValueError("password must contain at least 10 characters")
        salt = secrets.token_bytes(16)
        user = User(
            user_id="USR-" + uuid4().hex.upper(),
            username=username,
            created_at=datetime.now(timezone.utc),
        )
        self.store.put_user(user, salt, self._password_hash(password, salt))
        return user

    def login(self, username: str, password: str) -> tuple[str, datetime]:
        user = self.store.user_by_name(username.strip())
        if user is None or user.status != "active":
            raise AuthorizationError("invalid username or password")
        credential = self.store.credential(user.user_id)
        if credential is None:
            raise AuthorizationError("invalid username or password")
        salt, expected = credential
        supplied = self._password_hash(password, salt)
        if not hmac.compare_digest(supplied, expected):
            raise AuthorizationError("invalid username or password")
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires_at = now + self.session_duration
        self.store.put_session(
            "SES-" + uuid4().hex.upper(), self._token_hash(token), user.user_id,
            now, expires_at,
        )
        return token, expires_at

    def logout(self, token: str) -> None:
        if not self.store.revoke_session(self._token_hash(token), datetime.now(timezone.utc)):
            raise AuthorizationError("invalid or expired session")

    def create_workspace(self, token: str, name: str) -> Workspace:
        user, _, _, _ = self._session(token)
        name = name.strip()
        if not name:
            raise ValueError("workspace name must not be blank")
        if self.store.workspace_by_name_or_id(name) is not None:
            raise ValueError("workspace name already exists")
        now = datetime.now(timezone.utc)
        workspace = Workspace(
            workspace_id="WS-" + uuid4().hex.upper(),
            name=name,
            created_by=user.user_id,
            created_at=now,
        )
        membership = Membership(
            workspace_id=workspace.workspace_id,
            user_id=user.user_id,
            role="admin",
            created_at=now,
        )
        self.store.put_workspace(workspace, membership)
        return workspace

    def workspaces(self, token: str) -> tuple[tuple[Workspace, Membership], ...]:
        user, _, _, _ = self._session(token)
        return self.store.workspaces_for_user(user.user_id)

    def context(self, token: str, workspace_name_or_id: str) -> ActorContext:
        user, session_id, authenticated_at, expires_at = self._session(token)
        workspace = self.store.workspace_by_name_or_id(workspace_name_or_id)
        if workspace is None:
            raise AuthorizationError("access denied")
        membership = self.store.membership(workspace.workspace_id, user.user_id)
        if membership is None:
            raise AuthorizationError("access denied")
        return ActorContext(
            user_id=user.user_id,
            workspace_id=workspace.workspace_id,
            role=membership.role,
            session_id=session_id,
            authenticated_at=authenticated_at,
            expires_at=expires_at,
        )

    def add_user(
        self, context: ActorContext, username: str, password: str, role: str,
    ) -> Membership:
        self.policy.require(
            context, "workspace.members.manage", workspace_id=context.workspace_id
        )
        user = self.store.user_by_name(username.strip())
        if user is None:
            user = self._create_user(username, password)
        membership = Membership(
            workspace_id=context.workspace_id,
            user_id=user.user_id,
            role=role,
            created_at=datetime.now(timezone.utc),
        )
        self.store.put_membership(membership)
        return membership

    def members(
        self, context: ActorContext,
    ) -> tuple[tuple[User, Membership], ...]:
        self.policy.require(
            context, "workspace.members.manage", workspace_id=context.workspace_id
        )
        return self.store.memberships_for_workspace(context.workspace_id)

    def set_role(self, context: ActorContext, username: str, role: str) -> Membership:
        self.policy.require(
            context, "workspace.members.manage", workspace_id=context.workspace_id
        )
        user = self.store.user_by_name(username)
        if user is None:
            raise KeyError(f"unknown user: {username}")
        current = self.store.membership(context.workspace_id, user.user_id)
        if current is None:
            raise KeyError(f"user is not a workspace member: {username}")
        if user.user_id == context.user_id and current.role == "admin" and role != "admin":
            raise ValueError("administrators cannot demote their own membership")
        updated = current.model_copy(update={"role": role})
        self.store.put_membership(updated)
        return updated

    def remove_user(self, context: ActorContext, username: str) -> None:
        self.policy.require(
            context, "workspace.members.manage", workspace_id=context.workspace_id
        )
        user = self.store.user_by_name(username)
        if user is None:
            raise KeyError(f"user is not a workspace member: {username}")
        if user.user_id == context.user_id:
            raise ValueError("administrators cannot remove their own membership")
        if not self.store.remove_membership(context.workspace_id, user.user_id):
            raise KeyError(f"user is not a workspace member: {username}")

    def _session(self, token: str) -> tuple[User, str, datetime, datetime]:
        record = self.store.session_user(self._token_hash(token), datetime.now(timezone.utc))
        if record is None:
            raise AuthorizationError("invalid or expired session")
        user_id, session_id, authenticated_at, expires_at = record
        user = self.store.user_by_id(user_id)
        if user is None or user.status != "active":
            raise AuthorizationError("invalid or expired session")
        return user, session_id, authenticated_at, expires_at


__all__ = ["IdentityStore", "LocalIdentityService"]

"""SQLite adapter for local accounts, sessions, and memberships."""

from __future__ import annotations

from datetime import datetime

from specrag.models import Membership, User, Workspace

from .database import SQLiteDatabase


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SQLiteIdentityStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    @property
    def _connection(self):
        return self.database.connection

    def user_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM users").fetchone()
        return int(row[0])

    def user_by_name(self, username: str) -> User | None:
        row = self._connection.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
        ).fetchone()
        return self._user(row) if row else None

    def user_by_id(self, user_id: str) -> User | None:
        row = self._connection.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return self._user(row) if row else None

    @staticmethod
    def _user(row) -> User:
        return User(
            user_id=row["user_id"], username=row["username"],
            status=row["status"], created_at=_dt(row["created_at"]),
        )

    def put_user(self, user: User, salt: bytes, password_hash: bytes) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO users(user_id, username, status, created_at) VALUES (?, ?, ?, ?)",
                (user.user_id, user.username, user.status, user.created_at.isoformat()),
            )
            self._connection.execute(
                "INSERT INTO credentials(user_id, salt, password_hash) VALUES (?, ?, ?)",
                (user.user_id, salt, password_hash),
            )

    def credential(self, user_id: str) -> tuple[bytes, bytes] | None:
        row = self._connection.execute(
            "SELECT salt, password_hash FROM credentials WHERE user_id = ?", (user_id,)
        ).fetchone()
        return (bytes(row[0]), bytes(row[1])) if row else None

    def put_session(
        self, session_id: str, token_hash: str, user_id: str,
        created_at: datetime, expires_at: datetime,
    ) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO sessions(session_id, token_hash, user_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, token_hash, user_id, created_at.isoformat(), expires_at.isoformat()),
            )

    def session_user(
        self, token_hash: str, now: datetime,
    ) -> tuple[str, str, datetime, datetime] | None:
        row = self._connection.execute(
            "SELECT user_id, session_id, created_at, expires_at FROM sessions "
            "WHERE token_hash = ? AND revoked_at IS NULL AND expires_at > ?",
            (token_hash, now.isoformat()),
        ).fetchone()
        return (row[0], row[1], _dt(row[2]), _dt(row[3])) if row else None

    def revoke_session(self, token_hash: str, now: datetime) -> bool:
        with self._connection:
            cursor = self._connection.execute(
                "UPDATE sessions SET revoked_at = ? "
                "WHERE token_hash = ? AND revoked_at IS NULL AND expires_at > ?",
                (now.isoformat(), token_hash, now.isoformat()),
            )
        return cursor.rowcount == 1

    def put_workspace(self, workspace: Workspace, membership: Membership) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO workspaces(workspace_id, name, created_by, created_at) "
                "VALUES (?, ?, ?, ?)",
                (workspace.workspace_id, workspace.name, workspace.created_by,
                 workspace.created_at.isoformat()),
            )
            self._put_membership(membership)

    def workspace_by_name_or_id(self, value: str) -> Workspace | None:
        row = self._connection.execute(
            "SELECT * FROM workspaces WHERE workspace_id = ? OR name = ? COLLATE NOCASE",
            (value, value),
        ).fetchone()
        if not row:
            return None
        return Workspace(
            workspace_id=row["workspace_id"], name=row["name"],
            created_by=row["created_by"], created_at=_dt(row["created_at"]),
        )

    def workspaces_for_user(
        self, user_id: str,
    ) -> tuple[tuple[Workspace, Membership], ...]:
        rows = self._connection.execute(
            "SELECT w.*, m.role, m.created_at AS membership_created_at "
            "FROM memberships m JOIN workspaces w ON w.workspace_id = m.workspace_id "
            "WHERE m.user_id = ? ORDER BY w.name COLLATE NOCASE, w.workspace_id",
            (user_id,),
        ).fetchall()
        return tuple((
            Workspace(
                workspace_id=row["workspace_id"],
                name=row["name"],
                created_by=row["created_by"],
                created_at=_dt(row["created_at"]),
            ),
            Membership(
                workspace_id=row["workspace_id"],
                user_id=user_id,
                role=row["role"],
                created_at=_dt(row["membership_created_at"]),
            ),
        ) for row in rows)

    def membership(self, workspace_id: str, user_id: str) -> Membership | None:
        row = self._connection.execute(
            "SELECT * FROM memberships WHERE workspace_id = ? AND user_id = ?",
            (workspace_id, user_id),
        ).fetchone()
        if not row:
            return None
        return Membership(
            workspace_id=row["workspace_id"], user_id=row["user_id"],
            role=row["role"], created_at=_dt(row["created_at"]),
        )

    def memberships_for_workspace(
        self, workspace_id: str,
    ) -> tuple[tuple[User, Membership], ...]:
        rows = self._connection.execute(
            "SELECT u.*, m.role, m.created_at AS membership_created_at "
            "FROM memberships m JOIN users u ON u.user_id = m.user_id "
            "WHERE m.workspace_id = ? "
            "ORDER BY u.username COLLATE NOCASE, u.user_id",
            (workspace_id,),
        ).fetchall()
        return tuple((
            self._user(row),
            Membership(
                workspace_id=workspace_id,
                user_id=row["user_id"],
                role=row["role"],
                created_at=_dt(row["membership_created_at"]),
            ),
        ) for row in rows)

    def _put_membership(self, membership: Membership) -> None:
        self._connection.execute(
            "INSERT INTO memberships(workspace_id, user_id, role, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(workspace_id, user_id) "
            "DO UPDATE SET role = excluded.role",
            (membership.workspace_id, membership.user_id, membership.role,
             membership.created_at.isoformat()),
        )

    def put_membership(self, membership: Membership) -> None:
        with self._connection:
            self._put_membership(membership)

    def remove_membership(self, workspace_id: str, user_id: str) -> bool:
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM memberships WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            )
        return cursor.rowcount == 1


__all__ = ["SQLiteIdentityStore"]

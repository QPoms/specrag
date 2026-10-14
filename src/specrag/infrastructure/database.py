"""SQLite connection and monotonic schema migrations."""

from __future__ import annotations

from pathlib import Path
import sqlite3


DEFAULT_DATABASE_PATH = Path(".local/state/specrag.sqlite3")


_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, """
        CREATE TABLE users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            status TEXT NOT NULL CHECK (status IN ('active', 'disabled')),
            created_at TEXT NOT NULL
        );
        CREATE TABLE credentials (
            user_id TEXT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
            salt BLOB NOT NULL,
            password_hash BLOB NOT NULL
        );
        CREATE TABLE workspaces (
            workspace_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            created_by TEXT NOT NULL REFERENCES users(user_id),
            created_at TEXT NOT NULL
        );
        CREATE TABLE memberships (
            workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('member', 'reviewer', 'admin')),
            created_at TEXT NOT NULL,
            PRIMARY KEY (workspace_id, user_id)
        );
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            token_hash TEXT NOT NULL UNIQUE,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            revoked_at TEXT
        );
    """),
    (2, """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL UNIQUE,
            workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
            owner_user_id TEXT NOT NULL REFERENCES users(user_id),
            status TEXT NOT NULL,
            cancel_requested_at TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX runs_workspace_owner ON runs(workspace_id, owner_user_id);
        CREATE TABLE run_events (
            event_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            workspace_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            operation_key TEXT,
            payload_json TEXT NOT NULL,
            UNIQUE(run_id, sequence),
            UNIQUE(run_id, operation_key)
        );
        CREATE TABLE specfact_proposals (
            proposal_id TEXT PRIMARY KEY,
            operation_key TEXT NOT NULL UNIQUE,
            workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX proposals_workspace_status
            ON specfact_proposals(workspace_id, status);
        CREATE TABLE specfacts (
            fact_id TEXT PRIMARY KEY,
            proposal_id TEXT NOT NULL UNIQUE REFERENCES specfact_proposals(proposal_id),
            workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX facts_workspace_status ON specfacts(workspace_id, status);
        CREATE TABLE user_memory (
            memory_id TEXT PRIMARY KEY,
            owner_user_id TEXT NOT NULL REFERENCES users(user_id),
            workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
            kind TEXT NOT NULL,
            memory_key TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX memory_owner_workspace
            ON user_memory(owner_user_id, workspace_id, status);
    """),
)


class SQLiteDatabase:
    def __init__(self, path: Path = DEFAULT_DATABASE_PATH) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def _migrate(self) -> None:
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        applied = {
            int(row[0])
            for row in self.connection.execute("SELECT version FROM schema_migrations")
        }
        for version, script in _MIGRATIONS:
            if version in applied:
                continue
            with self.connection:
                self.connection.executescript(script)
                self.connection.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)", (version,)
                )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["DEFAULT_DATABASE_PATH", "SQLiteDatabase"]

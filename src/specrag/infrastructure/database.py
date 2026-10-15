"""SQLite connection and monotonic schema migrations."""

from __future__ import annotations

from pathlib import Path
import json
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
    (3, """
        CREATE TABLE memory_topics (
            topic_id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
            slug TEXT NOT NULL,
            status TEXT NOT NULL,
            pinned INTEGER NOT NULL CHECK (pinned IN (0, 1)),
            payload_json TEXT NOT NULL,
            UNIQUE(workspace_id, slug)
        );
        CREATE INDEX memory_topics_workspace
            ON memory_topics(workspace_id, status, pinned);
        CREATE TABLE memory_entries (
            memory_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL CHECK (scope IN ('user', 'workspace')),
            owner_user_id TEXT REFERENCES users(user_id),
            workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
            topic_id TEXT REFERENCES memory_topics(topic_id),
            status TEXT NOT NULL,
            operation_key TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX memory_entries_scope
            ON memory_entries(workspace_id, scope, owner_user_id, status, topic_id);
        CREATE UNIQUE INDEX memory_entries_operation
            ON memory_entries(workspace_id, operation_key);
        CREATE TABLE memory_proposals (
            proposal_id TEXT PRIMARY KEY,
            operation_key TEXT NOT NULL UNIQUE,
            workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
            owner_user_id TEXT REFERENCES users(user_id),
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX memory_proposals_scope
            ON memory_proposals(workspace_id, owner_user_id, status);
        CREATE TABLE memory_maintenance (
            maintenance_id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
            owner_user_id TEXT REFERENCES users(user_id),
            payload_json TEXT NOT NULL
        );
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
                if version == 3:
                    self._migrate_legacy_memory()
                self.connection.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)", (version,)
                )

    def _migrate_legacy_memory(self) -> None:
        exists = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_memory'"
        ).fetchone()
        if exists is None:
            return
        rows = self.connection.execute(
            "SELECT payload_json FROM user_memory ORDER BY memory_id"
        ).fetchall()
        for row in rows:
            old = json.loads(row[0])
            status = {
                "active": "active",
                "deleted": "retracted",
                "expired": "archived",
            }[old["status"]]
            payload = {
                "schema_version": "memory-entry/v1",
                "memory_id": old["memory_id"],
                "scope": "user",
                "layer": "core",
                "workspace_id": old["workspace_id"],
                "owner_user_id": old["owner_user_id"],
                "topic_id": None,
                "kind": old["kind"],
                "key": old["key"],
                "content": old["value"],
                "created_by": old["owner_user_id"],
                "created_at": old["created_at"],
                "updated_at": old["updated_at"],
                "expires_at": old.get("expires_at"),
                "status": status,
                "origin": "migrated",
                "revision": old.get("revision", 1),
                "supersedes_id": old.get("supersedes_id"),
                "merged_from_ids": [],
                "source_refs": [],
                "operation_key": old.get("operation_key") or (
                    "memory:migrated:" + old["memory_id"]
                ),
            }
            self.connection.execute(
                "INSERT OR IGNORE INTO memory_entries("
                "memory_id, scope, owner_user_id, workspace_id, topic_id, status, "
                "operation_key, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    payload["memory_id"],
                    payload["scope"],
                    payload["owner_user_id"],
                    payload["workspace_id"],
                    None,
                    payload["status"],
                    payload["operation_key"],
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
        self.connection.execute("DROP TABLE user_memory")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["DEFAULT_DATABASE_PATH", "SQLiteDatabase"]

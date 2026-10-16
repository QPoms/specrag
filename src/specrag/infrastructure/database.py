"""SQLite connection and monotonic schema migrations."""

from __future__ import annotations

from pathlib import Path
import json
import sqlite3
from threading import Lock, local


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
    (4, """
        CREATE TABLE corpus_versions (
            version_id TEXT PRIMARY KEY,
            release INTEGER NOT NULL,
            specification TEXT NOT NULL,
            version TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            UNIQUE(release, specification, version)
        );
        CREATE INDEX corpus_versions_release_status
            ON corpus_versions(release, status, specification);
        CREATE TABLE active_corpus_versions (
            release INTEGER NOT NULL,
            specification TEXT NOT NULL,
            version_id TEXT NOT NULL REFERENCES corpus_versions(version_id),
            PRIMARY KEY(release, specification)
        );
        CREATE TABLE run_corpus_bindings (
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            version_id TEXT NOT NULL REFERENCES corpus_versions(version_id),
            PRIMARY KEY(run_id, version_id)
        );
        CREATE INDEX run_corpus_versions
            ON run_corpus_bindings(version_id, run_id);
    """),
    (5, """
        CREATE TABLE run_jobs (
            run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
            action TEXT NOT NULL CHECK (action IN ('execute', 'resume')),
            state TEXT NOT NULL CHECK (state IN ('queued', 'leased')),
            available_at TEXT NOT NULL,
            lease_owner TEXT,
            lease_expires_at TEXT,
            attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (
                (state = 'queued' AND lease_owner IS NULL AND lease_expires_at IS NULL)
                OR
                (state = 'leased' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
            )
        );
        CREATE INDEX run_jobs_claimable
            ON run_jobs(state, available_at, lease_expires_at, created_at);
    """),
)


class SQLiteDatabase:
    def __init__(self, path: Path = DEFAULT_DATABASE_PATH) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._local = local()
        self._connections: list[sqlite3.Connection] = []
        self._connection_lock = Lock()
        self._closed = False
        connection = self._open_connection()
        self._local.connection = connection
        connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        self._connections.append(connection)
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            return connection
        with self._connection_lock:
            if self._closed:
                raise RuntimeError("SQLite database is closed")
            connection = self._open_connection()
            self._local.connection = connection
            return connection

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
        with self._connection_lock:
            if self._closed:
                return
            self._closed = True
            for connection in self._connections:
                connection.close()
            self._connections.clear()

    def __enter__(self) -> "SQLiteDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["DEFAULT_DATABASE_PATH", "SQLiteDatabase"]

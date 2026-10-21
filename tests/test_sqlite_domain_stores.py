from datetime import datetime, timedelta, timezone
import json
from threading import Thread

from specrag.infrastructure import (
    SQLiteDatabase,
    SQLiteMemoryStore,
    SQLiteRunStore,
)
from specrag.models import MemoryEntry, ResearchRun, RunEvent


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_sqlite_database_uses_one_connection_per_thread(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "threaded.sqlite3")
    main_connection = database.connection
    observations: list[tuple[bool, int]] = []

    def inspect_connection() -> None:
        connection = database.connection
        observations.append((
            connection is main_connection,
            connection.execute("PRAGMA foreign_keys").fetchone()[0],
        ))

    thread = Thread(target=inspect_connection)
    thread.start()
    thread.join()
    database.close()

    assert observations == [(False, 1)]


def test_sqlite_enables_foreign_keys_wal_and_records_schema_versions(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "state.sqlite3")
    try:
        assert database.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert database.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        versions = database.connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [row[0] for row in versions] == [1, 2, 3, 4, 5, 6]
    finally:
        database.close()


def test_run_events_are_idempotent_by_operation_key(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "state.sqlite3")
    try:
        connection = database.connection
        with connection:
            connection.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?)",
                ("alice", "alice", "active", NOW.isoformat()),
            )
            connection.execute(
                "INSERT INTO workspaces VALUES (?, ?, ?, ?)",
                ("WA", "Workspace A", "alice", NOW.isoformat()),
            )
        store = SQLiteRunStore(database)
        run = ResearchRun(
            run_id="RUN-" + "1" * 32, thread_id="THREAD-" + "1" * 32,
            workspace_id="WA", owner_user_id="alice", question="question",
            release=17, created_at=NOW,
        )
        store.put(run)
        first = RunEvent(
            event_id="EVT-" + "1" * 32, run_id=run.run_id, workspace_id="WA",
            sequence=1, event_type="run_started", occurred_at=NOW,
            operation_key="start",
        )
        duplicate = first.model_copy(update={"event_id": "EVT-" + "2" * 32, "sequence": 2})
        assert store.append_event(first) == first
        assert store.append_event(duplicate) == first
        assert store.events(run.run_id) == (first,)
    finally:
        database.close()


def test_memory_revisions_survive_sqlite_upserts(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "state.sqlite3")
    try:
        connection = database.connection
        with connection:
            connection.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?)",
                ("alice", "alice", "active", NOW.isoformat()),
            )
            connection.execute(
                "INSERT INTO workspaces VALUES (?, ?, ?, ?)",
                ("WA", "Workspace A", "alice", NOW.isoformat()),
            )
        store = SQLiteMemoryStore(database)
        first = MemoryEntry(
            memory_id="MEM-" + "1" * 32,
            scope="user",
            layer="core",
            owner_user_id="alice",
            workspace_id="WA",
            kind="detail_level",
            key="default",
            content="short",
            created_by="alice",
            created_at=NOW,
            updated_at=NOW,
            operation_key="memory:first",
        )
        second = first.model_copy(update={
            "memory_id": "MEM-" + "2" * 32,
            "content": "detailed",
            "created_at": NOW + timedelta(minutes=1),
            "updated_at": NOW + timedelta(minutes=1),
            "operation_key": "memory:second",
        })
        store.put_entry(first)
        store.put_entry(second)
        assert {item.content for item in store.all_entries()} == {"short", "detailed"}
    finally:
        database.close()


def test_legacy_user_memory_is_migrated_once_to_core_memory(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    database = SQLiteDatabase(path)
    connection = database.connection
    with connection:
        connection.execute("DROP TABLE memory_maintenance")
        connection.execute("DROP TABLE memory_proposals")
        connection.execute("DROP TABLE memory_entries")
        connection.execute("DROP TABLE memory_topics")
        connection.execute(
            "CREATE TABLE user_memory ("
            "memory_id TEXT PRIMARY KEY, owner_user_id TEXT NOT NULL, "
            "workspace_id TEXT NOT NULL, kind TEXT NOT NULL, memory_key TEXT NOT NULL, "
            "status TEXT NOT NULL, payload_json TEXT NOT NULL)"
        )
        connection.execute("DELETE FROM schema_migrations WHERE version = 3")
        connection.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?)",
            ("alice", "alice", "active", NOW.isoformat()),
        )
        connection.execute(
            "INSERT INTO workspaces VALUES (?, ?, ?, ?)",
            ("WA", "Workspace A", "alice", NOW.isoformat()),
        )
        payload = {
            "schema_version": "user-memory/v1",
            "memory_id": "UM-" + "A" * 32,
            "owner_user_id": "alice",
            "workspace_id": "WA",
            "kind": "research_lead",
            "key": "lead",
            "value": "Study registration recovery",
            "created_at": NOW.isoformat(),
            "updated_at": NOW.isoformat(),
            "expires_at": None,
            "status": "active",
            "origin": "explicit_user",
            "revision": 1,
            "supersedes_id": None,
            "operation_key": "legacy-operation",
        }
        connection.execute(
            "INSERT INTO user_memory VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                payload["memory_id"],
                "alice",
                "WA",
                "research_lead",
                "lead",
                "active",
                json.dumps(payload),
            ),
        )
    database.close()

    migrated = SQLiteDatabase(path)
    try:
        item = SQLiteMemoryStore(migrated).all_entries()[0]
        assert item.memory_id == payload["memory_id"]
        assert item.scope == "user"
        assert item.layer == "core"
        assert item.content == payload["value"]
        assert item.origin == "migrated"
        assert migrated.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'user_memory'"
        ).fetchone() is None
    finally:
        migrated.close()

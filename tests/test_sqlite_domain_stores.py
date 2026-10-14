from datetime import datetime, timedelta, timezone

from specrag.infrastructure import (
    SQLiteDatabase,
    SQLiteRunStore,
    SQLiteUserMemoryStore,
)
from specrag.models import ResearchRun, RunEvent, UserMemoryItem


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_sqlite_enables_foreign_keys_wal_and_records_schema_versions(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "state.sqlite3")
    try:
        assert database.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert database.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        versions = database.connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [row[0] for row in versions] == [1, 2]
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
        store = SQLiteUserMemoryStore(database)
        first = UserMemoryItem(
            memory_id="UM-" + "1" * 32, owner_user_id="alice", workspace_id="WA",
            kind="detail_level", key="default", value="short",
            created_at=NOW, updated_at=NOW,
        )
        second = first.model_copy(update={
            "memory_id": "UM-" + "2" * 32, "value": "detailed",
            "created_at": NOW + timedelta(minutes=1),
            "updated_at": NOW + timedelta(minutes=1),
        })
        store.put(first)
        store.put(second)
        assert {item.value for item in store.all()} == {"short", "detailed"}
    finally:
        database.close()

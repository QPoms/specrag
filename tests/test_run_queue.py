from __future__ import annotations

from datetime import datetime, timedelta, timezone

from specrag.infrastructure import SQLiteDatabase, SQLiteRunQueue


def _run_id(suffix: str = "A") -> str:
    return "RUN-" + suffix * 32


def _insert_run(database: SQLiteDatabase, run_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with database.connection:
        database.connection.execute(
            "INSERT INTO users(user_id, username, status, created_at) "
            "VALUES ('USR-1', 'queue-user', 'active', ?)",
            (now,),
        )
        database.connection.execute(
            "INSERT INTO workspaces(workspace_id, name, created_by, created_at) "
            "VALUES ('WS-1', 'Queue Workspace', 'USR-1', ?)",
            (now,),
        )
        database.connection.execute(
            "INSERT INTO runs(run_id, thread_id, workspace_id, owner_user_id, "
            "status, cancel_requested_at, payload_json) VALUES (?, ?, ?, ?, ?, NULL, ?)",
            (
                run_id,
                "THREAD-" + run_id[-32:],
                "WS-1",
                "USR-1",
                "queued",
                "{}",
            ),
        )


def test_sqlite_run_queue_claims_one_live_lease_and_completes(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "queue.sqlite3")
    try:
        run_id = _run_id()
        _insert_run(database, run_id)
        queue = SQLiteRunQueue(database)
        queue.enqueue(run_id, "execute")

        claimed = queue.claim("WORKER-1", lease_seconds=60)
        assert claimed == (run_id, "execute", 1)
        assert queue.claim("WORKER-2", lease_seconds=60) is None
        assert queue.renew(run_id, "WORKER-1", lease_seconds=60)
        assert queue.complete(run_id, "WORKER-1")
        assert queue.pending_count() == 0
    finally:
        database.close()


def test_sqlite_run_queue_reclaims_expired_lease(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "queue-expired.sqlite3")
    try:
        run_id = _run_id("B")
        _insert_run(database, run_id)
        queue = SQLiteRunQueue(database)
        queue.enqueue(run_id, "resume")
        assert queue.claim("WORKER-1", lease_seconds=60) is not None
        expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with database.connection:
            database.connection.execute(
                "UPDATE run_jobs SET lease_expires_at=? WHERE run_id=?",
                (expired, run_id),
            )

        reclaimed = queue.claim("WORKER-2", lease_seconds=60)
        assert reclaimed == (run_id, "resume", 2)
    finally:
        database.close()


def test_sqlite_run_queue_enqueue_does_not_steal_live_lease(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "queue-idempotent.sqlite3")
    try:
        run_id = _run_id("C")
        _insert_run(database, run_id)
        queue = SQLiteRunQueue(database)
        queue.enqueue(run_id, "execute")
        first = queue.claim("WORKER-1", lease_seconds=60)
        queue.enqueue(run_id, "execute")
        assert first == (run_id, "execute", 1)
        assert queue.claim("WORKER-2", lease_seconds=60) is None
    finally:
        database.close()

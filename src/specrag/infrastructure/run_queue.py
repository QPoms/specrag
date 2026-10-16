"""Durable SQLite queue for asynchronous research Run execution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from .database import SQLiteDatabase


RunJobAction = Literal["execute", "resume"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteRunQueue:
    """Atomically claim Run jobs with renewable, expiring leases."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def enqueue(self, run_id: str, action: RunJobAction) -> None:
        now = _now().isoformat()
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO run_jobs(run_id, action, state, available_at, "
                "lease_owner, lease_expires_at, attempts, created_at, updated_at) "
                "VALUES (?, ?, 'queued', ?, NULL, NULL, 0, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET action=excluded.action, "
                "state=CASE WHEN run_jobs.state='leased' THEN 'leased' "
                "ELSE 'queued' END, "
                "available_at=CASE WHEN run_jobs.state='leased' "
                "THEN run_jobs.available_at ELSE excluded.available_at END, "
                "lease_owner=CASE WHEN run_jobs.state='leased' "
                "THEN run_jobs.lease_owner ELSE NULL END, "
                "lease_expires_at=CASE WHEN run_jobs.state='leased' "
                "THEN run_jobs.lease_expires_at ELSE NULL END, "
                "attempts=CASE WHEN run_jobs.state='leased' "
                "THEN run_jobs.attempts ELSE 0 END, "
                "updated_at=excluded.updated_at",
                (run_id, action, now, now, now),
            )

    def ensure(self, run_id: str, action: RunJobAction) -> None:
        """Add a missing recovery job without disturbing an active lease."""

        now = _now().isoformat()
        with self.database.connection:
            self.database.connection.execute(
                "INSERT OR IGNORE INTO run_jobs(run_id, action, state, available_at, "
                "lease_owner, lease_expires_at, attempts, created_at, updated_at) "
                "VALUES (?, ?, 'queued', ?, NULL, NULL, 0, ?, ?)",
                (run_id, action, now, now, now),
            )

    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
    ) -> tuple[str, RunJobAction, int] | None:
        now = _now()
        current = now.isoformat()
        expires = (now + timedelta(seconds=lease_seconds)).isoformat()
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT run_id, action, attempts FROM run_jobs "
                "WHERE (state='queued' AND available_at <= ?) "
                "OR (state='leased' AND lease_expires_at <= ?) "
                "ORDER BY created_at, run_id LIMIT 1",
                (current, current),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            attempts = int(row["attempts"]) + 1
            connection.execute(
                "UPDATE run_jobs SET state='leased', lease_owner=?, "
                "lease_expires_at=?, attempts=?, updated_at=? WHERE run_id=?",
                (worker_id, expires, attempts, current, row["run_id"]),
            )
            connection.commit()
            return row["run_id"], row["action"], attempts
        except BaseException:
            connection.rollback()
            raise

    def renew(
        self, run_id: str, worker_id: str, *, lease_seconds: float,
    ) -> bool:
        now = _now()
        with self.database.connection:
            cursor = self.database.connection.execute(
                "UPDATE run_jobs SET lease_expires_at=?, updated_at=? "
                "WHERE run_id=? AND state='leased' AND lease_owner=?",
                (
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    now.isoformat(),
                    run_id,
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def complete(self, run_id: str, worker_id: str) -> bool:
        with self.database.connection:
            cursor = self.database.connection.execute(
                "DELETE FROM run_jobs WHERE run_id=? AND state='leased' "
                "AND lease_owner=?",
                (run_id, worker_id),
            )
        return cursor.rowcount == 1

    def release(
        self,
        run_id: str,
        worker_id: str,
        *,
        delay_seconds: float,
    ) -> bool:
        now = _now()
        with self.database.connection:
            cursor = self.database.connection.execute(
                "UPDATE run_jobs SET state='queued', available_at=?, "
                "lease_owner=NULL, lease_expires_at=NULL, updated_at=? "
                "WHERE run_id=? AND state='leased' AND lease_owner=?",
                (
                    (now + timedelta(seconds=delay_seconds)).isoformat(),
                    now.isoformat(),
                    run_id,
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def pending_count(self) -> int:
        row = self.database.connection.execute(
            "SELECT COUNT(*) FROM run_jobs"
        ).fetchone()
        return int(row[0])


__all__ = ["RunJobAction", "SQLiteRunQueue"]

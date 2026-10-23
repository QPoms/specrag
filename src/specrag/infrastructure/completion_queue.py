"""Lease queue for terminal Run completion reconciliation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .database import SQLiteDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteRunCompletionQueue:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def claim(
        self, worker_id: str, *, lease_seconds: float
    ) -> tuple[str, int, int] | None:
        now = _now()
        current = now.isoformat()
        expires = (now + timedelta(seconds=lease_seconds)).isoformat()
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT job.run_id, job.run_attempt, job.attempts "
                "FROM run_completion_jobs job WHERE ((job.state='queued' "
                "AND job.available_at<=?) OR (job.state='leased' "
                "AND job.lease_expires_at<=?)) AND NOT EXISTS ("
                "SELECT 1 FROM run_completion_jobs earlier "
                "WHERE earlier.run_id=job.run_id "
                "AND earlier.run_attempt<job.run_attempt) "
                "ORDER BY job.created_at, job.run_id, job.run_attempt LIMIT 1",
                (current, current),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            attempts = int(row["attempts"]) + 1
            connection.execute(
                "UPDATE run_completion_jobs SET state='leased', lease_owner=?, "
                "lease_expires_at=?, attempts=?, updated_at=? WHERE run_id=? "
                "AND run_attempt=?",
                (
                    worker_id,
                    expires,
                    attempts,
                    current,
                    row["run_id"],
                    row["run_attempt"],
                ),
            )
            connection.commit()
            return row["run_id"], int(row["run_attempt"]), attempts
        except BaseException:
            connection.rollback()
            raise

    def renew(
        self,
        run_id: str,
        run_attempt: int,
        worker_id: str,
        *,
        lease_seconds: float,
    ) -> bool:
        now = _now()
        with self.database.connection:
            cursor = self.database.connection.execute(
                "UPDATE run_completion_jobs SET lease_expires_at=?, updated_at=? "
                "WHERE run_id=? AND run_attempt=? AND state='leased' "
                "AND lease_owner=?",
                (
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    now.isoformat(),
                    run_id,
                    run_attempt,
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def complete(
        self, run_id: str, run_attempt: int, worker_id: str
    ) -> bool:
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            owned = connection.execute(
                "SELECT 1 FROM run_completion_jobs WHERE run_id=? AND run_attempt=? "
                "AND state='leased' AND lease_owner=?",
                (run_id, run_attempt, worker_id),
            ).fetchone()
            if owned is None:
                connection.commit()
                return False
            connection.execute(
                "INSERT OR IGNORE INTO run_completion_receipts(run_id, run_attempt, "
                "completed_at) VALUES (?, ?, ?)",
                (run_id, run_attempt, _now().isoformat()),
            )
            connection.execute(
                "DELETE FROM run_completion_jobs WHERE run_id=? AND run_attempt=?",
                (run_id, run_attempt),
            )
            connection.commit()
            return True
        except BaseException:
            connection.rollback()
            raise

    def release(
        self,
        run_id: str,
        run_attempt: int,
        worker_id: str,
        *,
        delay_seconds: float,
    ) -> bool:
        now = _now()
        with self.database.connection:
            cursor = self.database.connection.execute(
                "UPDATE run_completion_jobs SET state='queued', available_at=?, "
                "lease_owner=NULL, lease_expires_at=NULL, updated_at=? "
                "WHERE run_id=? AND run_attempt=? AND state='leased' "
                "AND lease_owner=?",
                (
                    (now + timedelta(seconds=delay_seconds)).isoformat(),
                    now.isoformat(),
                    run_id,
                    run_attempt,
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def pending_count(self) -> int:
        row = self.database.connection.execute(
            "SELECT COUNT(*) FROM run_completion_jobs"
        ).fetchone()
        return int(row[0])


__all__ = ["SQLiteRunCompletionQueue"]

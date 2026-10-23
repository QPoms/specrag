"""Lease-based SQLite queue for serialized Conversation Agent Turns."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .database import SQLiteDatabase


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteConversationTurnQueue:
    def __init__(
        self, database: SQLiteDatabase, *, max_active_per_user: int = 2
    ) -> None:
        if max_active_per_user < 1:
            raise ValueError("max_active_per_user must be positive")
        self.database = database
        self.max_active_per_user = max_active_per_user

    def ensure(self, turn_id: str) -> None:
        now = _now().isoformat()
        with self.database.connection:
            self.database.connection.execute(
                "INSERT OR IGNORE INTO conversation_turn_jobs(turn_id, state, "
                "available_at, lease_owner, lease_expires_at, attempts, created_at, "
                "updated_at) VALUES (?, 'queued', ?, NULL, NULL, 0, ?, ?)",
                (turn_id, now, now, now),
            )

    def claim(
        self, worker_id: str, *, lease_seconds: float
    ) -> tuple[str, int] | None:
        now = _now()
        current = now.isoformat()
        expires = (now + timedelta(seconds=lease_seconds)).isoformat()
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT job.turn_id, job.attempts FROM conversation_turn_jobs job "
                "JOIN conversation_turns turn ON turn.turn_id=job.turn_id "
                "JOIN conversations conversation "
                "ON conversation.conversation_id=turn.conversation_id "
                "WHERE conversation.status='active' AND ("
                "(job.state='queued' AND job.available_at<=?) OR "
                "(job.state='leased' AND job.lease_expires_at<=?)) "
                "AND NOT EXISTS ("
                "SELECT 1 FROM conversation_turns active "
                "WHERE active.conversation_id=turn.conversation_id "
                "AND active.status='running' AND active.turn_id<>turn.turn_id"
                ") AND NOT EXISTS ("
                "SELECT 1 FROM conversation_turn_jobs other_job "
                "JOIN conversation_turns other_turn "
                "ON other_turn.turn_id=other_job.turn_id "
                "WHERE other_turn.conversation_id=turn.conversation_id "
                "AND other_job.turn_id<>job.turn_id AND other_job.state='leased' "
                "AND other_job.lease_expires_at>?"
                ") AND (SELECT COUNT(*) FROM conversation_turn_jobs user_job "
                "JOIN conversation_turns user_turn "
                "ON user_turn.turn_id=user_job.turn_id "
                "JOIN conversations user_conversation "
                "ON user_conversation.conversation_id=user_turn.conversation_id "
                "WHERE user_conversation.workspace_id=conversation.workspace_id "
                "AND user_conversation.owner_user_id=conversation.owner_user_id "
                "AND user_job.state='leased' AND user_job.lease_expires_at>?"
                ") < ? ORDER BY job.created_at, job.turn_id LIMIT 1",
                (
                    current,
                    current,
                    current,
                    current,
                    self.max_active_per_user,
                ),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            attempts = int(row["attempts"]) + 1
            connection.execute(
                "UPDATE conversation_turn_jobs SET state='leased', lease_owner=?, "
                "lease_expires_at=?, attempts=?, updated_at=? WHERE turn_id=?",
                (worker_id, expires, attempts, current, row["turn_id"]),
            )
            connection.commit()
            return row["turn_id"], attempts
        except BaseException:
            connection.rollback()
            raise

    def renew(self, turn_id: str, worker_id: str, *, lease_seconds: float) -> bool:
        now = _now()
        with self.database.connection:
            cursor = self.database.connection.execute(
                "UPDATE conversation_turn_jobs SET lease_expires_at=?, updated_at=? "
                "WHERE turn_id=? AND state='leased' AND lease_owner=?",
                (
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    now.isoformat(),
                    turn_id,
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def complete(self, turn_id: str, worker_id: str) -> bool:
        with self.database.connection:
            cursor = self.database.connection.execute(
                "DELETE FROM conversation_turn_jobs WHERE turn_id=? "
                "AND state='leased' AND lease_owner=?",
                (turn_id, worker_id),
            )
        return cursor.rowcount == 1

    def release(
        self, turn_id: str, worker_id: str, *, delay_seconds: float
    ) -> bool:
        now = _now()
        with self.database.connection:
            cursor = self.database.connection.execute(
                "UPDATE conversation_turn_jobs SET state='queued', available_at=?, "
                "lease_owner=NULL, lease_expires_at=NULL, updated_at=? WHERE turn_id=? "
                "AND state='leased' AND lease_owner=?",
                (
                    (now + timedelta(seconds=delay_seconds)).isoformat(),
                    now.isoformat(),
                    turn_id,
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def pending_count(self) -> int:
        row = self.database.connection.execute(
            "SELECT COUNT(*) FROM conversation_turn_jobs"
        ).fetchone()
        return int(row[0])


__all__ = ["SQLiteConversationTurnQueue"]

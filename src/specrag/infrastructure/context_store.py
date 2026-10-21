"""SQLite persistence for Context Harness history, results, and checkpoints."""

from __future__ import annotations

import json

from specrag.models import (
    CompactionCheckpoint,
    ContextMessage,
    ContextRecord,
    StoredToolResult,
)

from .database import SQLiteDatabase


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SQLiteContextStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def put_context(self, context: ContextRecord) -> ContextRecord:
        existing = self.get_context(context.context_id)
        if existing is not None:
            if (
                existing.workspace_id,
                existing.owner_user_id,
                existing.run_id,
            ) != (context.workspace_id, context.owner_user_id, context.run_id):
                raise ValueError("context ID is already bound to a different scope")
            return existing
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO contexts(context_id, workspace_id, owner_user_id, run_id, "
                "active_checkpoint_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    context.context_id,
                    context.workspace_id,
                    context.owner_user_id,
                    context.run_id,
                    context.active_checkpoint_id,
                    context.created_at.isoformat(),
                    context.updated_at.isoformat(),
                ),
            )
        return context

    def get_context(self, context_id: str) -> ContextRecord | None:
        row = self.database.connection.execute(
            "SELECT * FROM contexts WHERE context_id = ?", (context_id,),
        ).fetchone()
        return ContextRecord.model_validate(dict(row)) if row is not None else None

    def append_message(self, message: ContextMessage) -> ContextMessage:
        existing = self.database.connection.execute(
            "SELECT * FROM context_messages WHERE message_id = ?", (message.message_id,),
        ).fetchone()
        if existing is not None:
            return self._message(existing)
        expected = int(self.database.connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM context_messages WHERE context_id = ?",
            (message.context_id,),
        ).fetchone()[0])
        if message.sequence != expected:
            raise ValueError("context message sequence is not contiguous")
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO context_messages(message_id, context_id, sequence, role, "
                "turn_id, token_estimate, content_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message.message_id,
                    message.context_id,
                    message.sequence,
                    message.role,
                    message.turn_id,
                    message.token_estimate,
                    _json(message.content),
                    message.created_at.isoformat(),
                ),
            )
        return message

    @staticmethod
    def _message(row: object) -> ContextMessage:
        values = dict(row)  # type: ignore[arg-type]
        values["content"] = json.loads(values.pop("content_json"))
        return ContextMessage.model_validate(values)

    def messages(self, context_id: str) -> tuple[ContextMessage, ...]:
        rows = self.database.connection.execute(
            "SELECT * FROM context_messages WHERE context_id = ? ORDER BY sequence",
            (context_id,),
        ).fetchall()
        return tuple(self._message(row) for row in rows)

    def put_tool_result(self, result: StoredToolResult) -> StoredToolResult:
        existing = self.get_tool_result_by_call(result.context_id, result.tool_call_id)
        if existing is not None:
            if existing.content_sha256 != result.content_sha256:
                raise ValueError("tool call ID was reused with different content")
            return existing
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO context_tool_results(tool_result_id, context_id, tool_call_id, "
                "content_sha256, token_estimate, content_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    result.tool_result_id,
                    result.context_id,
                    result.tool_call_id,
                    result.content_sha256,
                    result.token_estimate,
                    _json(result.content),
                    result.created_at.isoformat(),
                ),
            )
        return result

    @staticmethod
    def _tool_result(row: object) -> StoredToolResult:
        values = dict(row)  # type: ignore[arg-type]
        values["content"] = json.loads(values.pop("content_json"))
        return StoredToolResult.model_validate(values)

    def get_tool_result(self, tool_result_id: str) -> StoredToolResult | None:
        row = self.database.connection.execute(
            "SELECT * FROM context_tool_results WHERE tool_result_id = ?",
            (tool_result_id,),
        ).fetchone()
        return self._tool_result(row) if row is not None else None

    def get_tool_result_by_call(
        self, context_id: str, tool_call_id: str,
    ) -> StoredToolResult | None:
        row = self.database.connection.execute(
            "SELECT * FROM context_tool_results WHERE context_id = ? AND tool_call_id = ?",
            (context_id, tool_call_id),
        ).fetchone()
        return self._tool_result(row) if row is not None else None

    def activate_checkpoint(
        self, checkpoint: CompactionCheckpoint,
    ) -> CompactionCheckpoint:
        payload = checkpoint.model_dump(mode="json")
        with self.database.connection:
            self.database.connection.execute(
                "INSERT INTO compaction_checkpoints(checkpoint_id, context_id, "
                "covers_through_message_id, recent_start_message_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    checkpoint.checkpoint_id,
                    checkpoint.context_id,
                    checkpoint.covers_through_message_id,
                    checkpoint.recent_start_message_id,
                    _json(payload),
                    checkpoint.created_at.isoformat(),
                ),
            )
            cursor = self.database.connection.execute(
                "UPDATE contexts SET active_checkpoint_id = ?, updated_at = ? "
                "WHERE context_id = ?",
                (
                    checkpoint.checkpoint_id,
                    checkpoint.created_at.isoformat(),
                    checkpoint.context_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown context: {checkpoint.context_id}")
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> CompactionCheckpoint | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM compaction_checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ).fetchone()
        return (
            CompactionCheckpoint.model_validate(json.loads(row[0]))
            if row is not None else None
        )


__all__ = ["SQLiteContextStore"]

"""SQLite persistence for private Conversations and their ordered history."""

from __future__ import annotations

from datetime import datetime
import json
import sqlite3
from uuid import uuid4

from specrag.models import (
    ContextMessage,
    Conversation,
    ConversationEvent,
    ConversationMessageContent,
    ConversationResearchLink,
    ConversationScopeRevision,
    ConversationTurn,
    ConversationTurnBudget,
)

from .database import SQLiteDatabase


def _json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SQLiteConversationStore:
    """Own atomic aggregate writes while reusing Context Harness tables."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    @staticmethod
    def _conversation(row: object) -> Conversation:
        return Conversation.model_validate_json(dict(row)["payload_json"])  # type: ignore[arg-type]

    @staticmethod
    def _message(row: object) -> ContextMessage:
        values = dict(row)  # type: ignore[arg-type]
        values["content"] = json.loads(values.pop("content_json"))
        return ContextMessage.model_validate(values)

    @staticmethod
    def _turn(row: object) -> ConversationTurn:
        return ConversationTurn.model_validate_json(dict(row)["payload_json"])  # type: ignore[arg-type]

    @staticmethod
    def _event(row: object) -> ConversationEvent:
        return ConversationEvent.model_validate_json(dict(row)["payload_json"])  # type: ignore[arg-type]

    def create(
        self,
        conversation: Conversation,
        event: ConversationEvent,
    ) -> Conversation:
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO contexts(context_id, workspace_id, owner_user_id, run_id, "
                "active_checkpoint_id, created_at, updated_at) "
                "VALUES (?, ?, ?, NULL, NULL, ?, ?)",
                (
                    conversation.context_id,
                    conversation.workspace_id,
                    conversation.owner_user_id,
                    conversation.created_at.isoformat(),
                    conversation.updated_at.isoformat(),
                ),
            )
            connection.execute(
                "INSERT INTO conversations(conversation_id, workspace_id, owner_user_id, "
                "context_id, status, revision, last_message_sequence, "
                "last_user_message_sequence, last_event_sequence, active_turn_id, "
                "active_scope_revision_id, created_at, updated_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
                (
                    conversation.conversation_id,
                    conversation.workspace_id,
                    conversation.owner_user_id,
                    conversation.context_id,
                    conversation.status,
                    conversation.revision,
                    conversation.last_message_sequence,
                    conversation.last_user_message_sequence,
                    conversation.active_turn_id,
                    conversation.active_scope_revision_id,
                    conversation.created_at.isoformat(),
                    conversation.updated_at.isoformat(),
                    _json(conversation),
                ),
            )
            self._insert_event(connection, event)
            connection.commit()
            return conversation
        except BaseException:
            connection.rollback()
            raise

    def get(self, conversation_id: str) -> Conversation | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM conversations WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        return self._conversation(row) if row is not None else None

    def list(
        self, workspace_id: str, owner_user_id: str, *, limit: int
    ) -> tuple[Conversation, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM conversations WHERE workspace_id=? "
            "AND owner_user_id=? ORDER BY updated_at DESC, conversation_id LIMIT ?",
            (workspace_id, owner_user_id, limit),
        ).fetchall()
        return tuple(self._conversation(row) for row in rows)

    def messages(self, conversation_id: str) -> tuple[ContextMessage, ...]:
        row = self.database.connection.execute(
            "SELECT context_id FROM conversations WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        rows = self.database.connection.execute(
            "SELECT * FROM context_messages WHERE context_id=? ORDER BY sequence",
            (row["context_id"],),
        ).fetchall()
        return tuple(self._message(item) for item in rows)

    def turns(self, conversation_id: str) -> tuple[ConversationTurn, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM conversation_turns WHERE conversation_id=? "
            "ORDER BY created_at, turn_id",
            (conversation_id,),
        ).fetchall()
        return tuple(self._turn(row) for row in rows)

    def events(
        self, conversation_id: str, *, since: int = 0, limit: int = 500
    ) -> tuple[ConversationEvent, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM conversation_events WHERE conversation_id=? "
            "AND sequence>? ORDER BY sequence LIMIT ?",
            (conversation_id, since, limit),
        ).fetchall()
        return tuple(self._event(row) for row in rows)

    def event_cursor(self, conversation_id: str) -> int:
        row = self.database.connection.execute(
            "SELECT last_event_sequence FROM conversations WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return int(row[0])

    def append_user_message(
        self,
        *,
        conversation_id: str,
        workspace_id: str,
        owner_user_id: str,
        text: str,
        token_estimate: int,
        message_id: str,
        turn_id: str,
        operation_key: str,
        request_sha256: str,
        expected_revision: int | None,
        budget: ConversationTurnBudget,
        now: datetime,
    ) -> tuple[Conversation, ContextMessage, ConversationTurn, int, int]:
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(conversation_id)
            current = self._conversation(row)
            if (
                current.workspace_id != workspace_id
                or current.owner_user_id != owner_user_id
            ):
                raise PermissionError("access denied")
            prior = connection.execute(
                "SELECT payload_json, request_sha256, trigger_message_id "
                "FROM conversation_turns WHERE conversation_id=? AND operation_key=?",
                (conversation_id, operation_key),
            ).fetchone()
            if prior is not None:
                if prior["request_sha256"] != request_sha256:
                    raise ValueError("operation key was reused with different input")
                turn = ConversationTurn.model_validate_json(prior["payload_json"])
                message_row = connection.execute(
                    "SELECT * FROM context_messages WHERE message_id=?",
                    (prior["trigger_message_id"],),
                ).fetchone()
                if message_row is None:
                    raise RuntimeError("idempotent Turn is missing its trigger message")
                event_row = connection.execute(
                    "SELECT sequence, payload_json FROM conversation_events "
                    "WHERE conversation_id=? "
                    "AND operation_key=?",
                    (conversation_id, operation_key + ":turn"),
                ).fetchone()
                if event_row is None:
                    raise RuntimeError("idempotent Turn is missing its queued event")
                connection.commit()
                queued_event = self._event(event_row)
                accepted_revision = int(queued_event.payload["revision"])
                return (
                    current,
                    self._message(message_row),
                    turn,
                    int(event_row["sequence"]),
                    accepted_revision,
                )
            if current.status != "active":
                raise ValueError("Conversation is archived")
            if expected_revision is not None and current.revision != expected_revision:
                raise ValueError("Conversation revision conflict")

            message_sequence = current.last_message_sequence + 1
            revision = current.revision + 1
            event_sequence = int(row["last_event_sequence"])
            event_sequence = self._skip_pending_research(
                connection, current, event_sequence, now
            )
            message = ContextMessage(
                message_id=message_id,
                context_id=current.context_id,
                sequence=message_sequence,
                role="user",
                content=ConversationMessageContent(text=text),
                token_estimate=token_estimate,
                created_at=now,
                turn_id=turn_id,
            )
            turn = ConversationTurn(
                turn_id=turn_id,
                conversation_id=conversation_id,
                trigger="user_message",
                trigger_message_id=message_id,
                submitted_user_sequence=message_sequence,
                budget=budget,
                operation_key=operation_key,
                created_at=now,
            )
            updated = current.model_copy(update={
                "revision": revision,
                "last_message_sequence": message_sequence,
                "last_user_message_sequence": message_sequence,
                "updated_at": now,
            })
            connection.execute(
                "INSERT INTO context_messages(message_id, context_id, sequence, role, "
                "turn_id, token_estimate, content_json, created_at) "
                "VALUES (?, ?, ?, 'user', ?, ?, ?, ?)",
                (
                    message.message_id,
                    message.context_id,
                    message.sequence,
                    turn_id,
                    token_estimate,
                    _json(message.content),
                    now.isoformat(),
                ),
            )
            connection.execute(
                "INSERT INTO conversation_turns(turn_id, conversation_id, status, "
                "operation_key, request_sha256, trigger_message_id, trigger_result_id, "
                "trigger_run_id, payload_json, created_at, updated_at) "
                "VALUES (?, ?, 'queued', ?, ?, ?, NULL, NULL, ?, ?, ?)",
                (
                    turn_id,
                    conversation_id,
                    operation_key,
                    request_sha256,
                    message_id,
                    _json(turn),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            connection.execute(
                "INSERT INTO conversation_turn_jobs(turn_id, state, available_at, "
                "lease_owner, lease_expires_at, attempts, created_at, updated_at) "
                "VALUES (?, 'queued', ?, NULL, NULL, 0, ?, ?)",
                (turn_id, now.isoformat(), now.isoformat(), now.isoformat()),
            )
            message_event = ConversationEvent(
                event_id="CEVT-" + message_id.removeprefix("MSG-"),
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                sequence=event_sequence + 1,
                event_type="message_appended",
                operation_key=operation_key + ":message",
                occurred_at=now,
                message_id=message_id,
                turn_id=turn_id,
                payload={"role": "user", "revision": revision},
            )
            turn_event = ConversationEvent(
                event_id="CEVT-" + turn_id.removeprefix("TURN-"),
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                sequence=event_sequence + 2,
                event_type="turn_queued",
                operation_key=operation_key + ":turn",
                occurred_at=now,
                message_id=message_id,
                turn_id=turn_id,
                payload={"status": "queued", "revision": revision},
            )
            self._insert_event(connection, message_event)
            self._insert_event(connection, turn_event)
            self._update_conversation(
                connection, updated, last_event_sequence=event_sequence + 2
            )
            connection.commit()
            return updated, message, turn, event_sequence + 2, revision
        except BaseException:
            connection.rollback()
            raise

    def change_status(
        self,
        *,
        conversation_id: str,
        workspace_id: str,
        owner_user_id: str,
        target: str,
        expected_revision: int | None,
        event_id: str,
        now: datetime,
    ) -> tuple[Conversation, int]:
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(conversation_id)
            current = self._conversation(row)
            if (
                current.workspace_id != workspace_id
                or current.owner_user_id != owner_user_id
            ):
                raise PermissionError("access denied")
            if expected_revision is not None and current.revision != expected_revision:
                raise ValueError("Conversation revision conflict")
            if current.status == target:
                connection.commit()
                return current, int(row["last_event_sequence"])
            if target not in {"active", "archived"}:
                raise ValueError("invalid Conversation status")
            if target == "archived" and current.active_turn_id is not None:
                raise ValueError("Conversation has an active Turn")
            revision = current.revision + 1
            event_sequence = int(row["last_event_sequence"]) + 1
            updated = current.model_copy(update={
                "status": target,
                "revision": revision,
                "updated_at": now,
            })
            event_name = (
                "conversation_archived" if target == "archived"
                else "conversation_unarchived"
            )
            event = ConversationEvent(
                event_id=event_id,
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                sequence=event_sequence,
                event_type=event_name,
                operation_key=f"status:{target}:{revision}",
                occurred_at=now,
                payload={"status": target, "revision": revision},
            )
            self._insert_event(connection, event)
            self._update_conversation(
                connection, updated, last_event_sequence=event_sequence
            )
            connection.commit()
            return updated, event_sequence
        except BaseException:
            connection.rollback()
            raise

    def get_scope_revision(
        self, scope_revision_id: str
    ) -> ConversationScopeRevision | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM conversation_scope_revisions "
            "WHERE scope_revision_id=?",
            (scope_revision_id,),
        ).fetchone()
        return (
            ConversationScopeRevision.model_validate_json(row[0])
            if row is not None else None
        )

    @staticmethod
    def _skip_pending_research(
        connection: sqlite3.Connection,
        conversation: Conversation,
        event_sequence: int,
        now: datetime,
    ) -> int:
        rows = connection.execute(
            "SELECT payload_json FROM conversation_research_links "
            "WHERE conversation_id=? AND presentation_status IN ('pending', 'queued') "
            "ORDER BY updated_at, link_id",
            (conversation.conversation_id,),
        ).fetchall()
        for row in rows:
            link = ConversationResearchLink.model_validate_json(row[0])
            if link.latest_result_id is None:
                continue
            if link.presentation_status == "queued":
                turn_row = connection.execute(
                    "SELECT payload_json FROM conversation_turns "
                    "WHERE conversation_id=? AND trigger_result_id=? "
                    "AND status IN ('queued', 'running')",
                    (conversation.conversation_id, link.latest_result_id),
                ).fetchone()
                if turn_row is not None:
                    turn = ConversationTurn.model_validate_json(turn_row[0])
                    cancelled = turn.model_copy(update={
                        "status": "cancelled",
                        "stop_reason": "superseded_by_user_message",
                        "finished_at": now,
                    })
                    connection.execute(
                        "UPDATE conversation_turns SET status='cancelled', "
                        "payload_json=?, updated_at=? WHERE turn_id=?",
                        (_json(cancelled), now.isoformat(), turn.turn_id),
                    )
                    connection.execute(
                        "DELETE FROM conversation_turn_jobs WHERE turn_id=?",
                        (turn.turn_id,),
                    )
            skipped = link.model_copy(update={
                "presentation_status": "skipped",
                "status_reason": "newer_user_message",
                "updated_at": now,
            })
            connection.execute(
                "UPDATE conversation_research_links SET presentation_status='skipped', "
                "status_reason='newer_user_message', payload_json=?, updated_at=? "
                "WHERE link_id=?",
                (_json(skipped), now.isoformat(), link.link_id),
            )
            event_sequence += 1
            event = ConversationEvent(
                event_id="CEVT-" + uuid4().hex.upper(),
                conversation_id=conversation.conversation_id,
                workspace_id=conversation.workspace_id,
                sequence=event_sequence,
                event_type="auto_reply_skipped",
                operation_key=f"auto-reply-skipped:{link.latest_result_id}",
                occurred_at=now,
                turn_id=link.origin_turn_id,
                call_id=link.tool_call_id,
                run_id=link.run_id,
                result_id=link.latest_result_id,
                payload={"reason": "newer_user_message"},
            )
            SQLiteConversationStore._insert_event(
                connection, event
            )
        return event_sequence

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection, event: ConversationEvent
    ) -> None:
        connection.execute(
            "INSERT INTO conversation_events(event_id, conversation_id, workspace_id, "
            "sequence, event_type, operation_key, payload_json, occurred_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.conversation_id,
                event.workspace_id,
                event.sequence,
                event.event_type,
                event.operation_key,
                _json(event),
                event.occurred_at.isoformat(),
            ),
        )

    @staticmethod
    def _update_conversation(
        connection: sqlite3.Connection,
        conversation: Conversation,
        *,
        last_event_sequence: int,
    ) -> None:
        cursor = connection.execute(
            "UPDATE conversations SET status=?, revision=?, last_message_sequence=?, "
            "last_user_message_sequence=?, last_event_sequence=?, active_turn_id=?, "
            "active_scope_revision_id=?, updated_at=?, payload_json=? "
            "WHERE conversation_id=?",
            (
                conversation.status,
                conversation.revision,
                conversation.last_message_sequence,
                conversation.last_user_message_sequence,
                last_event_sequence,
                conversation.active_turn_id,
                conversation.active_scope_revision_id,
                conversation.updated_at.isoformat(),
                _json(conversation),
                conversation.conversation_id,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(conversation.conversation_id)


__all__ = ["SQLiteConversationStore"]

"""Transactional persistence for Conversation Agent Turns and tool calls."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from uuid import uuid4

from specrag.models import (
    ContextMessage,
    Conversation,
    ConversationEvent,
    ConversationMessageContent,
    ConversationResearchLink,
    ConversationToolCall,
    ConversationTurn,
    ResearchResult,
)

from .database import SQLiteDatabase


def _json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SQLiteConversationAgentStore:
    """Owns compare-and-set writes which release or complete one Agent Turn."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def turn(self, turn_id: str) -> ConversationTurn | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM conversation_turns WHERE turn_id=?", (turn_id,)
        ).fetchone()
        return ConversationTurn.model_validate_json(row[0]) if row else None

    def tool_calls(self, turn_id: str) -> tuple[ConversationToolCall, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM conversation_tool_calls WHERE turn_id=? "
            "ORDER BY created_at, call_id", (turn_id,),
        ).fetchall()
        return tuple(ConversationToolCall.model_validate_json(row[0]) for row in rows)

    def start_turn(self, turn_id: str, *, now: datetime | None = None) -> ConversationTurn:
        now = now or datetime.now(timezone.utc)
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            turn_row = connection.execute(
                "SELECT payload_json FROM conversation_turns WHERE turn_id=?", (turn_id,)
            ).fetchone()
            if turn_row is None:
                raise KeyError(turn_id)
            turn = ConversationTurn.model_validate_json(turn_row[0])
            conversation_row = self._conversation_row(connection, turn.conversation_id)
            conversation = self._conversation(conversation_row)
            if turn.status == "running" and conversation.active_turn_id == turn_id:
                connection.commit()
                return turn
            if turn.status != "queued" or conversation.status != "active":
                raise ValueError("Conversation Turn is not startable")
            if conversation.active_turn_id not in {None, turn_id}:
                raise ValueError("Conversation already has an active Turn")
            if not self._membership(connection, conversation):
                raise PermissionError("Conversation owner no longer has access")
            started = turn.model_copy(update={"status": "running", "started_at": now})
            updated_conversation = conversation.model_copy(update={
                "active_turn_id": turn_id,
                "updated_at": now,
            })
            sequence = int(conversation_row["last_event_sequence"])
            sequence = self._event_once(
                connection, conversation, sequence,
                event_type="turn_started",
                operation_key=f"turn-started:{turn_id}",
                now=now, turn_id=turn_id,
                payload={"status": "running"},
            )
            self._put_turn(connection, started, now)
            self._put_conversation(connection, updated_conversation, sequence)
            connection.commit()
            return started
        except BaseException:
            connection.rollback()
            raise

    def start_tool_call(
        self,
        turn_id: str,
        *,
        call_id: str,
        tool_name: str,
        arguments: dict[str, object],
        operation_key: str,
        now: datetime | None = None,
    ) -> ConversationToolCall:
        now = now or datetime.now(timezone.utc)
        digest = hashlib.sha256(_json(arguments).encode("utf-8")).hexdigest()
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                "SELECT payload_json, arguments_sha256 FROM conversation_tool_calls "
                "WHERE conversation_id=(SELECT conversation_id FROM conversation_turns "
                "WHERE turn_id=?) AND operation_key=?", (turn_id, operation_key),
            ).fetchone()
            if existing is not None:
                if existing["arguments_sha256"] != digest:
                    raise ValueError("tool operation key was reused with different arguments")
                connection.commit()
                return ConversationToolCall.model_validate_json(existing[0])
            turn, conversation, conversation_row = self._running_turn(connection, turn_id)
            call = ConversationToolCall(
                call_id=call_id,
                conversation_id=turn.conversation_id,
                turn_id=turn_id,
                tool_name=tool_name,
                arguments_sha256=digest,
                arguments=arguments,
                status="running",
                operation_key=operation_key,
                created_at=now,
                started_at=now,
            )
            connection.execute(
                "INSERT INTO conversation_tool_calls(call_id, conversation_id, turn_id, "
                "tool_name, status, operation_key, arguments_sha256, tool_result_id, "
                "run_id, result_id, payload_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'running', ?, ?, NULL, NULL, NULL, ?, ?, ?)",
                (call.call_id, call.conversation_id, call.turn_id, call.tool_name,
                 operation_key, digest, _json(call), now.isoformat(), now.isoformat()),
            )
            sequence = self._event_once(
                connection, conversation, int(conversation_row["last_event_sequence"]),
                event_type="tool_call_started",
                operation_key=f"tool-call-started:{call_id}", now=now,
                turn_id=turn_id, call_id=call_id,
                payload={"tool_name": tool_name},
            )
            self._put_conversation(
                connection, conversation.model_copy(update={"updated_at": now}), sequence
            )
            connection.commit()
            return call
        except BaseException:
            connection.rollback()
            raise

    def finish_tool_call(
        self,
        call_id: str,
        *,
        tool_result_id: str,
        run_id: str | None,
        result_id: str | None,
        now: datetime | None = None,
    ) -> ConversationToolCall:
        return self._stop_tool_call(
            call_id, status="completed", tool_result_id=tool_result_id,
            run_id=run_id, result_id=result_id, safe_error=None, now=now,
        )

    def fail_tool_call(
        self, call_id: str, *, safe_error: str, now: datetime | None = None
    ) -> ConversationToolCall:
        return self._stop_tool_call(
            call_id, status="failed", tool_result_id=None,
            run_id=None, result_id=None, safe_error=safe_error, now=now,
        )

    def finish_turn(
        self,
        turn_id: str,
        message: ConversationMessageContent,
        *,
        stop_reason: str,
        token_estimate: int,
        message_id: str,
        now: datetime | None = None,
    ) -> tuple[ConversationTurn, ContextMessage]:
        now = now or datetime.now(timezone.utc)
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            turn, conversation, row = self._running_turn(connection, turn_id)
            if (
                turn.trigger == "user_message"
                and conversation.last_user_message_sequence != turn.submitted_user_sequence
            ):
                raise ValueError("Conversation Turn was superseded by a newer user message")
            self._validate_sources(connection, conversation, message)
            sequence = conversation.last_message_sequence + 1
            reply = ContextMessage(
                message_id=message_id,
                context_id=conversation.context_id,
                sequence=sequence,
                role="assistant",
                content=message,
                token_estimate=token_estimate,
                created_at=now,
                turn_id=turn_id,
            )
            completed = turn.model_copy(update={
                "status": "completed", "reply_message_id": message_id,
                "stop_reason": stop_reason, "finished_at": now,
            })
            connection.execute(
                "INSERT INTO context_messages(message_id, context_id, sequence, role, "
                "turn_id, token_estimate, content_json, created_at) "
                "VALUES (?, ?, ?, 'assistant', ?, ?, ?, ?)",
                (message_id, conversation.context_id, sequence, turn_id,
                 token_estimate, _json(message), now.isoformat()),
            )
            event_sequence = int(row["last_event_sequence"])
            event_sequence = self._event_once(
                connection, conversation, event_sequence,
                event_type="message_appended",
                operation_key=f"turn-reply:{turn_id}", now=now,
                message_id=message_id, turn_id=turn_id,
                payload={"role": "assistant", "source_kind": message.source_kind},
            )
            event_sequence = self._event_once(
                connection, conversation, event_sequence,
                event_type="turn_stopped",
                operation_key=f"turn-stopped:{turn_id}", now=now,
                message_id=message_id, turn_id=turn_id,
                payload={"status": "completed", "stop_reason": stop_reason},
            )
            event_sequence = self._mark_presented(
                connection, conversation, turn, message, message_id,
                event_sequence, now
            )
            updated = conversation.model_copy(update={
                "revision": conversation.revision + 1,
                "last_message_sequence": sequence,
                "active_turn_id": None,
                "updated_at": now,
            })
            self._put_turn(connection, completed, now)
            self._put_conversation(connection, updated, event_sequence)
            connection.commit()
            return completed, reply
        except BaseException:
            connection.rollback()
            raise

    def fail_turn(
        self,
        turn_id: str,
        *,
        failure_kind: str,
        safe_message: str,
        now: datetime | None = None,
    ) -> ConversationTurn:
        now = now or datetime.now(timezone.utc)
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT payload_json FROM conversation_turns WHERE turn_id=?", (turn_id,)
            ).fetchone()
            if row is None:
                raise KeyError(turn_id)
            turn = ConversationTurn.model_validate_json(row[0])
            if turn.status in {"completed", "failed", "cancelled"}:
                connection.commit()
                return turn
            conversation_row = self._conversation_row(connection, turn.conversation_id)
            conversation = self._conversation(conversation_row)
            started_at = turn.started_at or now
            failed = turn.model_copy(update={
                "status": "failed", "started_at": started_at,
                "finished_at": now, "failure_kind": failure_kind,
                "failure_message": safe_message,
            })
            sequence = self._event_once(
                connection, conversation, int(conversation_row["last_event_sequence"]),
                event_type="turn_stopped", operation_key=f"turn-stopped:{turn_id}",
                now=now, turn_id=turn_id,
                payload={"status": "failed", "failure_kind": failure_kind},
            )
            updated = conversation.model_copy(update={
                "active_turn_id": (
                    None if conversation.active_turn_id == turn_id
                    else conversation.active_turn_id
                ),
                "updated_at": now,
            })
            self._put_turn(connection, failed, now)
            self._put_conversation(connection, updated, sequence)
            connection.commit()
            return failed
        except BaseException:
            connection.rollback()
            raise

    def _stop_tool_call(
        self, call_id: str, *, status: str, tool_result_id: str | None,
        run_id: str | None, result_id: str | None, safe_error: str | None,
        now: datetime | None,
    ) -> ConversationToolCall:
        now = now or datetime.now(timezone.utc)
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT payload_json FROM conversation_tool_calls WHERE call_id=?",
                (call_id,),
            ).fetchone()
            if row is None:
                raise KeyError(call_id)
            call = ConversationToolCall.model_validate_json(row[0])
            if call.status in {"completed", "failed"}:
                connection.commit()
                return call
            updated = call.model_copy(update={
                "status": status, "tool_result_id": tool_result_id,
                "run_id": run_id, "result_id": result_id,
                "safe_error": safe_error, "finished_at": now,
            })
            connection.execute(
                "UPDATE conversation_tool_calls SET status=?, tool_result_id=?, "
                "run_id=?, result_id=?, payload_json=?, updated_at=? WHERE call_id=?",
                (status, tool_result_id, run_id, result_id, _json(updated),
                 now.isoformat(), call_id),
            )
            conversation_row = self._conversation_row(connection, call.conversation_id)
            conversation = self._conversation(conversation_row)
            event_type = "tool_call_completed" if status == "completed" else "tool_call_failed"
            sequence = self._event_once(
                connection, conversation, int(conversation_row["last_event_sequence"]),
                event_type=event_type,
                operation_key=f"tool-call-{status}:{call_id}", now=now,
                turn_id=call.turn_id, call_id=call_id, run_id=run_id,
                result_id=result_id,
                payload={"status": status, **({"safe_error": safe_error} if safe_error else {})},
            )
            self._put_conversation(
                connection, conversation.model_copy(update={"updated_at": now}), sequence
            )
            connection.commit()
            return updated
        except BaseException:
            connection.rollback()
            raise

    @staticmethod
    def _validate_sources(
        connection: sqlite3.Connection,
        conversation: Conversation,
        message: ConversationMessageContent,
    ) -> None:
        if message.source_kind == "direct_conversation":
            return
        evidence_ids: set[str] = set()
        for result_id in message.research_result_ids:
            row = connection.execute(
                "SELECT payload_json FROM research_results WHERE result_id=? "
                "AND workspace_id=? AND owner_user_id=? AND conversation_id=?",
                (result_id, conversation.workspace_id, conversation.owner_user_id,
                 conversation.conversation_id),
            ).fetchone()
            if row is None:
                raise PermissionError("Assistant message references an inaccessible Result")
            result = ResearchResult.model_validate_json(row[0])
            evidence_ids.update(result.evidence_ids)
        if set(message.citations) - evidence_ids:
            raise ValueError("Assistant message cites evidence outside its Results")

    @classmethod
    def _mark_presented(
        cls, connection: sqlite3.Connection, conversation: Conversation,
        turn: ConversationTurn, message: ConversationMessageContent,
        message_id: str, sequence: int, now: datetime,
    ) -> int:
        for result_id in message.research_result_ids:
            row = connection.execute(
                "SELECT payload_json FROM conversation_research_links "
                "WHERE conversation_id=? AND latest_result_id=?",
                (conversation.conversation_id, result_id),
            ).fetchone()
            if row is None:
                continue
            link = ConversationResearchLink.model_validate_json(row[0])
            if link.presentation_status not in {"pending", "queued"}:
                continue
            presented = link.model_copy(update={
                "presentation_status": "presented", "status_reason": None,
                "updated_at": now, "presented_at": now,
            })
            connection.execute(
                "UPDATE conversation_research_links SET presentation_status='presented', "
                "status_reason=NULL, payload_json=?, updated_at=?, presented_at=? "
                "WHERE link_id=?",
                (_json(presented), now.isoformat(), now.isoformat(), link.link_id),
            )
            sequence = cls._event_once(
                connection, conversation, sequence,
                event_type="auto_reply_presented",
                operation_key=f"auto-reply-presented:{result_id}", now=now,
                message_id=message_id, turn_id=turn.turn_id,
                call_id=link.tool_call_id, run_id=link.run_id,
                result_id=result_id, payload={"status": "presented"},
            )
        return sequence

    @staticmethod
    def _running_turn(
        connection: sqlite3.Connection, turn_id: str
    ) -> tuple[ConversationTurn, Conversation, sqlite3.Row]:
        row = connection.execute(
            "SELECT payload_json FROM conversation_turns WHERE turn_id=?", (turn_id,)
        ).fetchone()
        if row is None:
            raise KeyError(turn_id)
        turn = ConversationTurn.model_validate_json(row[0])
        conversation_row = SQLiteConversationAgentStore._conversation_row(
            connection, turn.conversation_id
        )
        conversation = SQLiteConversationAgentStore._conversation(conversation_row)
        if turn.status != "running" or conversation.active_turn_id != turn_id:
            raise ValueError("Conversation Turn is not active")
        return turn, conversation, conversation_row

    @staticmethod
    def _membership(connection: sqlite3.Connection, conversation: Conversation) -> bool:
        return connection.execute(
            "SELECT 1 FROM memberships WHERE workspace_id=? AND user_id=?",
            (conversation.workspace_id, conversation.owner_user_id),
        ).fetchone() is not None

    @staticmethod
    def _conversation_row(connection: sqlite3.Connection, conversation_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM conversations WHERE conversation_id=?", (conversation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return row

    @staticmethod
    def _conversation(row: sqlite3.Row) -> Conversation:
        return Conversation.model_validate_json(row["payload_json"])

    @staticmethod
    def _put_turn(connection: sqlite3.Connection, turn: ConversationTurn, now: datetime) -> None:
        connection.execute(
            "UPDATE conversation_turns SET status=?, payload_json=?, updated_at=? "
            "WHERE turn_id=?", (turn.status, _json(turn), now.isoformat(), turn.turn_id),
        )

    @staticmethod
    def _put_conversation(
        connection: sqlite3.Connection, conversation: Conversation, event_sequence: int
    ) -> None:
        connection.execute(
            "UPDATE conversations SET revision=?, last_message_sequence=?, "
            "last_user_message_sequence=?, last_event_sequence=?, active_turn_id=?, "
            "updated_at=?, payload_json=? WHERE conversation_id=?",
            (conversation.revision, conversation.last_message_sequence,
             conversation.last_user_message_sequence, event_sequence,
             conversation.active_turn_id, conversation.updated_at.isoformat(),
             _json(conversation), conversation.conversation_id),
        )

    @staticmethod
    def _event_once(
        connection: sqlite3.Connection, conversation: Conversation, sequence: int,
        *, event_type: str, operation_key: str, now: datetime,
        message_id: str | None = None, turn_id: str | None = None,
        call_id: str | None = None, run_id: str | None = None,
        result_id: str | None = None, payload: dict[str, object] | None = None,
    ) -> int:
        prior = connection.execute(
            "SELECT sequence FROM conversation_events WHERE conversation_id=? "
            "AND operation_key=?", (conversation.conversation_id, operation_key),
        ).fetchone()
        if prior is not None:
            return max(sequence, int(prior[0]))
        event = ConversationEvent(
            event_id="CEVT-" + uuid4().hex.upper(),
            conversation_id=conversation.conversation_id,
            workspace_id=conversation.workspace_id,
            sequence=sequence + 1,
            event_type=event_type,
            operation_key=operation_key,
            occurred_at=now,
            message_id=message_id,
            turn_id=turn_id,
            call_id=call_id,
            run_id=run_id,
            result_id=result_id,
            payload=payload or {},
        )
        connection.execute(
            "INSERT INTO conversation_events(event_id, conversation_id, workspace_id, "
            "sequence, event_type, operation_key, payload_json, occurred_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (event.event_id, event.conversation_id, event.workspace_id,
             event.sequence, event.event_type, event.operation_key,
             _json(event), now.isoformat()),
        )
        return event.sequence
__all__ = ["SQLiteConversationAgentStore"]

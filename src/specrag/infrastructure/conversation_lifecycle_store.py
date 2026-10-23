"""Atomic Conversation projection of terminal research work."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from uuid import uuid4

from specrag.models import (
    Conversation,
    ConversationEvent,
    ConversationResearchLink,
    ConversationTurn,
    ConversationTurnBudget,
    ResearchResult,
    ResearchRun,
    RunStatus,
)

from .database import SQLiteDatabase
from .result_store import SQLiteResearchResultStore


def _json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_suffix(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32].upper()


class ResearchLinkPending(RuntimeError):
    """A Conversation Run completed before its tool-call link was committed."""


class SQLiteConversationLifecycleStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def create_link(
        self, link: ConversationResearchLink
    ) -> ConversationResearchLink:
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                "SELECT payload_json FROM conversation_research_links WHERE run_id=?",
                (link.run_id,),
            ).fetchone()
            if existing is not None:
                stored = ConversationResearchLink.model_validate_json(existing[0])
                immutable = (
                    "link_id",
                    "conversation_id",
                    "origin_message_id",
                    "origin_turn_id",
                    "tool_call_id",
                    "run_id",
                    "submitted_user_sequence",
                    "created_at",
                )
                if any(
                    getattr(stored, field) != getattr(link, field)
                    for field in immutable
                ):
                    raise ValueError("Run is already linked to different Conversation input")
                connection.commit()
                return stored
            conversation_row = self._conversation_row(
                connection, link.conversation_id
            )
            conversation = self._conversation(conversation_row)
            run_row = connection.execute(
                "SELECT payload_json FROM runs WHERE run_id=?", (link.run_id,)
            ).fetchone()
            if run_row is None:
                raise KeyError(link.run_id)
            run = ResearchRun.model_validate_json(run_row[0])
            if (
                run.conversation_id != link.conversation_id
                or run.workspace_id != conversation.workspace_id
                or run.owner_user_id != conversation.owner_user_id
                or run.origin_message_id != link.origin_message_id
                or run.origin_turn_id != link.origin_turn_id
                or run.submitted_user_sequence != link.submitted_user_sequence
            ):
                raise ValueError("research link does not match frozen Run origin")
            self._insert_link(connection, link)
            event_sequence = int(conversation_row["last_event_sequence"])
            event_sequence = self._event_once(
                connection,
                conversation,
                event_sequence,
                event_type="run_linked",
                operation_key=f"run-linked:{link.run_id}",
                now=link.created_at,
                turn_id=link.origin_turn_id,
                call_id=link.tool_call_id,
                run_id=link.run_id,
                payload={"status": link.task_status},
            )
            self._update_conversation_events(
                connection, conversation, event_sequence, link.updated_at
            )
            connection.commit()
            return link
        except BaseException:
            connection.rollback()
            raise

    def get_by_run(self, run_id: str) -> ConversationResearchLink | None:
        row = self.database.connection.execute(
            "SELECT payload_json FROM conversation_research_links WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return ConversationResearchLink.model_validate_json(row[0]) if row else None

    def list_for_conversation(
        self, conversation_id: str
    ) -> tuple[ConversationResearchLink, ...]:
        rows = self.database.connection.execute(
            "SELECT payload_json FROM conversation_research_links "
            "WHERE conversation_id=? ORDER BY submitted_user_sequence, created_at",
            (conversation_id,),
        ).fetchall()
        return tuple(
            ConversationResearchLink.model_validate_json(row[0]) for row in rows
        )

    def reconcile(
        self,
        run: ResearchRun,
        run_attempt: int,
        task_status: RunStatus,
        result: ResearchResult | None,
        *,
        now: datetime | None = None,
    ) -> ConversationResearchLink | None:
        now = now or datetime.now(timezone.utc)
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            if result is not None:
                SQLiteResearchResultStore.insert(connection, result)
            if run.conversation_id is None:
                connection.commit()
                return None
            row = connection.execute(
                "SELECT payload_json FROM conversation_research_links WHERE run_id=?",
                (run.run_id,),
            ).fetchone()
            if row is None:
                raise ResearchLinkPending(run.run_id)
            link = ConversationResearchLink.model_validate_json(row[0])
            conversation_row = self._conversation_row(
                connection, run.conversation_id
            )
            conversation = self._conversation(conversation_row)
            self._validate_bindings(conversation, run, link, result)
            event_sequence = int(conversation_row["last_event_sequence"])
            event_sequence = self._event_once(
                connection,
                conversation,
                event_sequence,
                event_type="run_status_changed",
                operation_key=(
                    f"run-status:{run.run_id}:attempt:{run_attempt}:{task_status}"
                ),
                now=now,
                turn_id=link.origin_turn_id,
                call_id=link.tool_call_id,
                run_id=run.run_id,
                payload={"status": task_status, "run_attempt": run_attempt},
            )
            updated = link.model_copy(update={
                "task_status": task_status,
                "updated_at": now,
            })
            already_final = (
                result is not None
                and link.latest_result_id == result.result_id
                and link.presentation_status in {"presented", "skipped"}
            )
            if (
                result is not None
                and not already_final
                and self._is_newest_result(connection, link, result)
            ):
                updated = updated.model_copy(update={
                    "latest_result_id": result.result_id,
                    "presentation_status": "pending",
                    "status_reason": None,
                    "presented_at": None,
                })
                event_sequence = self._event_once(
                    connection,
                    conversation,
                    event_sequence,
                    event_type="result_available",
                    operation_key=f"result-available:{result.result_id}",
                    now=now,
                    turn_id=link.origin_turn_id,
                    call_id=link.tool_call_id,
                    run_id=run.run_id,
                    result_id=result.result_id,
                    payload={
                        "evidence_status": result.evidence_status,
                        "run_attempt": result.run_attempt,
                    },
                )
                updated, event_sequence = self._schedule_or_defer(
                    connection,
                    conversation,
                    updated,
                    result,
                    event_sequence,
                    now,
                )
            self._put_link(connection, updated)
            self._update_conversation_events(
                connection, conversation, event_sequence, now
            )
            connection.commit()
            return updated
        except BaseException:
            connection.rollback()
            raise

    def schedule_pending(
        self, conversation_id: str, *, now: datetime | None = None
    ) -> ConversationResearchLink | None:
        """Queue the oldest eligible pending Result after a Turn releases the slot."""

        now = now or datetime.now(timezone.utc)
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            conversation_row = self._conversation_row(connection, conversation_id)
            conversation = self._conversation(conversation_row)
            row = connection.execute(
                "SELECT payload_json FROM conversation_research_links "
                "WHERE conversation_id=? AND presentation_status='pending' "
                "ORDER BY updated_at, link_id LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            link = ConversationResearchLink.model_validate_json(row[0])
            result_row = connection.execute(
                "SELECT payload_json FROM research_results WHERE result_id=?",
                (link.latest_result_id,),
            ).fetchone()
            if result_row is None:
                raise RuntimeError("pending research link has no Result")
            result = ResearchResult.model_validate_json(result_row[0])
            event_sequence = int(conversation_row["last_event_sequence"])
            updated, event_sequence = self._schedule_or_defer(
                connection,
                conversation,
                link,
                result,
                event_sequence,
                now,
            )
            self._put_link(connection, updated)
            self._update_conversation_events(
                connection, conversation, event_sequence, now
            )
            connection.commit()
            return updated
        except BaseException:
            connection.rollback()
            raise

    def _schedule_or_defer(
        self,
        connection: sqlite3.Connection,
        conversation: Conversation,
        link: ConversationResearchLink,
        result: ResearchResult,
        event_sequence: int,
        now: datetime,
    ) -> tuple[ConversationResearchLink, int]:
        reason = self._ineligible_reason(connection, conversation, link)
        if reason is not None:
            skipped = link.model_copy(update={
                "presentation_status": "skipped",
                "status_reason": reason,
                "updated_at": now,
            })
            event_sequence = self._event_once(
                connection,
                conversation,
                event_sequence,
                event_type="auto_reply_skipped",
                operation_key=f"auto-reply-skipped:{result.result_id}",
                now=now,
                turn_id=link.origin_turn_id,
                call_id=link.tool_call_id,
                run_id=link.run_id,
                result_id=result.result_id,
                payload={"reason": reason},
            )
            return skipped, event_sequence
        origin_status = connection.execute(
            "SELECT status FROM conversation_turns WHERE turn_id=?",
            (link.origin_turn_id,),
        ).fetchone()
        presentation_inflight = connection.execute(
            "SELECT 1 FROM conversation_research_links WHERE conversation_id=? "
            "AND presentation_status='queued' AND link_id<>? LIMIT 1",
            (link.conversation_id, link.link_id),
        ).fetchone()
        if (
            conversation.active_turn_id is not None
            or origin_status is None
            or origin_status[0] in {"queued", "running"}
            or presentation_inflight is not None
        ):
            pending = link.model_copy(update={
                "presentation_status": "pending",
                "status_reason": "active_turn",
                "updated_at": now,
            })
            event_sequence = self._event_once(
                connection,
                conversation,
                event_sequence,
                event_type="auto_reply_deferred",
                operation_key=f"auto-reply-deferred:{result.result_id}",
                now=now,
                turn_id=link.origin_turn_id,
                call_id=link.tool_call_id,
                run_id=link.run_id,
                result_id=result.result_id,
                payload={"reason": "active_turn"},
            )
            return pending, event_sequence
        turn = self._presentation_turn(link, result, now)
        connection.execute(
            "INSERT OR IGNORE INTO conversation_turns(turn_id, conversation_id, "
            "status, operation_key, request_sha256, trigger_message_id, "
            "trigger_result_id, trigger_run_id, payload_json, created_at, updated_at) "
            "VALUES (?, ?, 'queued', ?, ?, NULL, ?, ?, ?, ?, ?)",
            (
                turn.turn_id,
                turn.conversation_id,
                turn.operation_key,
                hashlib.sha256(turn.operation_key.encode("utf-8")).hexdigest(),
                result.result_id,
                link.run_id,
                _json(turn),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        connection.execute(
            "INSERT OR IGNORE INTO conversation_turn_jobs(turn_id, state, "
            "available_at, lease_owner, lease_expires_at, attempts, created_at, "
            "updated_at) VALUES (?, 'queued', ?, NULL, NULL, 0, ?, ?)",
            (turn.turn_id, now.isoformat(), now.isoformat(), now.isoformat()),
        )
        queued = link.model_copy(update={
            "presentation_status": "queued",
            "status_reason": None,
            "updated_at": now,
        })
        event_sequence = self._event_once(
            connection,
            conversation,
            event_sequence,
            event_type="auto_reply_queued",
            operation_key=f"auto-reply-queued:{result.result_id}",
            now=now,
            turn_id=turn.turn_id,
            call_id=link.tool_call_id,
            run_id=link.run_id,
            result_id=result.result_id,
            payload={"status": "queued"},
        )
        return queued, event_sequence

    @staticmethod
    def _ineligible_reason(
        connection: sqlite3.Connection,
        conversation: Conversation,
        link: ConversationResearchLink,
    ) -> str | None:
        if conversation.status != "active":
            return "conversation_archived"
        if conversation.last_user_message_sequence != link.submitted_user_sequence:
            return "newer_user_message"
        membership = connection.execute(
            "SELECT 1 FROM memberships WHERE workspace_id=? AND user_id=?",
            (conversation.workspace_id, conversation.owner_user_id),
        ).fetchone()
        return None if membership is not None else "access_revoked"

    @staticmethod
    def _presentation_turn(
        link: ConversationResearchLink, result: ResearchResult, now: datetime
    ) -> ConversationTurn:
        operation_key = f"present-result:{link.conversation_id}:{result.result_id}"
        return ConversationTurn(
            turn_id="TURN-" + _stable_suffix(operation_key),
            conversation_id=link.conversation_id,
            trigger="research_result_ready",
            trigger_result_id=result.result_id,
            trigger_run_id=link.run_id,
            submitted_user_sequence=link.submitted_user_sequence,
            budget=ConversationTurnBudget(max_research_submissions=0),
            operation_key=operation_key,
            created_at=now,
        )

    @staticmethod
    def _validate_bindings(
        conversation: Conversation,
        run: ResearchRun,
        link: ConversationResearchLink,
        result: ResearchResult | None,
    ) -> None:
        if (
            conversation.conversation_id != run.conversation_id
            or conversation.workspace_id != run.workspace_id
            or conversation.owner_user_id != run.owner_user_id
            or link.conversation_id != conversation.conversation_id
            or link.origin_message_id != run.origin_message_id
            or link.origin_turn_id != run.origin_turn_id
            or link.submitted_user_sequence != run.submitted_user_sequence
        ):
            raise ValueError("completion binding does not match Conversation origin")
        if result is not None and (
            result.run_id != run.run_id
            or result.workspace_id != run.workspace_id
            or result.owner_user_id != run.owner_user_id
            or result.conversation_id != run.conversation_id
        ):
            raise ValueError("Result scope does not match its Run")

    @staticmethod
    def _is_newest_result(
        connection: sqlite3.Connection,
        link: ConversationResearchLink,
        result: ResearchResult,
    ) -> bool:
        if link.latest_result_id is None:
            return True
        row = connection.execute(
            "SELECT run_attempt FROM research_results WHERE result_id=?",
            (link.latest_result_id,),
        ).fetchone()
        return row is None or int(row[0]) <= result.run_attempt

    @staticmethod
    def _conversation_row(
        connection: sqlite3.Connection, conversation_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM conversations WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return row

    @staticmethod
    def _conversation(row: sqlite3.Row) -> Conversation:
        return Conversation.model_validate_json(row["payload_json"])

    @staticmethod
    def _insert_link(
        connection: sqlite3.Connection, link: ConversationResearchLink
    ) -> None:
        connection.execute(
            "INSERT INTO conversation_research_links(link_id, conversation_id, "
            "origin_message_id, origin_turn_id, tool_call_id, run_id, "
            "submitted_user_sequence, task_status, latest_result_id, "
            "presentation_status, status_reason, payload_json, created_at, updated_at, "
            "presented_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                link.link_id,
                link.conversation_id,
                link.origin_message_id,
                link.origin_turn_id,
                link.tool_call_id,
                link.run_id,
                link.submitted_user_sequence,
                link.task_status,
                link.latest_result_id,
                link.presentation_status,
                link.status_reason,
                _json(link),
                link.created_at.isoformat(),
                link.updated_at.isoformat(),
                link.presented_at.isoformat() if link.presented_at else None,
            ),
        )

    @classmethod
    def _put_link(
        cls, connection: sqlite3.Connection, link: ConversationResearchLink
    ) -> None:
        cursor = connection.execute(
            "UPDATE conversation_research_links SET task_status=?, "
            "latest_result_id=?, presentation_status=?, status_reason=?, "
            "payload_json=?, updated_at=?, presented_at=? WHERE link_id=?",
            (
                link.task_status,
                link.latest_result_id,
                link.presentation_status,
                link.status_reason,
                _json(link),
                link.updated_at.isoformat(),
                link.presented_at.isoformat() if link.presented_at else None,
                link.link_id,
            ),
        )
        if cursor.rowcount != 1:
            raise KeyError(link.link_id)

    @staticmethod
    def _event_once(
        connection: sqlite3.Connection,
        conversation: Conversation,
        sequence: int,
        *,
        event_type: str,
        operation_key: str,
        now: datetime,
        turn_id: str | None = None,
        call_id: str | None = None,
        run_id: str | None = None,
        result_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> int:
        existing = connection.execute(
            "SELECT sequence FROM conversation_events WHERE conversation_id=? "
            "AND operation_key=?",
            (conversation.conversation_id, operation_key),
        ).fetchone()
        if existing is not None:
            return max(sequence, int(existing[0]))
        event = ConversationEvent(
            event_id="CEVT-" + uuid4().hex.upper(),
            conversation_id=conversation.conversation_id,
            workspace_id=conversation.workspace_id,
            sequence=sequence + 1,
            event_type=event_type,
            operation_key=operation_key,
            occurred_at=now,
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
        return event.sequence

    @staticmethod
    def _update_conversation_events(
        connection: sqlite3.Connection,
        conversation: Conversation,
        event_sequence: int,
        now: datetime,
    ) -> None:
        updated = conversation.model_copy(update={"updated_at": now})
        connection.execute(
            "UPDATE conversations SET last_event_sequence=?, updated_at=?, "
            "payload_json=? WHERE conversation_id=?",
            (
                event_sequence,
                now.isoformat(),
                _json(updated),
                conversation.conversation_id,
            ),
        )


__all__ = ["ResearchLinkPending", "SQLiteConversationLifecycleStore"]

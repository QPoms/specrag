"""Atomic persistence for confirmed Conversation research scopes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

from specrag.models import Conversation, ConversationScopeRevision, SpecificationScope

from .database import SQLiteDatabase


def _json(value: object) -> str:
    return json.dumps(
        value.model_dump(mode="json") if hasattr(value, "model_dump") else value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class SQLiteConversationScopeStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    def activate(
        self,
        *,
        conversation_id: str,
        workspace_id: str,
        owner_user_id: str,
        objective: str,
        conditions: tuple[str, ...],
        scope: SpecificationScope,
        source_message_id: str,
        now: datetime | None = None,
    ) -> ConversationScopeRevision:
        now = now or datetime.now(timezone.utc)
        canonical = _json({
            "conditions": conditions,
            "conversation_id": conversation_id,
            "objective": objective,
            "scope": scope.model_dump(mode="json"),
            "source_message_id": source_message_id,
        })
        scope_id = "SCOPE-" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()[:32].upper()
        connection = self.database.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                "SELECT payload_json FROM conversation_scope_revisions "
                "WHERE scope_revision_id=?", (scope_id,),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return ConversationScopeRevision.model_validate_json(existing[0])
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(conversation_id)
            conversation = Conversation.model_validate_json(row["payload_json"])
            if (
                conversation.workspace_id != workspace_id
                or conversation.owner_user_id != owner_user_id
            ):
                raise PermissionError("Conversation scope mismatch")
            source = connection.execute(
                "SELECT 1 FROM context_messages WHERE context_id=? "
                "AND message_id=? AND role='user'",
                (conversation.context_id, source_message_id),
            ).fetchone()
            if source is None:
                raise ValueError("scope source message is unavailable")
            latest = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM conversation_scope_revisions "
                "WHERE conversation_id=?", (conversation_id,),
            ).fetchone()
            revision = ConversationScopeRevision(
                scope_revision_id=scope_id,
                conversation_id=conversation_id,
                revision=int(latest[0]) + 1,
                objective=objective,
                conditions=conditions,
                scope=scope,
                source_message_ids=(source_message_id,),
                predecessor_id=conversation.active_scope_revision_id,
                created_at=now,
            )
            connection.execute(
                "INSERT INTO conversation_scope_revisions(scope_revision_id, "
                "conversation_id, revision, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (scope_id, conversation_id, revision.revision, _json(revision),
                 now.isoformat()),
            )
            updated = conversation.model_copy(update={
                "active_scope_revision_id": scope_id, "updated_at": now,
            })
            connection.execute(
                "UPDATE conversations SET active_scope_revision_id=?, updated_at=?, "
                "payload_json=? WHERE conversation_id=?",
                (scope_id, now.isoformat(), _json(updated), conversation_id),
            )
            connection.commit()
            return revision
        except BaseException:
            connection.rollback()
            raise


__all__ = ["SQLiteConversationScopeStore"]

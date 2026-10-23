"""Authorized application service for the persistent Conversation aggregate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Protocol
from uuid import uuid4

from specrag.access import AccessPolicy, AuthorizationError, Permission
from specrag.context import ContextHarness
from specrag.models import (
    ActorContext,
    ContextMessage,
    Conversation,
    ConversationEvent,
    ConversationStatus,
    ConversationTurn,
    ConversationTurnBudget,
    ReconstructedContext,
    require_conversation_status_transition,
)


class ConversationStore(Protocol):
    def create(
        self, conversation: Conversation, event: ConversationEvent
    ) -> Conversation: ...
    def get(self, conversation_id: str) -> Conversation | None: ...
    def list(
        self, workspace_id: str, owner_user_id: str, *, limit: int
    ) -> tuple[Conversation, ...]: ...
    def messages(self, conversation_id: str) -> tuple[ContextMessage, ...]: ...
    def turns(self, conversation_id: str) -> tuple[ConversationTurn, ...]: ...
    def events(
        self, conversation_id: str, *, since: int = 0, limit: int = 500
    ) -> tuple[ConversationEvent, ...]: ...
    def event_cursor(self, conversation_id: str) -> int: ...
    def append_user_message(self, **kwargs: object) -> tuple[
        Conversation, ContextMessage, ConversationTurn, int, int
    ]: ...
    def change_status(self, **kwargs: object) -> tuple[Conversation, int]: ...


@dataclass(frozen=True)
class ConversationSnapshot:
    conversation: Conversation
    messages: tuple[ContextMessage, ...]
    event_cursor: int


@dataclass(frozen=True)
class AcceptedConversationMessage:
    message: ContextMessage
    turn: ConversationTurn
    conversation_revision: int
    event_cursor: int


class ConversationService:
    def __init__(
        self,
        store: ConversationStore,
        context_harness: ContextHarness,
        policy: AccessPolicy | None = None,
    ) -> None:
        self.store = store
        self.context_harness = context_harness
        self.policy = policy or AccessPolicy()

    def create(
        self, context: ActorContext, *, title: str | None = None
    ) -> ConversationSnapshot:
        self.policy.require(
            context, "conversation.create", workspace_id=context.workspace_id
        )
        if title is not None and not title.strip():
            raise ValueError("Conversation title must not be blank")
        now = datetime.now(timezone.utc)
        suffix = uuid4().hex.upper()
        conversation = Conversation(
            conversation_id="CONV-" + suffix,
            workspace_id=context.workspace_id,
            owner_user_id=context.user_id,
            context_id="CONTEXT-" + uuid4().hex.upper(),
            title=title.strip() if title is not None else None,
            created_at=now,
            updated_at=now,
        )
        event = ConversationEvent(
            event_id="CEVT-" + suffix,
            conversation_id=conversation.conversation_id,
            workspace_id=context.workspace_id,
            sequence=1,
            event_type="conversation_created",
            operation_key="conversation-created",
            occurred_at=now,
            payload={"revision": 0, "status": "active"},
        )
        self.store.create(conversation, event)
        return ConversationSnapshot(conversation, (), 1)

    def list(
        self, context: ActorContext, *, limit: int = 50
    ) -> tuple[Conversation, ...]:
        self.policy.require(
            context,
            "conversation.read",
            workspace_id=context.workspace_id,
            owner_user_id=context.user_id,
        )
        return self.store.list(context.workspace_id, context.user_id, limit=limit)

    def get(
        self, context: ActorContext, conversation_id: str
    ) -> ConversationSnapshot:
        conversation = self._authorized(
            context, conversation_id, "conversation.read"
        )
        return ConversationSnapshot(
            conversation=conversation,
            messages=self.store.messages(conversation_id),
            event_cursor=self.store.event_cursor(conversation_id),
        )

    def reconstruct(
        self, context: ActorContext, conversation_id: str
    ) -> ReconstructedContext:
        conversation = self._authorized(
            context, conversation_id, "conversation.read"
        )
        return self.context_harness.reconstruct(conversation.context_id)

    def validate_origin(
        self,
        context: ActorContext,
        *,
        conversation_id: str,
        message_id: str,
        turn_id: str,
        user_sequence: int,
    ) -> None:
        """Validate trusted MCP origin metadata against canonical history."""

        self._authorized(context, conversation_id, "conversation.read")
        message = next((
            item for item in self.store.messages(conversation_id)
            if item.message_id == message_id
        ), None)
        turn = next((
            item for item in self.store.turns(conversation_id)
            if item.turn_id == turn_id
        ), None)
        if (
            message is None
            or message.role != "user"
            or message.sequence != user_sequence
            or message.turn_id != turn_id
            or turn is None
            or turn.trigger != "user_message"
            or turn.trigger_message_id != message_id
            or turn.submitted_user_sequence != user_sequence
        ):
            raise AuthorizationError("invalid Conversation research origin")

    def append_user_message(
        self,
        context: ActorContext,
        conversation_id: str,
        *,
        text: str,
        operation_key: str,
        expected_revision: int | None = None,
        budget: ConversationTurnBudget | None = None,
    ) -> AcceptedConversationMessage:
        self._authorized(context, conversation_id, "conversation.write")
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("message text must not be blank")
        now = datetime.now(timezone.utc)
        try:
            (
                _conversation,
                message,
                turn,
                cursor,
                accepted_revision,
            ) = self.store.append_user_message(
                conversation_id=conversation_id,
                workspace_id=context.workspace_id,
                owner_user_id=context.user_id,
                text=normalized_text,
                token_estimate=self.context_harness.estimator.estimate(
                    {"text": normalized_text}
                ),
                message_id="MSG-" + uuid4().hex.upper(),
                turn_id="TURN-" + uuid4().hex.upper(),
                operation_key=operation_key,
                request_sha256=_request_digest(
                    normalized_text, expected_revision
                ),
                expected_revision=expected_revision,
                budget=budget or ConversationTurnBudget(),
                now=now,
            )
        except PermissionError as exc:
            raise AuthorizationError("access denied") from exc
        return AcceptedConversationMessage(
            message=message,
            turn=turn,
            conversation_revision=accepted_revision,
            event_cursor=cursor,
        )

    def archive(
        self,
        context: ActorContext,
        conversation_id: str,
        *,
        expected_revision: int | None = None,
    ) -> tuple[Conversation, int]:
        return self._change_status(
            context,
            conversation_id,
            target="archived",
            expected_revision=expected_revision,
        )

    def unarchive(
        self,
        context: ActorContext,
        conversation_id: str,
        *,
        expected_revision: int | None = None,
    ) -> tuple[Conversation, int]:
        return self._change_status(
            context,
            conversation_id,
            target="active",
            expected_revision=expected_revision,
        )

    def events(
        self,
        context: ActorContext,
        conversation_id: str,
        *,
        since: int = 0,
        limit: int = 500,
    ) -> tuple[ConversationEvent, ...]:
        self._authorized(context, conversation_id, "conversation.events.read")
        return self.store.events(conversation_id, since=since, limit=limit)

    def _change_status(
        self,
        context: ActorContext,
        conversation_id: str,
        *,
        target: ConversationStatus,
        expected_revision: int | None,
    ) -> tuple[Conversation, int]:
        current = self._authorized(
            context, conversation_id, "conversation.archive"
        )
        if current.status != target:
            require_conversation_status_transition(current.status, target)
        try:
            return self.store.change_status(
                conversation_id=conversation_id,
                workspace_id=context.workspace_id,
                owner_user_id=context.user_id,
                target=target,
                expected_revision=expected_revision,
                event_id="CEVT-" + uuid4().hex.upper(),
                now=datetime.now(timezone.utc),
            )
        except PermissionError as exc:
            raise AuthorizationError("access denied") from exc

    def _authorized(
        self,
        context: ActorContext,
        conversation_id: str,
        permission: Permission,
    ) -> Conversation:
        conversation = self.store.get(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        self.policy.require(
            context,
            permission,
            workspace_id=conversation.workspace_id,
            owner_user_id=conversation.owner_user_id,
        )
        return conversation


def _request_digest(text: str, expected_revision: int | None) -> str:
    payload = json.dumps(
        {"expected_revision": expected_revision, "text": text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AcceptedConversationMessage",
    "ConversationService",
    "ConversationSnapshot",
    "ConversationStore",
]

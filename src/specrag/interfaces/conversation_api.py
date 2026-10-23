"""Owner-private HTTP and SSE routes for persistent Conversations."""

from __future__ import annotations

import json
import time
from typing import Annotated, Iterator

from fastapi import Depends, FastAPI, Header, Query, Request, status
from fastapi.responses import StreamingResponse

from specrag.access import AuthorizationError
from specrag.models import (
    ActorContext,
    Conversation,
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationMessageAcceptedResponse,
    ConversationMessageCreateRequest,
    ConversationStatusRequest,
    ConversationStatusResponse,
)

from .api_errors import error, forbidden, unauthenticated
from .workspace_services import WorkspaceServices


def _services(request: Request) -> WorkspaceServices:
    return request.app.state.services


def _token(authorization: Annotated[str | None, Header()] = None) -> str:
    if authorization is None:
        raise unauthenticated()
    scheme, separator, token = authorization.partition(" ")
    if (
        not separator
        or scheme.casefold() != "bearer"
        or not token.strip()
        or len(token) > 4096
    ):
        raise unauthenticated()
    return token.strip()


def _workspace_context(
    workspace: str,
    token: Annotated[str, Depends(_token)],
    services: Annotated[WorkspaceServices, Depends(_services)],
) -> ActorContext:
    try:
        return services.identity.context(token, workspace)
    except AuthorizationError as exc:
        raise forbidden() from exc


Token = Annotated[str, Depends(_token)]
Services = Annotated[WorkspaceServices, Depends(_services)]
Context = Annotated[ActorContext, Depends(_workspace_context)]


def _not_found():
    return error(404, "conversation not found", code="conversation_not_found")


def install_conversation_routes(app: FastAPI) -> None:
    @app.post(
        "/api/v1/workspaces/{workspace}/conversations",
        response_model=ConversationDetailResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_conversation(
        payload: ConversationCreateRequest,
        context: Context,
        services: Services,
    ) -> ConversationDetailResponse:
        try:
            snapshot = services.conversations.create(context, title=payload.title)
        except AuthorizationError as exc:
            raise forbidden() from exc
        except ValueError as exc:
            raise error(422, str(exc), code="conversation_invalid") from exc
        return ConversationDetailResponse(
            conversation=snapshot.conversation,
            messages=snapshot.messages,
            turns=(),
            tool_calls=(),
            research_links=(),
            event_cursor=snapshot.event_cursor,
        )

    @app.get(
        "/api/v1/workspaces/{workspace}/conversations",
        response_model=tuple[Conversation, ...],
    )
    def list_conversations(
        context: Context,
        services: Services,
        limit: int = Query(default=50, ge=1, le=100),
    ) -> tuple[Conversation, ...]:
        try:
            return services.conversations.list(context, limit=limit)
        except AuthorizationError as exc:
            raise forbidden() from exc

    @app.get(
        "/api/v1/workspaces/{workspace}/conversations/{conversation_id}",
        response_model=ConversationDetailResponse,
    )
    def get_conversation(
        conversation_id: str,
        context: Context,
        services: Services,
    ) -> ConversationDetailResponse:
        try:
            snapshot = services.conversations.get(context, conversation_id)
        except (AuthorizationError, KeyError) as exc:
            raise _not_found() from exc
        return ConversationDetailResponse(
            conversation=snapshot.conversation,
            messages=snapshot.messages,
            turns=services.conversation_store.turns(conversation_id),
            tool_calls=tuple(
                call
                for turn in services.conversation_store.turns(conversation_id)
                for call in services.conversation_agent_store.tool_calls(turn.turn_id)
            ),
            research_links=(
                services.conversation_lifecycle_store.list_for_conversation(
                    conversation_id
                )
            ),
            event_cursor=snapshot.event_cursor,
        )

    @app.post(
        "/api/v1/workspaces/{workspace}/conversations/{conversation_id}/messages",
        response_model=ConversationMessageAcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def append_conversation_message(
        conversation_id: str,
        payload: ConversationMessageCreateRequest,
        context: Context,
        services: Services,
    ) -> ConversationMessageAcceptedResponse:
        try:
            accepted = services.conversations.append_user_message(
                context,
                conversation_id,
                text=payload.text,
                operation_key=payload.operation_key,
                expected_revision=payload.expected_revision,
            )
        except (AuthorizationError, KeyError) as exc:
            raise _not_found() from exc
        except ValueError as exc:
            code = (
                "conversation_archived"
                if "archived" in str(exc).casefold()
                else "conversation_conflict"
            )
            raise error(409, str(exc), code=code) from exc
        return ConversationMessageAcceptedResponse(
            message=accepted.message,
            turn=accepted.turn,
            conversation_revision=accepted.conversation_revision,
            event_cursor=accepted.event_cursor,
        )

    def change_status(
        conversation_id: str,
        payload: ConversationStatusRequest,
        context: ActorContext,
        services: WorkspaceServices,
        *,
        archived: bool,
    ) -> ConversationStatusResponse:
        try:
            operation = (
                services.conversations.archive
                if archived else services.conversations.unarchive
            )
            conversation, cursor = operation(
                context,
                conversation_id,
                expected_revision=payload.expected_revision,
            )
        except (AuthorizationError, KeyError) as exc:
            raise _not_found() from exc
        except ValueError as exc:
            raise error(409, str(exc), code="conversation_conflict") from exc
        return ConversationStatusResponse(
            conversation=conversation, event_cursor=cursor
        )

    @app.post(
        "/api/v1/workspaces/{workspace}/conversations/{conversation_id}/archive",
        response_model=ConversationStatusResponse,
    )
    def archive_conversation(
        conversation_id: str,
        payload: ConversationStatusRequest,
        context: Context,
        services: Services,
    ) -> ConversationStatusResponse:
        return change_status(
            conversation_id, payload, context, services, archived=True
        )

    @app.post(
        "/api/v1/workspaces/{workspace}/conversations/{conversation_id}/unarchive",
        response_model=ConversationStatusResponse,
    )
    def unarchive_conversation(
        conversation_id: str,
        payload: ConversationStatusRequest,
        context: Context,
        services: Services,
    ) -> ConversationStatusResponse:
        return change_status(
            conversation_id, payload, context, services, archived=False
        )

    @app.get(
        "/api/v1/workspaces/{workspace}/conversations/{conversation_id}/events",
        response_class=StreamingResponse,
    )
    def stream_conversation_events(
        conversation_id: str,
        workspace: str,
        context: Context,
        token: Token,
        services: Services,
        since: int = Query(default=0, ge=0),
        wait_seconds: float = Query(default=30.0, ge=0.1, le=300.0),
        last_event_id: Annotated[
            str | None, Header(alias="Last-Event-ID")
        ] = None,
    ) -> StreamingResponse:
        try:
            services.conversations.get(context, conversation_id)
        except (AuthorizationError, KeyError) as exc:
            raise _not_found() from exc
        sequence = since
        if last_event_id is not None:
            try:
                sequence = max(sequence, int(last_event_id))
            except ValueError as exc:
                raise error(
                    422,
                    "Last-Event-ID must be an integer",
                    code="validation_error",
                ) from exc

        def stream() -> Iterator[str]:
            cursor = sequence
            deadline = time.monotonic() + wait_seconds
            while True:
                try:
                    current_context = services.identity.context(token, workspace)
                    events = services.conversations.events(
                        current_context,
                        conversation_id,
                        since=cursor,
                    )
                except (AuthorizationError, KeyError):
                    return
                for event in events:
                    cursor = event.sequence
                    payload = json.dumps(
                        event.model_dump(mode="json"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    yield (
                        f"id: {event.sequence}\n"
                        f"event: {event.event_type}\n"
                        f"data: {payload}\n\n"
                    )
                if time.monotonic() >= deadline:
                    yield ": heartbeat\n\n"
                    return
                time.sleep(0.1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


__all__ = ["install_conversation_routes"]

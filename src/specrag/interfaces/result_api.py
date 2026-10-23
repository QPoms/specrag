"""Owner-private HTTP readback routes for durable Research Results."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, Header, Query, Request

from specrag.access import AuthorizationError
from specrag.models import ActorContext, ResearchResult

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


Services = Annotated[WorkspaceServices, Depends(_services)]
Context = Annotated[ActorContext, Depends(_workspace_context)]


def _result_not_found():
    return error(404, "research result not found", code="result_not_found")


def _conversation_not_found():
    return error(404, "conversation not found", code="conversation_not_found")


def install_result_routes(app: FastAPI) -> None:
    @app.get(
        "/api/v1/workspaces/{workspace}/research-results/{result_id}",
        response_model=ResearchResult,
    )
    def get_result(
        result_id: str,
        context: Context,
        services: Services,
    ) -> ResearchResult:
        try:
            return services.research_results.get(context, result_id)
        except (AuthorizationError, KeyError) as exc:
            raise _result_not_found() from exc

    @app.get(
        "/api/v1/workspaces/{workspace}/conversations/{conversation_id}/results",
        response_model=tuple[ResearchResult, ...],
    )
    def list_conversation_results(
        conversation_id: str,
        context: Context,
        services: Services,
        limit: int = Query(default=100, ge=1, le=200),
    ) -> tuple[ResearchResult, ...]:
        try:
            return services.research_results.list_for_conversation(
                context, conversation_id, limit=limit
            )
        except (AuthorizationError, KeyError) as exc:
            raise _conversation_not_found() from exc


__all__ = ["install_result_routes"]

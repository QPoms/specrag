"""Content-free Research Run lifecycle routes for workspace operators."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, Header, Query, Request

from specrag.access import AuthorizationError
from specrag.models import ActorContext, RunMetadataResponse

from .api_errors import forbidden, run_not_found, unauthenticated
from .api_inspection import metadata_response
from .workspace_services import WorkspaceServices


def _services(request: Request) -> WorkspaceServices:
    return request.app.state.services


def _token(authorization: Annotated[str | None, Header()] = None) -> str:
    if authorization is None:
        raise unauthenticated()
    scheme, separator, token = authorization.partition(" ")
    if (
        not separator or scheme.casefold() != "bearer"
        or not token.strip() or len(token) > 4096
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


def install_run_metadata_routes(app: FastAPI) -> None:
    @app.get(
        "/api/v1/workspaces/{workspace}/run-metadata",
        response_model=tuple[RunMetadataResponse, ...],
    )
    def list_run_metadata(
        context: Context,
        services: Services,
        limit: int = Query(default=50, ge=1, le=100),
    ) -> tuple[RunMetadataResponse, ...]:
        try:
            return tuple(
                metadata_response(run)
                for run in services.harness.list_run_metadata(context, limit=limit)
            )
        except AuthorizationError as exc:
            raise forbidden() from exc

    @app.get(
        "/api/v1/workspaces/{workspace}/runs/{run_id}/metadata",
        response_model=RunMetadataResponse,
    )
    def inspect_run_metadata(
        run_id: str, context: Context, services: Services,
    ) -> RunMetadataResponse:
        try:
            return metadata_response(
                services.harness.metadata_status(context, run_id)
            )
        except (AuthorizationError, KeyError) as exc:
            raise run_not_found() from exc


__all__ = ["install_run_metadata_routes"]

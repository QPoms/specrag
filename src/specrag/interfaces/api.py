"""Authenticated HTTP boundary for the existing workspace services."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import time
from typing import Annotated, AsyncIterator, Iterator

from fastapi import Depends, FastAPI, Header, Query, Request, status
from fastapi.responses import StreamingResponse
import uvicorn

from specrag.access import AuthorizationError
from specrag.models import (
    ActorContext,
    LoginRequest,
    LogoutResponse,
    MemberCreateRequest,
    MemberRemovedResponse,
    MemberRoleRequest,
    MemoryDashboardResponse,
    MemoryProposal,
    MemoryTopicResponse,
    Membership,
    ResearchRequest,
    ResearchBudget,
    ResearchRun,
    RunCreateRequest,
    RunBudgetRequest,
    RunInspectionResponse,
    RunMetadataResponse,
    RunRestartRequest,
    ReviewDecisionRequest,
    SessionResponse,
    SpecFactDashboardResponse,
    SpecFactProposal,
    Workspace,
    WorkspaceContextResponse,
    WorkspaceCreateRequest,
    WorkspaceMemberResponse,
    WorkspaceMembershipResponse,
)

from .workspace_services import WorkspaceServices
from .api_inspection import inspection_response, metadata_response
from .api_errors import error as _error
from .api_errors import forbidden as _forbidden
from .api_errors import install_error_handlers
from .api_errors import run_not_found as _run_not_found
from .api_errors import unauthenticated as _unauthenticated
from .conversation_api import install_conversation_routes
from .result_api import install_result_routes
from .run_metadata_api import install_run_metadata_routes
from .conversation_runtime import ApplicationWorkers
from specrag.orchestration import ConversationTurnWorker, ResearchLifecycleWorker, RunWorker


def _services(request: Request) -> WorkspaceServices:
    return request.app.state.services


def _worker(request: Request) -> RunWorker:
    return request.app.state.worker


def _token(authorization: Annotated[str | None, Header()] = None) -> str:
    if authorization is None:
        raise _unauthenticated()
    scheme, separator, token = authorization.partition(" ")
    if (
        not separator
        or scheme.casefold() != "bearer"
        or not token.strip()
        or len(token) > 4096
    ):
        raise _unauthenticated()
    return token.strip()


def _workspace_context(
    workspace: str,
    token: Annotated[str, Depends(_token)],
    services: Annotated[WorkspaceServices, Depends(_services)],
) -> ActorContext:
    try:
        return services.identity.context(token, workspace)
    except AuthorizationError as exc:
        raise _forbidden() from exc


Token = Annotated[str, Depends(_token)]
Services = Annotated[WorkspaceServices, Depends(_services)]
Worker = Annotated[RunWorker, Depends(_worker)]
Context = Annotated[ActorContext, Depends(_workspace_context)]


def create_app(
    services: WorkspaceServices | None = None,
    worker: RunWorker | None = None,
    lifecycle_worker: ResearchLifecycleWorker | None = None,
    conversation_worker: ConversationTurnWorker | None = None, *,
    worker_mode: str = "embedded",
) -> FastAPI:
    if worker_mode not in {"embedded", "external"}:
        raise ValueError("worker_mode must be 'embedded' or 'external'")
    owns_services = services is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services or WorkspaceServices.create()
        runtime = ApplicationWorkers.create(
            app.state.services, worker, lifecycle_worker, conversation_worker
        )
        app.state.worker = runtime.run
        app.state.lifecycle_worker = runtime.lifecycle
        app.state.conversation_worker = runtime.conversation
        if worker_mode == "embedded":
            runtime.start()
        try:
            yield
        finally:
            if worker_mode == "embedded":
                runtime.stop()
            if owns_services:
                app.state.services.close()

    app = FastAPI(
        title="SpecRAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    install_error_handlers(app)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/auth/login", response_model=SessionResponse)
    def login(payload: LoginRequest, services: Services) -> SessionResponse:
        try:
            token, expires_at = services.identity.login(
                payload.username, payload.password
            )
        except AuthorizationError as exc:
            raise _unauthenticated() from exc
        return SessionResponse(access_token=token, expires_at=expires_at)

    @app.post("/api/v1/auth/logout", response_model=LogoutResponse)
    def logout(token: Token, services: Services) -> LogoutResponse:
        try:
            services.identity.logout(token)
        except AuthorizationError as exc:
            raise _unauthenticated() from exc
        return LogoutResponse()

    @app.post(
        "/api/v1/workspaces",
        response_model=Workspace,
        status_code=status.HTTP_201_CREATED,
    )
    def create_workspace(
        payload: WorkspaceCreateRequest, token: Token, services: Services,
    ) -> Workspace:
        try:
            return services.identity.create_workspace(token, payload.name)
        except AuthorizationError as exc:
            raise _unauthenticated() from exc
        except ValueError as exc:
            raise _error(409, str(exc), code="workspace_conflict") from exc

    @app.get(
        "/api/v1/workspaces",
        response_model=tuple[WorkspaceMembershipResponse, ...],
    )
    def list_workspaces(
        token: Token, services: Services,
    ) -> tuple[WorkspaceMembershipResponse, ...]:
        try:
            records = services.identity.workspaces(token)
        except AuthorizationError as exc:
            raise _unauthenticated() from exc
        return tuple(
            WorkspaceMembershipResponse(workspace=workspace, membership=membership)
            for workspace, membership in records
        )

    @app.get(
        "/api/v1/workspaces/{workspace}/me",
        response_model=WorkspaceContextResponse,
    )
    def workspace_context(context: Context) -> WorkspaceContextResponse:
        return WorkspaceContextResponse(
            user_id=context.user_id,
            workspace_id=context.workspace_id,
            role=context.role,
            session_expires_at=context.expires_at,
        )

    install_conversation_routes(app)
    install_result_routes(app)
    install_run_metadata_routes(app)

    @app.get(
        "/api/v1/workspaces/{workspace}/members",
        response_model=tuple[WorkspaceMemberResponse, ...],
    )
    def list_members(
        context: Context, services: Services,
    ) -> tuple[WorkspaceMemberResponse, ...]:
        try:
            records = services.identity.members(context)
        except AuthorizationError as exc:
            raise _forbidden() from exc
        return tuple(
            WorkspaceMemberResponse(user=user, membership=membership)
            for user, membership in records
        )

    @app.post(
        "/api/v1/workspaces/{workspace}/members",
        response_model=Membership,
        status_code=status.HTTP_201_CREATED,
    )
    def add_member(
        payload: MemberCreateRequest, context: Context, services: Services,
    ) -> Membership:
        try:
            return services.identity.add_user(
                context, payload.username, payload.password, payload.role
            )
        except AuthorizationError as exc:
            raise _forbidden() from exc
        except ValueError as exc:
            raise _error(409, str(exc), code="member_conflict") from exc

    @app.patch(
        "/api/v1/workspaces/{workspace}/members/{username}",
        response_model=Membership,
    )
    def set_member_role(
        username: str,
        payload: MemberRoleRequest,
        context: Context,
        services: Services,
    ) -> Membership:
        try:
            return services.identity.set_role(context, username, payload.role)
        except AuthorizationError as exc:
            raise _forbidden() from exc
        except KeyError as exc:
            raise _error(404, "member not found", code="member_not_found") from exc
        except ValueError as exc:
            raise _error(409, str(exc), code="member_conflict") from exc

    @app.delete(
        "/api/v1/workspaces/{workspace}/members/{username}",
        response_model=MemberRemovedResponse,
    )
    def remove_member(
        username: str, context: Context, services: Services,
    ) -> MemberRemovedResponse:
        try:
            services.identity.remove_user(context, username)
        except AuthorizationError as exc:
            raise _forbidden() from exc
        except KeyError as exc:
            raise _error(404, "member not found", code="member_not_found") from exc
        except ValueError as exc:
            raise _error(409, str(exc), code="member_conflict") from exc
        return MemberRemovedResponse(username=username)

    @app.get(
        "/api/v1/workspaces/{workspace}/memory",
        response_model=MemoryDashboardResponse,
    )
    def memory_dashboard(
        context: Context, services: Services,
    ) -> MemoryDashboardResponse:
        try:
            topics = services.memory.list_topics(context)
            return MemoryDashboardResponse(
                user_core=services.memory.list(context),
                project_core=services.memory.list_project_core(context),
                topics=tuple(
                    MemoryTopicResponse(
                        topic=topic,
                        entries=services.memory.list_topic_entries(
                            context, topic.topic_id
                        ),
                    )
                    for topic in topics
                ),
                proposals=services.memory_proposals.list(context),
            )
        except AuthorizationError as exc:
            raise _forbidden() from exc

    @app.post(
        "/api/v1/workspaces/{workspace}/memory/proposals/{proposal_id}/review",
        response_model=MemoryProposal,
    )
    def review_memory_proposal(
        proposal_id: str,
        payload: ReviewDecisionRequest,
        context: Context,
        services: Services,
    ) -> MemoryProposal:
        try:
            if proposal_id not in {
                item.proposal_id for item in services.memory_proposals.list(context)
            }:
                raise KeyError(proposal_id)
            return services.memory_proposals.review(
                context,
                proposal_id,
                approve=payload.approve,
                reason=payload.reason,
            )
        except AuthorizationError as exc:
            raise _forbidden() from exc
        except KeyError as exc:
            raise _error(404, "proposal not found", code="proposal_not_found") from exc
        except ValueError as exc:
            raise _error(422, str(exc), code="review_invalid") from exc

    @app.get(
        "/api/v1/workspaces/{workspace}/specfacts",
        response_model=SpecFactDashboardResponse,
    )
    def specfact_dashboard(
        context: Context, services: Services,
    ) -> SpecFactDashboardResponse:
        try:
            return SpecFactDashboardResponse(
                facts=services.specfacts.list_facts(context),
                proposals=services.specfacts.list_proposals(context),
            )
        except AuthorizationError as exc:
            raise _forbidden() from exc

    @app.post(
        "/api/v1/workspaces/{workspace}/specfacts/proposals/{proposal_id}/review",
        response_model=SpecFactProposal,
    )
    def review_specfact_proposal(
        proposal_id: str,
        payload: ReviewDecisionRequest,
        context: Context,
        services: Services,
    ) -> SpecFactProposal:
        try:
            if proposal_id not in {
                item.proposal_id for item in services.specfacts.list_proposals(context)
            }:
                raise KeyError(proposal_id)
            return services.specfacts.review(
                context,
                proposal_id,
                approve=payload.approve,
                reason=payload.reason,
            )
        except AuthorizationError as exc:
            raise _forbidden() from exc
        except KeyError as exc:
            raise _error(404, "proposal not found", code="proposal_not_found") from exc
        except ValueError as exc:
            raise _error(422, str(exc), code="review_invalid") from exc

    @app.post(
        "/api/v1/workspaces/{workspace}/runs",
        response_model=RunInspectionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def submit_run(
        payload: RunCreateRequest,
        context: Context,
        worker: Worker,
    ) -> RunInspectionResponse:
        request = ResearchRequest(
            question=payload.question,
            release=payload.release,
            answer_language=payload.answer_language,
            detail_level=payload.detail_level,
            budget=ResearchBudget(
                max_iterations=payload.budget.max_iterations,
                max_tool_calls=payload.budget.max_tool_calls,
                max_gap_tool_calls=payload.budget.max_gap_tool_calls,
                max_replans=payload.budget.max_replans,
            ),
            elapsed_time_seconds=payload.elapsed_time_seconds,
            development_question_id=payload.development_question_id,
        )
        try:
            return inspection_response(worker.submit(context, request))
        except AuthorizationError as exc:
            raise _forbidden() from exc
        except ValueError as exc:
            raise _error(409, str(exc), code="run_submission_conflict") from exc

    @app.get(
        "/api/v1/workspaces/{workspace}/runs",
        response_model=tuple[ResearchRun, ...],
    )
    def list_runs(
        context: Context,
        services: Services,
        limit: int = Query(default=50, ge=1, le=100),
    ) -> tuple[ResearchRun, ...]:
        try:
            return services.harness.list_runs(context, limit=limit)
        except AuthorizationError as exc:
            raise _forbidden() from exc

    @app.get(
        "/api/v1/workspaces/{workspace}/runs/{run_id}",
        response_model=RunInspectionResponse,
    )
    def inspect_run(
        run_id: str, context: Context, services: Services,
    ) -> RunInspectionResponse:
        try:
            return inspection_response(services.harness.inspect(context, run_id))
        except (AuthorizationError, KeyError) as exc:
            raise _run_not_found() from exc

    @app.post(
        "/api/v1/workspaces/{workspace}/runs/{run_id}/cancel",
        response_model=RunMetadataResponse,
    )
    def cancel_run(
        run_id: str, context: Context, services: Services,
    ) -> RunMetadataResponse:
        try:
            return metadata_response(services.harness.cancel(context, run_id))
        except (AuthorizationError, KeyError) as exc:
            raise _run_not_found() from exc

    @app.post(
        "/api/v1/workspaces/{workspace}/runs/{run_id}/resume",
        response_model=RunInspectionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def resume_run(
        run_id: str,
        payload: RunRestartRequest,
        context: Context,
        worker: Worker,
    ) -> RunInspectionResponse:
        try:
            return inspection_response(worker.resume(
                context,
                run_id,
                elapsed_time_seconds=payload.elapsed_time_seconds,
            ))
        except (AuthorizationError, KeyError) as exc:
            raise _run_not_found() from exc
        except ValueError as exc:
            raise _error(409, str(exc), code="run_lifecycle_conflict") from exc

    @app.post(
        "/api/v1/workspaces/{workspace}/runs/{run_id}/retry",
        response_model=RunInspectionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def retry_run(
        run_id: str,
        payload: RunRestartRequest,
        context: Context,
        worker: Worker,
    ) -> RunInspectionResponse:
        try:
            return inspection_response(worker.retry(
                context,
                run_id,
                elapsed_time_seconds=payload.elapsed_time_seconds,
            ))
        except (AuthorizationError, KeyError) as exc:
            raise _run_not_found() from exc
        except ValueError as exc:
            raise _error(409, str(exc), code="run_lifecycle_conflict") from exc

    @app.get(
        "/api/v1/workspaces/{workspace}/runs/{run_id}/events",
        response_class=StreamingResponse,
    )
    def stream_run_events(
        run_id: str,
        workspace: str,
        context: Context,
        token: Token,
        services: Services,
        since: int = Query(default=0, ge=0),
        wait_seconds: float = Query(default=30.0, ge=0.1, le=300.0),
    ) -> StreamingResponse:
        try:
            services.harness.status(context, run_id)
        except (AuthorizationError, KeyError) as exc:
            raise _run_not_found() from exc

        def stream() -> Iterator[str]:
            sequence = since
            deadline = time.monotonic() + wait_seconds
            while True:
                try:
                    current_context = services.identity.context(token, workspace)
                    events = services.harness.events(current_context, run_id)
                except AuthorizationError:
                    return
                for event in events:
                    if event.sequence <= sequence:
                        continue
                    sequence = event.sequence
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
                try:
                    run = services.harness.status(current_context, run_id)
                except (AuthorizationError, KeyError):
                    return
                attempt_started_at = max(
                    (
                        event.sequence
                        for event in events
                        if event.event_type in {"run_started", "resumed"}
                    ),
                    default=0,
                )
                terminal_event_sent = any(
                    event.event_type == "run_stopped"
                    and attempt_started_at < event.sequence <= sequence
                    for event in events
                )
                if (
                    run.status in {"completed", "failed", "cancelled", "timed_out"}
                    and terminal_event_sent
                ):
                    return
                if time.monotonic() >= deadline:
                    return
                time.sleep(0.1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="specrag-api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--worker-mode",
        choices=("embedded", "external"),
        default="embedded",
        help="run the durable Worker in this process or separately",
    )
    arguments = parser.parse_args()
    uvicorn.run(
        create_app(worker_mode=arguments.worker_mode),
        host=arguments.host,
        port=arguments.port,
    )


__all__ = ["create_app", "main"]

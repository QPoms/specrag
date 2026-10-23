"""Transport-neutral application service behind the internal MCP tools."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Protocol

from specrag.models import (
    ActorContext,
    ConversationResearchLink,
    MCPResearchResultResponse,
    MCPResearchStartResponse,
    MCPResearchStatusResponse,
    ResearchRequest,
    ResearchToolRequest,
    ServiceDelegation,
    SpecificationScope,
)

from .conversations import ConversationService
from .harness import RunHarness
from .results import ResearchResultService
from .worker import RunWorker


class ResearchLinkWriter(Protocol):
    def create_link(
        self, link: ConversationResearchLink
    ) -> ConversationResearchLink: ...


class ConversationScopeWriter(Protocol):
    def activate(self, **kwargs: object): ...


class ResearchToolService:
    def __init__(
        self,
        harness: RunHarness,
        worker: RunWorker,
        results: ResearchResultService,
        conversations: ConversationService,
        links: ResearchLinkWriter,
        scope_writer: ConversationScopeWriter | None = None,
        *,
        wait_seconds: float = 10.0,
        poll_seconds: float = 0.05,
        default_release: int = 17,
    ) -> None:
        if wait_seconds < 0 or poll_seconds <= 0:
            raise ValueError("research-tool wait values are invalid")
        self.harness = harness
        self.worker = worker
        self.results = results
        self.conversations = conversations
        self.links = links
        self.scope_writer = scope_writer
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds
        self.default_release = default_release

    async def start_research(
        self,
        context: ActorContext,
        delegation: ServiceDelegation,
        request: ResearchToolRequest,
    ) -> MCPResearchStartResponse:
        self._validate_delegation(context, delegation)
        self.conversations.validate_origin(
            context,
            conversation_id=delegation.conversation_id,
            message_id=delegation.origin_message_id,
            turn_id=delegation.origin_turn_id,
            user_sequence=delegation.submitted_user_sequence,
        )
        for dependency in request.prior_results:
            stored = self.results.get(context, dependency.result_id)
            if stored.content_sha256 != dependency.content_sha256:
                raise ValueError("prior Research Result hash does not match")
        release = self._release(request)
        inspection = self.worker.submit(context, ResearchRequest(
            question=request.question,
            objective=request.objective,
            conditions=request.conditions,
            requested_scope=request.requested_scope,
            prior_results=request.prior_results,
            release=release,
            answer_language=request.answer_language,
            detail_level=request.detail_level,
            conversation_id=delegation.conversation_id,
            origin_message_id=delegation.origin_message_id,
            origin_turn_id=delegation.origin_turn_id,
            submitted_user_sequence=delegation.submitted_user_sequence,
            submission_operation_key=delegation.operation_key,
        ))
        run = inspection.run
        self.links.create_link(ConversationResearchLink(
            link_id="LINK-" + run.run_id.removeprefix("RUN-"),
            conversation_id=delegation.conversation_id,
            origin_message_id=delegation.origin_message_id,
            origin_turn_id=delegation.origin_turn_id,
            tool_call_id=delegation.tool_call_id,
            run_id=run.run_id,
            submitted_user_sequence=delegation.submitted_user_sequence,
            task_status=run.status,
            created_at=run.created_at,
            updated_at=run.created_at,
        ))
        if self.scope_writer is not None:
            self.scope_writer.activate(
                conversation_id=delegation.conversation_id,
                workspace_id=context.workspace_id,
                owner_user_id=context.user_id,
                objective=request.objective or request.question,
                conditions=request.conditions,
                scope=request.requested_scope or SpecificationScope(
                    policy="highest_version_within_release", release=release
                ),
                source_message_id=delegation.origin_message_id,
            )
        deadline = asyncio.get_running_loop().time() + self.wait_seconds
        while True:
            response = self._start_projection(context, run.run_id)
            if response.disposition != "accepted":
                return response
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return response
            await asyncio.sleep(min(self.poll_seconds, remaining))

    def get_research_status(
        self, context: ActorContext, run_id: str
    ) -> MCPResearchStatusResponse:
        run = self.harness.content_status(context, run_id)
        events = self.harness.events(context, run_id)
        result = self.results.for_run(context, run_id)
        latest = events[-1] if events else None
        return MCPResearchStatusResponse(
            run_id=run.run_id,
            status=run.status,
            stop_reason=run.stop_reason,
            result_id=result.result_id if result else None,
            failure_kind=run.failure_kind,
            failure_retryable=run.failure_retryable,
            last_event_sequence=latest.sequence if latest else 0,
            last_event_type=latest.event_type if latest else None,
        )

    def get_research_result(
        self,
        context: ActorContext,
        *,
        result_id: str | None = None,
        run_id: str | None = None,
    ) -> MCPResearchResultResponse:
        if (result_id is None) == (run_id is None):
            raise ValueError("provide exactly one Result ID or Run ID")
        result = (
            self.results.get(context, result_id)
            if result_id is not None
            else self.results.for_run(context, run_id)  # type: ignore[arg-type]
        )
        if result is None:
            raise KeyError(run_id)
        return MCPResearchResultResponse(result=result)

    def _start_projection(
        self, context: ActorContext, run_id: str
    ) -> MCPResearchStartResponse:
        run = self.harness.content_status(context, run_id)
        result = self.results.for_run(context, run_id)
        if result is not None:
            return MCPResearchStartResponse(
                disposition="completed",
                run_id=run_id,
                status="completed",
                result_id=result.result_id,
                result=result,
            )
        if run.status in {"failed", "cancelled", "timed_out"}:
            return MCPResearchStartResponse(
                disposition="terminal", run_id=run_id, status=run.status
            )
        return MCPResearchStartResponse(
            disposition="accepted", run_id=run_id, status=run.status
        )

    def _release(self, request: ResearchToolRequest) -> int:
        scope = request.requested_scope
        if scope is None or scope.policy == "latest_per_spec":
            return self.default_release
        if scope.release is not None:
            return scope.release
        return max(item.release for item in scope.versions)

    @staticmethod
    def _validate_delegation(
        context: ActorContext, delegation: ServiceDelegation
    ) -> None:
        now = datetime.now(timezone.utc)
        if (
            context.user_id != delegation.user_id
            or context.workspace_id != delegation.workspace_id
            or delegation.expires_at <= now
        ):
            raise PermissionError("delegation context mismatch")


__all__ = ["ConversationScopeWriter", "ResearchLinkWriter", "ResearchToolService"]

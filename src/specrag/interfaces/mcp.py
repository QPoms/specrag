"""Internal Streamable HTTP MCP facade for specification research."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import Context, MCPServer

from specrag.access import (
    AuthorizationError,
    ServiceDelegationSigner,
)
from specrag.infrastructure import DEFAULT_DATABASE_PATH
from specrag.models import (
    MCPResearchResultResponse,
    MCPResearchStartResponse,
    MCPResearchStatusResponse,
    PriorResultRef,
    ResearchToolRequest,
    SpecificationScope,
)
from specrag.orchestration import (
    ResearchLifecycleWorker,
    ResearchToolService,
    RunWorker,
)

from .workspace_services import WorkspaceServices


_DELEGATION_ENV = "SPECRAG_MCP_DELEGATION_SECRET"


def _bearer(ctx: Context[Any, Any]) -> str:
    authorization = (ctx.headers or {}).get("authorization")
    if authorization is None:
        raise AuthorizationError("service delegation required")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer" or not token.strip():
        raise AuthorizationError("service delegation required")
    return token.strip()


def create_mcp_server(
    services: WorkspaceServices,
    research: ResearchToolService,
    signer: ServiceDelegationSigner,
) -> MCPServer:
    server = MCPServer(
        "SpecRAG",
        description="Internal evidence-backed 3GPP specification research service.",
        version="0.1.0",
    )

    def trusted(ctx: Context[Any, Any]):
        delegation = signer.verify(_bearer(ctx))
        actor = services.identity.delegated_context(
            user_id=delegation.user_id,
            workspace_id=delegation.workspace_id,
            delegation_id=delegation.nonce,
            authenticated_at=delegation.issued_at,
            expires_at=delegation.expires_at,
        )
        return actor, delegation

    @server.tool(
        name="start_research",
        description=(
            "Create exactly one evidence-backed 3GPP Research Run. The server "
            "waits briefly for that same Run, otherwise returns its durable status."
        ),
        structured_output=True,
    )
    async def start_research(
        ctx: Context[Any, Any],
        question: str,
        objective: str | None = None,
        conditions: list[str] | None = None,
        requested_scope: SpecificationScope | None = None,
        prior_results: list[PriorResultRef] | None = None,
        answer_language: str = "English",
        detail_level: str = "standard",
    ) -> MCPResearchStartResponse:
        actor, delegation = trusted(ctx)
        return await research.start_research(
            actor,
            delegation,
            ResearchToolRequest(
                question=question,
                objective=objective,
                conditions=tuple(conditions or ()),
                requested_scope=requested_scope,
                prior_results=tuple(prior_results or ()),
                answer_language=answer_language,
                detail_level=detail_level,
            ),
        )

    @server.tool(
        name="get_research_status",
        description="Read the authorized lifecycle status of one Research Run.",
        structured_output=True,
    )
    def get_research_status(
        ctx: Context[Any, Any], run_id: str
    ) -> MCPResearchStatusResponse:
        actor, _ = trusted(ctx)
        return research.get_research_status(actor, run_id)

    @server.tool(
        name="get_research_result",
        description=(
            "Read one authorized durable Research Result by Result ID or Run ID."
        ),
        structured_output=True,
    )
    def get_research_result(
        ctx: Context[Any, Any],
        result_id: str | None = None,
        run_id: str | None = None,
    ) -> MCPResearchResultResponse:
        actor, _ = trusted(ctx)
        return research.get_research_result(
            actor, result_id=result_id, run_id=run_id
        )

    return server


class SpecRAGMCPClient:
    """Standard MCP Client wrapper used by the future Conversation Agent."""

    def __init__(
        self,
        url: str,
        *,
        http_transport: httpx2.AsyncBaseTransport | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("MCP timeout must be positive")
        self.url = url
        self.http_transport = http_transport
        self.timeout_seconds = timeout_seconds

    async def start_research(
        self, delegation_token: str, request: ResearchToolRequest
    ) -> MCPResearchStartResponse:
        payload = request.model_dump(mode="json", exclude_none=True)
        return MCPResearchStartResponse.model_validate(
            await self._call("start_research", payload, delegation_token)
        )

    async def get_research_status(
        self, delegation_token: str, run_id: str
    ) -> MCPResearchStatusResponse:
        return MCPResearchStatusResponse.model_validate(
            await self._call(
                "get_research_status", {"run_id": run_id}, delegation_token
            )
        )

    async def get_research_result(
        self,
        delegation_token: str,
        *,
        result_id: str | None = None,
        run_id: str | None = None,
    ) -> MCPResearchResultResponse:
        payload = {
            key: value
            for key, value in {"result_id": result_id, "run_id": run_id}.items()
            if value is not None
        }
        return MCPResearchResultResponse.model_validate(
            await self._call("get_research_result", payload, delegation_token)
        )

    async def _call(
        self, tool_name: str, arguments: dict[str, object], token: str
    ) -> Any:
        timeout = httpx2.Timeout(self.timeout_seconds)
        async with httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            transport=self.http_transport,
        ) as http_client:
            transport = streamable_http_client(
                self.url, http_client=http_client
            )
            async with Client(transport, raise_exceptions=True) as client:
                response = await client.call_tool(tool_name, arguments)
        if response.is_error or response.structured_content is None:
            raise RuntimeError(f"MCP tool failed: {tool_name}")
        return response.structured_content


def _secret_from_environment() -> bytes:
    value = os.environ.get(_DELEGATION_ENV)
    if value is None:
        raise RuntimeError(f"{_DELEGATION_ENV} is required")
    return value.encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="specrag-mcp")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    arguments = parser.parse_args()
    if arguments.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the first MCP release is restricted to a loopback host")
    services = WorkspaceServices.create(arguments.database)
    try:
        services.prewarm_research_runtime(17)
    except Exception:
        services.close()
        raise
    run_worker = RunWorker(services.harness, services.run_queue)
    lifecycle_worker = ResearchLifecycleWorker(
        services.runs,
        services.research_results,
        services.conversation_lifecycle_store,
        services.completion_queue,
    )
    research = ResearchToolService(
        services.harness,
        run_worker,
        services.research_results,
        services.conversations,
        services.conversation_lifecycle_store,
        services.conversation_scope_store,
    )
    server = create_mcp_server(
        services, research, ServiceDelegationSigner(_secret_from_environment())
    )
    run_worker.start()
    lifecycle_worker.start()
    try:
        server.run(
            transport="streamable-http",
            host=arguments.host,
            port=arguments.port,
            json_response=True,
            stateless_http=True,
        )
    finally:
        run_worker.stop()
        lifecycle_worker.stop()
        services.close()


__all__ = ["SpecRAGMCPClient", "create_mcp_server", "main"]


if __name__ == "__main__":
    main()

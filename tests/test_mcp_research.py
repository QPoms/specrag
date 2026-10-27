from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest

from specrag.access import AuthorizationError, ServiceDelegationSigner
from specrag.interfaces.mcp import SpecRAGMCPClient, create_mcp_server
from specrag.interfaces.mcp_runtime import MCPRuntimeState
from specrag.interfaces.workspace_services import WorkspaceServices
from specrag.models import (
    CorpusVersionRef,
    ResearchToolRequest,
    SpecificationScope,
)
from specrag.orchestration import (
    ResearchLifecycleWorker,
    ResearchToolService,
    RunHarness,
    RunWorker,
)


VERSION = CorpusVersionRef(specification="24.501", version="17.9.0")
SECRET = b"mcp-test-delegation-secret-32-bytes-minimum"


class _Workflow:
    def resume(self, *, thread_id):
        return {"stop_reason": "insufficient_evidence"}

    def checkpoint(self, *, thread_id):
        return SimpleNamespace(values={"thread_id": thread_id})


class _Application:
    workflow = _Workflow()

    def run(self, **kwargs):
        return {"stop_reason": "insufficient_evidence"}


def _factory(release, control, event_sink, research_context):
    return _Application()


def _setup(path: Path):
    services = WorkspaceServices.create(path)
    services.identity.bootstrap("alice", "correct horse battery")
    token, _ = services.identity.login("alice", "correct horse battery")
    workspace = services.identity.create_workspace(token, "MCP Workspace")
    actor = services.identity.context(token, workspace.workspace_id)
    conversation = services.conversations.create(actor).conversation
    accepted = services.conversations.append_user_message(
        actor,
        conversation.conversation_id,
        text="When shall the UE register?",
        operation_key="message-1",
        expected_revision=0,
    )
    harness = RunHarness(
        services.runs,
        _factory,
        corpus_snapshot=lambda _scope: (VERSION,),
        result_lookup=lambda run_id: (
            result.result_id
            if (result := services.result_store.latest_for_run(run_id))
            else None
        ),
    )
    run_worker = RunWorker(
        harness, services.run_queue, poll_seconds=0.01, retry_delay_seconds=0.01
    )
    lifecycle = ResearchLifecycleWorker(
        services.runs,
        services.research_results,
        services.conversation_lifecycle_store,
        services.completion_queue,
        poll_seconds=0.01,
        retry_delay_seconds=0.01,
    )
    return services, actor, conversation, accepted, harness, run_worker, lifecycle


def _delegation(signer, actor, conversation, accepted, *, call="A"):
    return signer.issue(
        actor,
        conversation_id=conversation.conversation_id,
        origin_message_id=accepted.message.message_id,
        origin_turn_id=accepted.turn.turn_id,
        submitted_user_sequence=accepted.message.sequence,
        tool_call_id="CALL-" + call * 32,
        operation_key=f"tool-call-{call}",
    )


def test_signed_delegation_is_tamper_evident_and_rechecks_membership(
    tmp_path: Path,
) -> None:
    services, actor, conversation, accepted, *_ = _setup(
        tmp_path / "delegation.sqlite3"
    )
    try:
        signer = ServiceDelegationSigner(SECRET)
        token = _delegation(signer, actor, conversation, accepted)
        claim = signer.verify(token)
        delegated = services.identity.delegated_context(
            user_id=claim.user_id,
            workspace_id=claim.workspace_id,
            delegation_id=claim.nonce,
            authenticated_at=claim.issued_at,
            expires_at=claim.expires_at,
        )
        assert delegated.user_id == actor.user_id
        assert delegated.role == actor.role
        with pytest.raises(AuthorizationError):
            signer.verify(token[:-1] + ("A" if token[-1] != "A" else "B"))

        expired_at = datetime.now(UTC) - timedelta(seconds=1)
        with pytest.raises(AuthorizationError):
            signer.verify(token, now=expired_at + timedelta(minutes=10))
        assert services.identity.store.remove_membership(
            actor.workspace_id, actor.user_id
        )
        with pytest.raises(AuthorizationError):
            services.identity.delegated_context(
                user_id=claim.user_id,
                workspace_id=claim.workspace_id,
                delegation_id=claim.nonce,
                authenticated_at=claim.issued_at,
                expires_at=claim.expires_at,
            )
    finally:
        services.close()


def test_same_tool_operation_returns_one_run_and_sync_result(tmp_path: Path) -> None:
    (
        services,
        actor,
        conversation,
        accepted,
        harness,
        run_worker,
        lifecycle,
    ) = _setup(tmp_path / "sync.sqlite3")
    signer = ServiceDelegationSigner(SECRET)
    claim = signer.verify(_delegation(signer, actor, conversation, accepted))
    service = ResearchToolService(
        harness,
        run_worker,
        services.research_results,
        services.conversations,
        services.conversation_lifecycle_store,
        services.conversation_scope_store,
        wait_seconds=2,
        poll_seconds=0.01,
    )
    run_worker.start()
    lifecycle.start()
    try:
        request = ResearchToolRequest(question="When shall the UE register?")
        first = asyncio.run(service.start_research(actor, claim, request))
        replay = asyncio.run(service.start_research(actor, claim, request))

        assert first.disposition == replay.disposition == "completed"
        assert first.run_id == replay.run_id
        assert first.result_id == replay.result_id
        assert first.result.evidence_status == "insufficient"
        assert len(services.runs.list_runs(
            actor.workspace_id, owner_user_id=actor.user_id, limit=10
        )) == 1
        assert services.run_queue.pending_count() == 0
        status = service.get_research_status(actor, first.run_id)
        assert status.status == "completed"
        assert status.result_id == first.result_id
        assert service.get_research_result(
            actor, run_id=first.run_id
        ).result.result_id == first.result_id
    finally:
        run_worker.stop()
        lifecycle.stop()
        services.close()


def test_wait_timeout_keeps_one_queued_run(tmp_path: Path) -> None:
    services, actor, conversation, accepted, harness, run_worker, _ = _setup(
        tmp_path / "async.sqlite3"
    )
    signer = ServiceDelegationSigner(SECRET)
    claim = signer.verify(_delegation(signer, actor, conversation, accepted))
    service = ResearchToolService(
        harness,
        run_worker,
        services.research_results,
        services.conversations,
        services.conversation_lifecycle_store,
        services.conversation_scope_store,
        wait_seconds=0.01,
        poll_seconds=0.005,
    )
    try:
        request = ResearchToolRequest(question="When shall the UE register?")
        first = asyncio.run(service.start_research(actor, claim, request))
        replay = asyncio.run(service.start_research(actor, claim, request))
        assert first.disposition == replay.disposition == "accepted"
        assert first.run_id == replay.run_id
        assert first.status == replay.status == "queued"
        assert services.run_queue.pending_count() == 1
        run = services.runs.get(first.run_id)
        assert run is not None
        assert run.input_snapshot is not None
        assert run.input_snapshot.requested_scope == SpecificationScope(
            policy="highest_version_within_release", release=17
        )
        assert run.deadline_at is None
    finally:
        services.close()


def test_research_service_rejects_an_explicit_non_release17_scope(
    tmp_path: Path,
) -> None:
    services, actor, conversation, accepted, harness, run_worker, _ = _setup(
        tmp_path / "release-policy.sqlite3"
    )
    signer = ServiceDelegationSigner(SECRET)
    claim = signer.verify(_delegation(signer, actor, conversation, accepted))
    service = ResearchToolService(
        harness,
        run_worker,
        services.research_results,
        services.conversations,
        services.conversation_lifecycle_store,
        wait_seconds=0,
    )
    try:
        request = ResearchToolRequest(
            question="Compare Release 18 behavior.",
            requested_scope=SpecificationScope(
                policy="highest_version_within_release", release=18
            ),
        )
        with pytest.raises(ValueError, match="only Release 17"):
            asyncio.run(service.start_research(actor, claim, request))
        assert services.run_queue.pending_count() == 0
    finally:
        services.close()


def test_streamable_http_tools_use_header_delegation_not_identity_arguments(
    tmp_path: Path,
) -> None:
    services, actor, conversation, accepted, harness, run_worker, _ = _setup(
        tmp_path / "transport.sqlite3"
    )
    signer = ServiceDelegationSigner(SECRET)
    token = _delegation(signer, actor, conversation, accepted)
    research = ResearchToolService(
        harness,
        run_worker,
        services.research_results,
        services.conversations,
        services.conversation_lifecycle_store,
        services.conversation_scope_store,
        wait_seconds=0,
    )
    server = create_mcp_server(services, research, signer)
    app = server.streamable_http_app(
        json_response=True,
        stateless_http=True,
        host="127.0.0.1",
    )

    async def exercise():
        transport = httpx2.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            client = SpecRAGMCPClient(
                "http://127.0.0.1:8000/mcp", http_transport=transport
            )
            response = await client.start_research(
                token, ResearchToolRequest(question="When shall the UE register?")
            )
            assert response.disposition == "accepted"
            assert response.status == "queued"
            status = await client.get_research_status(token, response.run_id)
            assert status.run_id == response.run_id
            with pytest.raises(Exception):
                await client.get_research_status("invalid", response.run_id)

    try:
        asyncio.run(exercise())
        tools = asyncio.run(server.list_tools())
        start = next(item for item in tools if item.name == "start_research")
        properties = start.input_schema["properties"]
        assert "user_id" not in properties
        assert "workspace_id" not in properties
        assert "role" not in properties
        assert set(properties) >= {"question", "conditions", "requested_scope"}
    finally:
        services.close()


def test_mcp_readiness_blocks_new_runs_until_runtime_is_ready(
    tmp_path: Path,
) -> None:
    services, actor, conversation, accepted, harness, run_worker, _ = _setup(
        tmp_path / "readiness.sqlite3"
    )
    signer = ServiceDelegationSigner(SECRET)
    token = _delegation(signer, actor, conversation, accepted)
    research = ResearchToolService(
        harness,
        run_worker,
        services.research_results,
        services.conversations,
        services.conversation_lifecycle_store,
        services.conversation_scope_store,
        wait_seconds=0,
    )
    readiness = MCPRuntimeState()
    readiness.begin_warmup()
    server = create_mcp_server(services, research, signer, readiness)
    app = server.streamable_http_app(
        json_response=True,
        stateless_http=True,
        host="127.0.0.1",
    )

    async def exercise() -> None:
        transport = httpx2.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            client = SpecRAGMCPClient(
                "http://127.0.0.1:8000/mcp", http_transport=transport
            )
            warming = await client.readiness()
            assert warming.status == "warming"
            with pytest.raises(Exception):
                await client.start_research(
                    token,
                    ResearchToolRequest(question="When shall the UE register?"),
                )
            assert services.run_queue.pending_count() == 0

            readiness.record_stage("corpus_snapshot", 12.5)
            readiness.mark_ready()
            ready = await client.readiness()
            assert ready.status == "ready"
            assert ready.stages[0].duration_ms == 12.5
            response = await client.start_research(
                token,
                ResearchToolRequest(question="When shall the UE register?"),
            )
            assert response.disposition == "accepted"

    try:
        asyncio.run(exercise())
    finally:
        services.close()


def test_mcp_request_limits_bound_conditions_and_client_timeout() -> None:
    with pytest.raises(ValueError):
        ResearchToolRequest(
            question="Q",
            conditions=tuple(f"condition-{index}" for index in range(51)),
        )
    with pytest.raises(ValueError):
        ResearchToolRequest(question="Q", conditions=("x" * 4001,))
    with pytest.raises(ValueError):
        SpecRAGMCPClient("http://127.0.0.1:8000/mcp", timeout_seconds=0)

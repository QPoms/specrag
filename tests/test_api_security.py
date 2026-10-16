from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from specrag.interfaces.api import create_app
from specrag.interfaces.api_inspection import inspection_response
from specrag.interfaces.workspace_services import WorkspaceServices
from specrag.models import (
    CandidateEvidence,
    EvidenceBackedClaim,
    EvidenceSynthesis,
    MemoryMutationDraft,
    ResearchRun,
    SpecFactProposal,
    SpecFactSource,
    ValidatedEvidence,
)
from specrag.orchestration import RunHarness, RunInspection


class _Workflow:
    def checkpoint(self, *, thread_id: str):
        return SimpleNamespace(values={"thread_id": thread_id})

    def resume(self, *, thread_id: str):
        return {"stop_reason": "completed"}


class _Application:
    workflow = _Workflow()

    def run(self, **kwargs):
        return {"stop_reason": "completed"}


def _authorization(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _login(client: TestClient, username: str, password: str) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def _services() -> tuple[WorkspaceServices, Path]:
    database_path = Path(".local/state") / f"api-test-{uuid4().hex}.sqlite3"
    services = WorkspaceServices.create(database_path)
    services.identity.bootstrap("alice", "alice secure password")
    services.harness = RunHarness(
        services.runs,
        lambda *args: _Application(),
    )
    return services, database_path


def _close(services: WorkspaceServices, database_path: Path) -> None:
    services.close()
    for suffix in ("", "-shm", "-wal"):
        Path(str(database_path) + suffix).unlink(missing_ok=True)


def test_api_authenticates_server_sessions_and_rechecks_membership() -> None:
    services, database_path = _services()
    try:
        with TestClient(create_app(services)) as client:
            assert client.post(
                "/api/v1/workspaces", json={"name": "denied"}
            ).status_code == 401
            assert client.post(
                "/api/v1/auth/login",
                json={"username": "alice", "password": "wrong password"},
            ).status_code == 401
            assert client.post(
                "/api/v1/auth/login",
                json={
                    "username": "alice",
                    "password": "alice secure password",
                    "role": "admin",
                },
            ).status_code == 422

            alice_token = _login(client, "alice", "alice secure password")
            created = client.post(
                "/api/v1/workspaces",
                headers=_authorization(alice_token),
                json={"name": "Workspace A"},
            )
            assert created.status_code == 201
            assert created.headers["cache-control"] == "no-store"
            workspace_id = created.json()["workspace_id"]

            added = client.post(
                f"/api/v1/workspaces/{workspace_id}/members",
                headers=_authorization(alice_token),
                json={
                    "username": "bob",
                    "password": "bob secure password",
                    "role": "member",
                },
            )
            assert added.status_code == 201
            bob_token = _login(client, "bob", "bob secure password")
            workspaces = client.get(
                "/api/v1/workspaces", headers=_authorization(bob_token)
            )
            assert workspaces.status_code == 200
            assert [item["workspace"]["workspace_id"] for item in workspaces.json()] == [
                workspace_id
            ]
            context = client.get(
                f"/api/v1/workspaces/{workspace_id}/me",
                headers=_authorization(bob_token),
            )
            assert context.status_code == 200
            assert context.json()["role"] == "member"
            assert "session_id" not in context.json()

            member_list = client.get(
                f"/api/v1/workspaces/{workspace_id}/members",
                headers=_authorization(alice_token),
            )
            assert member_list.status_code == 200
            assert [item["user"]["username"] for item in member_list.json()] == [
                "alice", "bob"
            ]
            assert client.get(
                f"/api/v1/workspaces/{workspace_id}/members",
                headers=_authorization(bob_token),
            ).status_code == 403
            assert client.patch(
                f"/api/v1/workspaces/{workspace_id}/members/alice",
                headers=_authorization(alice_token),
                json={"role": "member"},
            ).status_code == 409
            assert client.delete(
                f"/api/v1/workspaces/{workspace_id}/members/alice",
                headers=_authorization(alice_token),
            ).status_code == 409

            denied = client.post(
                f"/api/v1/workspaces/{workspace_id}/members",
                headers=_authorization(bob_token),
                json={
                    "username": "mallory",
                    "password": "mallory secure password",
                    "role": "admin",
                },
            )
            assert denied.status_code == 403

            removed = client.delete(
                f"/api/v1/workspaces/{workspace_id}/members/bob",
                headers=_authorization(alice_token),
            )
            assert removed.status_code == 200
            assert client.get(
                f"/api/v1/workspaces/{workspace_id}/me",
                headers=_authorization(bob_token),
            ).status_code == 403
            assert client.get(
                "/api/v1/workspaces", headers=_authorization(bob_token)
            ).json() == []

            assert client.post(
                "/api/v1/auth/logout", headers=_authorization(bob_token)
            ).status_code == 200
            assert client.post(
                "/api/v1/auth/logout", headers=_authorization(bob_token)
            ).status_code == 401
    finally:
        _close(services, database_path)


def test_api_hides_cross_workspace_and_cross_owner_runs() -> None:
    services, database_path = _services()
    try:
        with TestClient(create_app(services)) as client:
            alice_token = _login(client, "alice", "alice secure password")
            workspace_a = client.post(
                "/api/v1/workspaces",
                headers=_authorization(alice_token),
                json={"name": "Workspace A"},
            ).json()["workspace_id"]
            client.post(
                f"/api/v1/workspaces/{workspace_a}/members",
                headers=_authorization(alice_token),
                json={
                    "username": "carol",
                    "password": "carol secure password",
                    "role": "member",
                },
            )
            client.post(
                f"/api/v1/workspaces/{workspace_a}/members",
                headers=_authorization(alice_token),
                json={
                    "username": "dave",
                    "password": "dave secure password",
                    "role": "admin",
                },
            )
            client.post(
                f"/api/v1/workspaces/{workspace_a}/members",
                headers=_authorization(alice_token),
                json={
                    "username": "bob",
                    "password": "bob secure password",
                    "role": "member",
                },
            )
            carol_token = _login(client, "carol", "carol secure password")
            dave_token = _login(client, "dave", "dave secure password")
            bob_token = _login(client, "bob", "bob secure password")
            client.delete(
                f"/api/v1/workspaces/{workspace_a}/members/bob",
                headers=_authorization(alice_token),
            )
            workspace_b = client.post(
                "/api/v1/workspaces",
                headers=_authorization(bob_token),
                json={"name": "Workspace B"},
            ).json()["workspace_id"]

            alice = services.identity.context(alice_token, workspace_a)
            run = ResearchRun(
                run_id="RUN-" + "A" * 32,
                thread_id="THREAD-" + "B" * 32,
                workspace_id=workspace_a,
                owner_user_id=alice.user_id,
                question="What starts registration?",
                release=17,
                created_at=datetime.now(timezone.utc),
            )
            services.runs.put(run)

            path = f"/api/v1/workspaces/{workspace_a}/runs/{run.run_id}"
            assert client.get(
                path, headers=_authorization(alice_token)
            ).status_code == 200
            assert client.get(
                path, headers=_authorization(dave_token)
            ).status_code == 200

            alice_runs = client.get(
                f"/api/v1/workspaces/{workspace_a}/runs",
                headers=_authorization(alice_token),
            )
            admin_runs = client.get(
                f"/api/v1/workspaces/{workspace_a}/runs",
                headers=_authorization(dave_token),
            )
            member_runs = client.get(
                f"/api/v1/workspaces/{workspace_a}/runs",
                headers=_authorization(carol_token),
            )
            assert [item["run_id"] for item in alice_runs.json()] == [run.run_id]
            assert [item["run_id"] for item in admin_runs.json()] == [run.run_id]
            assert member_runs.json() == []

            denied = client.get(path, headers=_authorization(carol_token))
            unknown = client.get(
                f"/api/v1/workspaces/{workspace_a}/runs/RUN-{'F' * 32}",
                headers=_authorization(carol_token),
            )
            assert denied.status_code == unknown.status_code == 404
            assert denied.json() == unknown.json()
            denied_events = client.get(
                f"{path}/events", headers=_authorization(carol_token)
            )
            unknown_events = client.get(
                f"/api/v1/workspaces/{workspace_a}/runs/RUN-{'F' * 32}/events",
                headers=_authorization(carol_token),
            )
            assert denied_events.status_code == unknown_events.status_code == 404
            assert denied_events.json() == unknown_events.json()

            cross_workspace = client.get(
                f"/api/v1/workspaces/{workspace_b}/runs/{run.run_id}",
                headers=_authorization(bob_token),
            )
            assert cross_workspace.status_code == 404
            cancelled = client.post(
                f"{path}/cancel", headers=_authorization(dave_token)
            )
            assert cancelled.status_code == 200
            assert cancelled.json()["cancel_requested_at"] is not None
    finally:
        _close(services, database_path)


def test_api_submits_run_to_worker_and_streams_durable_events() -> None:
    services, database_path = _services()
    try:
        with TestClient(create_app(services)) as client:
            alice_token = _login(client, "alice", "alice secure password")
            workspace_id = client.post(
                "/api/v1/workspaces",
                headers=_authorization(alice_token),
                json={"name": "Workspace A"},
            ).json()["workspace_id"]
            submitted = client.post(
                f"/api/v1/workspaces/{workspace_id}/runs",
                headers=_authorization(alice_token),
                json={"question": "What starts registration?"},
            )
            assert submitted.status_code == 202
            run_id = submitted.json()["run"]["run_id"]
            assert submitted.json()["run"]["workspace_id"] == workspace_id

            stream = client.get(
                f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/events",
                headers=_authorization(alice_token),
                params={"wait_seconds": 5},
            )
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith("text/event-stream")
            assert "event: run_started" in stream.text
            assert "event: run_stopped" in stream.text
            inspection = client.get(
                f"/api/v1/workspaces/{workspace_id}/runs/{run_id}",
                headers=_authorization(alice_token),
            )
            assert inspection.status_code == 200
            assert inspection.json()["run"]["status"] == "completed"

            original = services.runs.get(run_id)
            assert original is not None
            services.runs.put(original.model_copy(update={
                "status": "failed",
                "stop_reason": "failed",
            }))
            resumed = client.post(
                f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/resume",
                headers=_authorization(alice_token),
                json={"elapsed_time_seconds": 300},
            )
            assert resumed.status_code == 202
            assert resumed.json()["run"]["run_id"] == run_id
            assert resumed.json()["run"]["status"] == "queued"
            resume_sequence = resumed.json()["events"][-1]["sequence"]
            resumed_stream = client.get(
                f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/events",
                headers=_authorization(alice_token),
                params={"since": resume_sequence, "wait_seconds": 5},
            )
            assert "event: run_stopped" in resumed_stream.text

            retried = client.post(
                f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/retry",
                headers=_authorization(alice_token),
                json={"elapsed_time_seconds": 300},
            )
            assert retried.status_code == 202
            assert retried.json()["run"]["run_id"] != run_id
            assert retried.json()["run"]["retry_of_run_id"] == run_id
    finally:
        _close(services, database_path)


def test_api_inspection_exposes_only_supported_claim_evidence() -> None:
    now = datetime.now(timezone.utc)
    run = ResearchRun(
        run_id="RUN-" + "A" * 32,
        thread_id="THREAD-" + "B" * 32,
        workspace_id="WS-1",
        owner_user_id="USR-1",
        question="What is the registration procedure?",
        release=17,
        created_at=now,
        started_at=now,
        finished_at=now,
        status="completed",
    )
    supported = CandidateEvidence(
        evidence_id="E:supported",
        task_id="task-1",
        chunk_id="chunk-1",
        specification="38.331",
        release=17,
        version="17.0.0",
        source_path="data/38.331.md",
        document_sha256="A" * 64,
        heading_path=("Registration",),
        line_start=10,
        line_end=12,
        excerpt="The UE sends a registration request.",
    )
    rejected = supported.model_copy(update={"evidence_id": "E:rejected"})
    synthesis = EvidenceSynthesis(
        answer="ignored",
        claims=(EvidenceBackedClaim(
            claim_id="claim-1",
            text="The UE sends a registration request.",
            evidence_ids=(supported.evidence_id,),
        ),),
    )
    value = RunInspection(
        run=run,
        events=(),
        checkpoint={
            "candidate_evidence": (supported, rejected),
            "validated_evidence": (
                ValidatedEvidence(
                    evidence_id=supported.evidence_id,
                    task_id="task-1",
                    source_valid=True,
                    supports_target=True,
                    validation_status="supported",
                    rationale="exact source match",
                ),
                ValidatedEvidence(
                    evidence_id=rejected.evidence_id,
                    task_id="task-1",
                    source_valid=True,
                    supports_target=False,
                    validation_status="rejected",
                    rationale="does not support the target",
                ),
            ),
            "synthesis": synthesis,
            "final_answer": synthesis.answer,
        },
        stage_latency_ms={},
        input_tokens=0,
        output_tokens=0,
    )

    response = inspection_response(value)

    assert response.answer == "The UE sends a registration request."
    assert [item.evidence_id for item in response.evidence] == [supported.evidence_id]
    assert response.claims[0].evidence_ids == (supported.evidence_id,)
    assert response.failure is None


def test_api_inspection_returns_redacted_failure_summary() -> None:
    now = datetime.now(timezone.utc)
    run = ResearchRun(
        run_id="RUN-" + "D" * 32,
        thread_id="THREAD-" + "E" * 32,
        workspace_id="WS-1",
        owner_user_id="USR-1",
        question="Why did this fail?",
        release=17,
        created_at=now,
        started_at=now,
        finished_at=now,
        status="failed",
        stop_reason="insufficient_evidence",
    )
    response = inspection_response(RunInspection(
        run=run,
        events=(),
        checkpoint=None,
        stage_latency_ms={"sufficiency_judge": 12.5},
        input_tokens=42,
        output_tokens=7,
    ))

    assert response.failure is not None
    assert response.failure.stop_reason == "insufficient_evidence"
    assert "evidence" in response.failure.summary
    assert response.checkpoint_available is False


def test_api_scopes_knowledge_and_enforces_proposal_review_roles() -> None:
    services, database_path = _services()
    try:
        with TestClient(create_app(services)) as client:
            alice_token = _login(client, "alice", "alice secure password")
            workspace_id = client.post(
                "/api/v1/workspaces",
                headers=_authorization(alice_token),
                json={"name": "Knowledge Workspace"},
            ).json()["workspace_id"]
            client.post(
                f"/api/v1/workspaces/{workspace_id}/members",
                headers=_authorization(alice_token),
                json={
                    "username": "bob",
                    "password": "bob secure password",
                    "role": "member",
                },
            )
            bob_token = _login(client, "bob", "bob secure password")
            alice = services.identity.context(alice_token, workspace_id)
            bob = services.identity.context(bob_token, workspace_id)

            alice_memory = services.memory.set(
                alice, kind="preference", key="answer-style", value="concise"
            )
            bob_memory = services.memory.set(
                bob, kind="preference", key="answer-style", value="detailed"
            )
            project_memory = services.memory.set_project_core(
                alice,
                kind="project_decision",
                key="target-release",
                content="Release 18",
            )
            memory_proposal = services.memory_proposals.propose(
                bob,
                MemoryMutationDraft(
                    operation="add",
                    scope="workspace",
                    kind="project_decision",
                    key="review-policy",
                    content="Every durable fact needs human review.",
                ),
            )
            specfact_proposal = SpecFactProposal(
                proposal_id="SFP-" + "A" * 24,
                operation_key="specfact:test:knowledge-api",
                workspace_id=workspace_id,
                run_id="RUN-" + "B" * 32,
                claim_id="claim-knowledge-api",
                assertion="Registration starts with a Registration Request.",
                subject="registration",
                predicate="starts with",
                object="Registration Request",
                sources=(SpecFactSource(
                    evidence_id="E:knowledge-api",
                    specification="24.501",
                    release=18,
                    version="18.1.0",
                    source_path="corpus/24.501.md",
                    document_sha256="C" * 64,
                    heading_path=("Registration",),
                    line_start=10,
                    line_end=12,
                ),),
                proposed_by=bob.user_id,
                proposed_at=datetime.now(timezone.utc),
            )
            services.specfacts.store.put_proposal(specfact_proposal)

            bob_memory_view = client.get(
                f"/api/v1/workspaces/{workspace_id}/memory",
                headers=_authorization(bob_token),
            )
            assert bob_memory_view.status_code == 200
            assert [item["memory_id"] for item in bob_memory_view.json()["user_core"]] == [
                bob_memory.memory_id
            ]
            assert alice_memory.memory_id not in str(bob_memory_view.json())
            assert bob_memory_view.json()["project_core"][0]["memory_id"] == (
                project_memory.memory_id
            )

            denied_memory_review = client.post(
                f"/api/v1/workspaces/{workspace_id}/memory/proposals/"
                f"{memory_proposal.proposal_id}/review",
                headers=_authorization(bob_token),
                json={"approve": True},
            )
            denied_fact_review = client.post(
                f"/api/v1/workspaces/{workspace_id}/specfacts/proposals/"
                f"{specfact_proposal.proposal_id}/review",
                headers=_authorization(bob_token),
                json={"approve": True},
            )
            assert denied_memory_review.status_code == 403
            assert denied_fact_review.status_code == 403

            approved_memory = client.post(
                f"/api/v1/workspaces/{workspace_id}/memory/proposals/"
                f"{memory_proposal.proposal_id}/review",
                headers=_authorization(alice_token),
                json={"approve": True},
            )
            approved_fact = client.post(
                f"/api/v1/workspaces/{workspace_id}/specfacts/proposals/"
                f"{specfact_proposal.proposal_id}/review",
                headers=_authorization(alice_token),
                json={"approve": True},
            )
            assert approved_memory.json()["status"] == "approved"
            assert approved_fact.json()["status"] == "approved"
            facts = client.get(
                f"/api/v1/workspaces/{workspace_id}/specfacts",
                headers=_authorization(bob_token),
            ).json()
            assert facts["facts"][0]["assertion"] == specfact_proposal.assertion

            other_workspace = client.post(
                "/api/v1/workspaces",
                headers=_authorization(bob_token),
                json={"name": "Other Workspace"},
            ).json()["workspace_id"]
            hidden = client.post(
                f"/api/v1/workspaces/{other_workspace}/specfacts/proposals/"
                f"{specfact_proposal.proposal_id}/review",
                headers=_authorization(bob_token),
                json={"approve": True},
            )
            assert hidden.status_code == 404
    finally:
        _close(services, database_path)

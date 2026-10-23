from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from specrag.interfaces.api import create_app
from specrag.interfaces.workspace_services import WorkspaceServices


def _authorization(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _login(client: TestClient, username: str, password: str) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_conversation_crud_idempotency_events_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "api.sqlite3"
    services = WorkspaceServices.create(path)
    services.identity.bootstrap("alice", "correct horse battery")
    with TestClient(create_app(services, worker_mode="external")) as client:
        token = _login(client, "alice", "correct horse battery")
        workspace = client.post(
            "/api/v1/workspaces",
            headers=_authorization(token),
            json={"name": "Conversation API"},
        ).json()
        workspace_id = workspace["workspace_id"]
        base = f"/api/v1/workspaces/{workspace_id}/conversations"
        created = client.post(
            base,
            headers=_authorization(token),
            json={"title": "Registration behavior"},
        )
        assert created.status_code == 201
        conversation_id = created.json()["conversation"]["conversation_id"]
        message_url = f"{base}/{conversation_id}/messages"
        payload = {
            "text": "What does the UE do?",
            "operation_key": "browser-message-1",
            "expected_revision": 0,
        }
        accepted = client.post(
            message_url, headers=_authorization(token), json=payload
        )
        replayed = client.post(
            message_url, headers=_authorization(token), json=payload
        )
        assert accepted.status_code == 202
        assert replayed.status_code == 202
        assert replayed.json() == accepted.json()
        turn_id = accepted.json()["turn"]["turn_id"]
        call_id = "CALL-" + "A" * 32
        services.conversation_agent_store.start_turn(turn_id)
        services.conversation_agent_store.start_tool_call(
            turn_id,
            call_id=call_id,
            tool_name="get_research_status",
            arguments={"run_id": "RUN-" + "B" * 32},
            operation_key="api-audit-call",
        )
        tool_page = services.context_harness.record_tool_result(
            created.json()["conversation"]["context_id"],
            call_id,
            {"status": "queued"},
        )
        services.conversation_agent_store.finish_tool_call(
            call_id,
            tool_result_id=tool_page.tool_result_id,
            run_id=None,
            result_id=None,
        )
        services.conversation_agent_store.fail_turn(
            turn_id,
            failure_kind="test_stop",
            safe_message="stopped by test",
        )

        stale = client.post(
            message_url,
            headers=_authorization(token),
            json={
                "text": "stale",
                "operation_key": "browser-message-stale",
                "expected_revision": 0,
            },
        )
        assert stale.status_code == 409
        assert stale.json()["code"] == "conversation_conflict"

        events = client.get(
            f"{base}/{conversation_id}/events",
            headers={**_authorization(token), "Last-Event-ID": "1"},
            params={"wait_seconds": 0.1},
        )
        assert events.status_code == 200
        assert "event: conversation_created" not in events.text
        assert "event: message_appended" in events.text
        assert "event: turn_queued" in events.text
        assert ": heartbeat" in events.text

        detail = client.get(
            f"{base}/{conversation_id}", headers=_authorization(token)
        )
        assert detail.status_code == 200
        assert [item["sequence"] for item in detail.json()["messages"]] == [1]
        assert detail.json()["tool_calls"][0]["call_id"] == call_id
        assert detail.json()["tool_calls"][0]["status"] == "completed"
    services.close()

    restored = WorkspaceServices.create(path)
    try:
        with TestClient(create_app(restored, worker_mode="external")) as client:
            detail = client.get(
                f"/api/v1/workspaces/{workspace_id}/conversations/{conversation_id}",
                headers=_authorization(token),
            )
            assert detail.status_code == 200
            assert detail.json()["messages"][0]["content"]["text"] == (
                "What does the UE do?"
            )
    finally:
        restored.close()


def test_admin_cannot_discover_members_private_conversation(tmp_path: Path) -> None:
    services = WorkspaceServices.create(tmp_path / "isolation.sqlite3")
    services.identity.bootstrap("alice", "correct horse battery")
    try:
        with TestClient(create_app(services, worker_mode="external")) as client:
            alice_token = _login(client, "alice", "correct horse battery")
            workspace_id = client.post(
                "/api/v1/workspaces",
                headers=_authorization(alice_token),
                json={"name": "Private Conversation API"},
            ).json()["workspace_id"]
            member = client.post(
                f"/api/v1/workspaces/{workspace_id}/members",
                headers=_authorization(alice_token),
                json={
                    "username": "bob",
                    "password": "bob secure password",
                    "role": "admin",
                },
            )
            assert member.status_code == 201
            base = f"/api/v1/workspaces/{workspace_id}/conversations"
            conversation_id = client.post(
                base,
                headers=_authorization(alice_token),
                json={},
            ).json()["conversation"]["conversation_id"]
            bob_token = _login(client, "bob", "bob secure password")

            hidden = client.get(
                f"{base}/{conversation_id}", headers=_authorization(bob_token)
            )
            assert hidden.status_code == 404
            assert hidden.json()["code"] == "conversation_not_found"
            listed = client.get(base, headers=_authorization(bob_token))
            assert listed.status_code == 200
            assert listed.json() == []
    finally:
        services.close()

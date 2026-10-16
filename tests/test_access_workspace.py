from __future__ import annotations

from datetime import timedelta

import pytest

from specrag.access import AuthorizationError, LocalIdentityService
from specrag.infrastructure import SQLiteDatabase, SQLiteIdentityStore


def _service(path, *, duration=timedelta(hours=12)):
    database = SQLiteDatabase(path)
    return database, LocalIdentityService(
        SQLiteIdentityStore(database), session_duration=duration
    )


def test_password_session_and_role_matrix(tmp_path) -> None:
    database, service = _service(tmp_path / "state.sqlite3")
    try:
        service.bootstrap("alice", "correct horse battery")
        with pytest.raises(AuthorizationError):
            service.login("alice", "wrong password")
        token, _ = service.login("alice", "correct horse battery")
        workspace = service.create_workspace(token, "Workspace A")
        admin = service.context(token, workspace.workspace_id)
        membership = service.add_user(
            admin, "bob", "bob secure password", "reviewer"
        )
        assert membership.role == "reviewer"
        assert [user.username for user, _ in service.members(admin)] == [
            "alice", "bob"
        ]
        with pytest.raises(ValueError, match="cannot demote"):
            service.set_role(admin, "alice", "member")
        with pytest.raises(ValueError, match="cannot remove"):
            service.remove_user(admin, "alice")
        bob_token, _ = service.login("bob", "bob secure password")
        bob = service.context(bob_token, "Workspace A")
        assert bob.role == "reviewer"
        with pytest.raises(AuthorizationError):
            service.members(bob)
        service.logout(bob_token)
        with pytest.raises(AuthorizationError):
            service.context(bob_token, "Workspace A")
    finally:
        database.close()


def test_expired_session_and_cross_workspace_default_deny(tmp_path) -> None:
    database, service = _service(
        tmp_path / "state.sqlite3", duration=timedelta(microseconds=1)
    )
    try:
        service.bootstrap("alice", "correct horse battery")
        token, _ = service.login("alice", "correct horse battery")
        with pytest.raises(AuthorizationError):
            service.create_workspace(token, "expired")
    finally:
        database.close()

    database, service = _service(tmp_path / "separate.sqlite3")
    try:
        service.bootstrap("alice", "correct horse battery")
        alice_token, _ = service.login("alice", "correct horse battery")
        workspace_a = service.create_workspace(alice_token, "Workspace A")
        alice = service.context(alice_token, workspace_a.workspace_id)
        service.add_user(alice, "bob", "bob secure password", "member")
        service.remove_user(alice, "bob")
        bob_token, _ = service.login("bob", "bob secure password")
        workspace_b = service.create_workspace(bob_token, "Workspace B")
        with pytest.raises(AuthorizationError):
            service.context(alice_token, workspace_b.workspace_id)
        with pytest.raises(AuthorizationError):
            service.context(bob_token, workspace_a.workspace_id)
    finally:
        database.close()

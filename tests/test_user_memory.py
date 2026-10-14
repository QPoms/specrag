from datetime import datetime, timedelta, timezone

import pytest

from specrag.access import AuthorizationError
from specrag.infrastructure import InMemoryUserMemoryStore
from specrag.memory import UserMemoryService, resolve_user_configuration
from specrag.models import ActorContext


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _context(user="alice", workspace="WA", role="member"):
    return ActorContext(
        user_id=user, workspace_id=workspace, role=role, session_id="session",
        authenticated_at=NOW, expires_at=NOW + timedelta(hours=12),
    )


def test_user_memory_is_private_and_workspace_scoped() -> None:
    service = UserMemoryService(InMemoryUserMemoryStore())
    item = service.set(
        _context(), kind="answer_language", key="default", value="Chinese", now=NOW
    )
    assert service.list(_context()) == (item,)
    assert service.list(_context("bob")) == ()
    assert service.list(_context(workspace="WB")) == ()
    with pytest.raises(AuthorizationError):
        service.delete(_context("bob"), item.memory_id)


def test_memory_update_creates_immutable_revision() -> None:
    store = InMemoryUserMemoryStore()
    service = UserMemoryService(store)
    first = service.set(
        _context(), kind="detail_level", key="default", value="short", now=NOW
    )
    second = service.set(
        _context(), kind="detail_level", key="default", value="detailed",
        now=NOW + timedelta(minutes=1),
    )
    assert first.memory_id != second.memory_id
    assert next(item for item in store.all() if item.memory_id == first.memory_id).status == "deleted"
    assert service.list(_context()) == (second,)


def test_configuration_precedence_is_explicit_then_memory_then_config() -> None:
    service = UserMemoryService(InMemoryUserMemoryStore())
    release = service.set(
        _context(), kind="default_release", key="default", value="18", now=NOW
    )
    language = service.set(
        _context(), kind="answer_language", key="default", value="Chinese", now=NOW
    )
    resolved = resolve_user_configuration(
        (release, language), config_release=17, explicit_release=19
    )
    assert resolved.release == 19
    assert resolved.answer_language == "Chinese"
    assert resolved.detail_level == "standard"

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
    assert service.list(_context("admin", role="admin")) == ()
    with pytest.raises(AuthorizationError):
        service.delete(_context("admin", role="admin"), item.memory_id)


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


def test_memory_values_are_typed_and_writes_are_explicit() -> None:
    service = UserMemoryService(InMemoryUserMemoryStore())
    with pytest.raises(ValueError, match="integer"):
        service.set(_context(), kind="default_release", key="default", value="R17", now=NOW)
    with pytest.raises(ValueError, match="detail_level"):
        service.set(_context(), kind="detail_level", key="default", value="verbose", now=NOW)
    with pytest.raises(ValueError, match="explicit_user"):
        service.set(
            _context(), kind="research_lead", key="one", value="topic",
            origin="model_inferred", now=NOW,
        )


def test_memory_revision_records_predecessor_and_operation_key() -> None:
    service = UserMemoryService(InMemoryUserMemoryStore())
    first = service.set(
        _context(), kind="followed_topic", key="topic", value="RAN", now=NOW,
        operation_key="memory:set:first",
    )
    second = service.set(
        _context(), kind="followed_topic", key="topic", value="Core", now=NOW + timedelta(minutes=1),
        operation_key="memory:set:second",
    )
    assert first.revision == 1
    assert second.revision == 2
    assert second.supersedes_id == first.memory_id
    assert second.operation_key == "memory:set:second"
    assert service.set(
        _context(), kind="followed_topic", key="topic", value="Core",
        now=NOW + timedelta(minutes=2), operation_key="memory:set:second",
    ) == second
    with pytest.raises(ValueError, match="operation_key"):
        service.set(
            _context(), kind="followed_topic", key="topic", value="IMS",
            now=NOW + timedelta(minutes=2), operation_key="memory:set:second",
        )


def test_snapshot_is_exact_and_resume_can_read_historical_revision() -> None:
    service = UserMemoryService(InMemoryUserMemoryStore())
    first = service.set(
        _context(), kind="research_lead", key="lead", value="study X", now=NOW,
    )
    deleted = service.delete(_context(), first.memory_id)
    assert service.list(_context()) == ()
    assert service.snapshot(_context(), (first.memory_id,), include_historical=True) == (deleted,)
    with pytest.raises(AuthorizationError):
        service.snapshot(_context("bob"), (first.memory_id,))


def test_memory_context_is_marked_untrusted() -> None:
    service = UserMemoryService(InMemoryUserMemoryStore())
    item = service.set(_context(), kind="answer_language", key="default", value="Chinese", now=NOW)
    context = service.as_context((item,))[0]
    assert context["trust"] == "untrusted_context"
    assert "evidence" in context["instruction"]


def test_admin_can_load_only_memory_ids_captured_by_an_authorized_run() -> None:
    service = UserMemoryService(InMemoryUserMemoryStore())
    item = service.set(
        _context(), kind="research_lead", key="lead", value="study X", now=NOW,
    )
    admin = _context("admin", role="admin")
    assert service.snapshot_for_run(
        admin, owner_user_id="alice", memory_ids=(item.memory_id,),
    ) == (item,)
    with pytest.raises(AuthorizationError):
        service.snapshot_for_run(
            _context("bob"), owner_user_id="alice", memory_ids=(item.memory_id,),
        )
    with pytest.raises(AuthorizationError):
        service.snapshot_for_run(
            admin, owner_user_id="alice", memory_ids=("UM-" + "F" * 32,),
        )

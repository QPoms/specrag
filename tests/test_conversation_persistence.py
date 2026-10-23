from __future__ import annotations

from pathlib import Path
import sqlite3
from threading import Thread

import pytest

from specrag.access import AuthorizationError, LocalIdentityService
from specrag.context import ContextHarness, ModelProfileResolver, TokenBudgetManager
from specrag.infrastructure import (
    SQLiteContextStore,
    SQLiteConversationStore,
    SQLiteConversationTurnQueue,
    SQLiteDatabase,
    SQLiteIdentityStore,
)
from specrag.infrastructure.database import _MIGRATIONS
from specrag.models import CompactionSummary, ModelContextProfile
from specrag.orchestration import ConversationService


def _harness(
    database: SQLiteDatabase, *, recent_history_tokens: int = 15_000
) -> ContextHarness:
    profile = ModelContextProfile(
        model_id="test",
        context_window_tokens=10_000,
        max_input_tokens=9_000,
        expected_output_tokens=1_000,
    )
    return ContextHarness(
        SQLiteContextStore(database),
        TokenBudgetManager(ModelProfileResolver(profile)),
        recent_history_tokens=recent_history_tokens,
    )


def _setup(path: Path):
    database = SQLiteDatabase(path)
    identity = LocalIdentityService(SQLiteIdentityStore(database))
    identity.bootstrap("alice", "correct horse battery")
    token, _ = identity.login("alice", "correct horse battery")
    workspace = identity.create_workspace(token, "Conversation Workspace")
    actor = identity.context(token, workspace.workspace_id)
    store = SQLiteConversationStore(database)
    service = ConversationService(store, _harness(database))
    return database, identity, token, actor, store, service


def test_message_append_is_atomic_ordered_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "conversation.sqlite3"
    database, _, _, actor, store, service = _setup(path)
    try:
        created = service.create(actor, title="Registration behavior")
        conversation_id = created.conversation.conversation_id
        accepted = service.append_user_message(
            actor,
            conversation_id,
            text="What does the UE do?",
            operation_key="client-message-1",
            expected_revision=0,
        )
        replayed = service.append_user_message(
            actor,
            conversation_id,
            text="What does the UE do?",
            operation_key="client-message-1",
            expected_revision=0,
        )

        assert replayed.message.message_id == accepted.message.message_id
        assert replayed.turn.turn_id == accepted.turn.turn_id
        assert replayed.conversation_revision == 1
        assert replayed.event_cursor == 3
        assert len(store.messages(conversation_id)) == 1
        assert len(store.turns(conversation_id)) == 1
        assert [event.sequence for event in store.events(conversation_id)] == [1, 2, 3]
        assert SQLiteConversationTurnQueue(database).pending_count() == 1

        with pytest.raises(ValueError, match="different input"):
            service.append_user_message(
                actor,
                conversation_id,
                text="A different request",
                operation_key="client-message-1",
                expected_revision=0,
            )
        with pytest.raises(ValueError, match="revision conflict"):
            service.append_user_message(
                actor,
                conversation_id,
                text="A stale write",
                operation_key="client-message-stale",
                expected_revision=0,
            )
        assert len(store.messages(conversation_id)) == 1
        assert len(store.turns(conversation_id)) == 1
        assert SQLiteConversationTurnQueue(database).pending_count() == 1

        second = service.append_user_message(
            actor,
            conversation_id,
            text="Assume the UE is already registered.",
            operation_key="client-message-2",
            expected_revision=1,
        )
        assert second.message.sequence == 2
        assert second.conversation_revision == 2
        assert second.event_cursor == 5
        reconstructed = service.reconstruct(actor, conversation_id)
        assert [item.message_id for item in reconstructed.messages] == [
            accepted.message.message_id,
            second.message.message_id,
        ]
    finally:
        database.close()


def test_conversation_survives_restart_and_archive_blocks_writes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "restart.sqlite3"
    database, identity, token, actor, _, service = _setup(path)
    created = service.create(actor)
    conversation_id = created.conversation.conversation_id
    first = service.append_user_message(
        actor,
        conversation_id,
        text="Keep this complete message.",
        operation_key="message-before-restart",
        expected_revision=0,
    )
    archived, cursor = service.archive(
        actor, conversation_id, expected_revision=1
    )
    assert archived.status == "archived"
    assert cursor == 4
    with pytest.raises(ValueError, match="archived"):
        service.append_user_message(
            actor,
            conversation_id,
            text="This must not be appended.",
            operation_key="archived-write",
            expected_revision=2,
        )
    database.close()

    restored_database = SQLiteDatabase(path)
    try:
        restored_identity = LocalIdentityService(SQLiteIdentityStore(restored_database))
        restored_actor = restored_identity.context(token, actor.workspace_id)
        restored_store = SQLiteConversationStore(restored_database)
        restored = ConversationService(restored_store, _harness(restored_database))
        snapshot = restored.get(restored_actor, conversation_id)
        assert snapshot.conversation.status == "archived"
        assert snapshot.messages[0].message_id == first.message.message_id
        assert snapshot.event_cursor == 4
        active, active_cursor = restored.unarchive(
            restored_actor,
            conversation_id,
            expected_revision=2,
        )
        assert active.status == "active"
        assert active_cursor == 5
    finally:
        restored_database.close()


def test_workspace_admin_cannot_read_another_users_conversation(
    tmp_path: Path,
) -> None:
    database, identity, _, alice, _, service = _setup(
        tmp_path / "private.sqlite3"
    )
    try:
        conversation_id = service.create(alice).conversation.conversation_id
        identity.add_user(alice, "bob", "bob secure password", "admin")
        bob_token, _ = identity.login("bob", "bob secure password")
        bob = identity.context(bob_token, alice.workspace_id)

        with pytest.raises(AuthorizationError):
            service.get(bob, conversation_id)
        assert service.list(bob) == ()
    finally:
        database.close()


def test_turn_queue_serializes_jobs_per_conversation(tmp_path: Path) -> None:
    database, _, _, actor, _, service = _setup(tmp_path / "turn-queue.sqlite3")
    try:
        conversation_id = service.create(actor).conversation.conversation_id
        first = service.append_user_message(
            actor,
            conversation_id,
            text="First",
            operation_key="first",
            expected_revision=0,
        )
        second = service.append_user_message(
            actor,
            conversation_id,
            text="Second",
            operation_key="second",
            expected_revision=1,
        )
        queue = SQLiteConversationTurnQueue(database)
        assert queue.claim("worker-1", lease_seconds=60) == (
            first.turn.turn_id,
            1,
        )
        assert queue.claim("worker-2", lease_seconds=60) is None
        assert queue.renew(first.turn.turn_id, "worker-1", lease_seconds=60)
        assert queue.complete(first.turn.turn_id, "worker-1")
        assert queue.claim("worker-2", lease_seconds=60) == (
            second.turn.turn_id,
            1,
        )
        assert queue.release(
            second.turn.turn_id, "worker-2", delay_seconds=0
        )
        assert queue.claim("worker-3", lease_seconds=60) == (
            second.turn.turn_id,
            2,
        )
    finally:
        database.close()


def test_turn_queue_bounds_parallel_work_per_workspace_user(tmp_path: Path) -> None:
    database, _, _, actor, _, service = _setup(tmp_path / "user-limit.sqlite3")
    try:
        first_conversation = service.create(actor).conversation.conversation_id
        second_conversation = service.create(actor).conversation.conversation_id
        first = service.append_user_message(
            actor,
            first_conversation,
            text="First conversation",
            operation_key="first-conversation",
            expected_revision=0,
        )
        second = service.append_user_message(
            actor,
            second_conversation,
            text="Second conversation",
            operation_key="second-conversation",
            expected_revision=0,
        )
        queue = SQLiteConversationTurnQueue(database, max_active_per_user=1)

        assert queue.claim("worker-1", lease_seconds=60) == (
            first.turn.turn_id,
            1,
        )
        assert queue.claim("worker-2", lease_seconds=60) is None
        assert queue.complete(first.turn.turn_id, "worker-1")
        assert queue.claim("worker-2", lease_seconds=60) == (
            second.turn.turn_id,
            1,
        )
    finally:
        database.close()


def test_concurrent_message_appends_receive_contiguous_sequences(
    tmp_path: Path,
) -> None:
    database, _, _, actor, store, service = _setup(
        tmp_path / "concurrent.sqlite3"
    )
    try:
        conversation_id = service.create(actor).conversation.conversation_id
        receipts = []
        failures = []

        def append(index: int) -> None:
            try:
                receipts.append(service.append_user_message(
                    actor,
                    conversation_id,
                    text=f"message {index}",
                    operation_key=f"concurrent-{index}",
                ))
            except BaseException as exc:  # pragma: no cover - assertion below
                failures.append(exc)

        threads = [Thread(target=append, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert failures == []
        assert sorted(item.message.sequence for item in receipts) == [1, 2]
        assert [item.sequence for item in store.messages(conversation_id)] == [1, 2]
        assert service.get(actor, conversation_id).conversation.revision == 2
    finally:
        database.close()


def test_conversation_compaction_keeps_canonical_messages(tmp_path: Path) -> None:
    database, _, _, actor, store, service = _setup(
        tmp_path / "compaction.sqlite3"
    )
    try:
        compacting_harness = _harness(database, recent_history_tokens=20)
        service = ConversationService(store, compacting_harness)
        conversation_id = service.create(actor).conversation.conversation_id
        first = service.append_user_message(
            actor,
            conversation_id,
            text="old context " * 200,
            operation_key="old-context",
            expected_revision=0,
        )
        service.append_user_message(
            actor,
            conversation_id,
            text="recent context " * 40,
            operation_key="recent-context",
            expected_revision=1,
        )

        class Summary:
            def summarize(self, request):
                return CompactionSummary(
                    objective="Continue the Conversation research.",
                    decisions=("Retain the earlier user condition.",),
                    references=(first.message.message_id,),
                )

        context_id = service.get(actor, conversation_id).conversation.context_id
        compacting_harness.compact(context_id, Summary(), trigger="manual")
        assert len(service.get(actor, conversation_id).messages) == 2
        reconstructed = service.reconstruct(actor, conversation_id)
        assert reconstructed.summary is not None
        assert len(reconstructed.messages) == 1
    finally:
        database.close()


def test_version_six_database_migrates_without_losing_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "migration.sqlite3"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, "
        "applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    for version, statements in _MIGRATIONS[:6]:
        legacy.executescript(statements)
        legacy.execute(
            "INSERT INTO schema_migrations(version) VALUES (?)", (version,)
        )
    legacy.execute(
        "INSERT INTO users(user_id, username, status, created_at) "
        "VALUES ('USER-LEGACY', 'alice', 'active', '2026-01-01T00:00:00+00:00')"
    )
    legacy.commit()
    legacy.close()

    migrated = SQLiteDatabase(path)
    try:
        assert migrated.connection.execute(
            "SELECT username FROM users WHERE user_id='USER-LEGACY'"
        ).fetchone()[0] == "alice"
        assert [
            row[0] for row in migrated.connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ] == list(range(1, 11))
        tables = {
            row[0]
            for row in migrated.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "conversations",
            "conversation_turns",
            "conversation_events",
            "conversation_turn_jobs",
            "run_outcomes",
            "run_completion_jobs",
            "run_completion_receipts",
            "research_results",
            "conversation_research_links",
        } <= tables
    finally:
        migrated.close()

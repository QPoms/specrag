from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from specrag.interfaces.workspace_services import WorkspaceServices
from specrag.models import (
    ConversationResearchLink,
    CorpusVersionRef,
    ResearchInputSnapshot,
    ResearchOutcome,
    ResearchRun,
    RunEvent,
    SpecificationScope,
)
from specrag.orchestration import ResearchLifecycleWorker


VERSION = CorpusVersionRef(specification="24.501", version="17.9.0")
SCOPE = SpecificationScope(policy="exact_version", versions=(VERSION,))


def _setup(path: Path):
    services = WorkspaceServices.create(path)
    services.identity.bootstrap("alice", "correct horse battery")
    token, _ = services.identity.login("alice", "correct horse battery")
    workspace = services.identity.create_workspace(token, "Lifecycle Workspace")
    actor = services.identity.context(token, workspace.workspace_id)
    conversation = services.conversations.create(actor).conversation
    accepted = services.conversations.append_user_message(
        actor,
        conversation.conversation_id,
        text="When shall the UE register?",
        operation_key="message-1",
        expected_revision=0,
    )
    return services, actor, conversation, accepted


def _linked_run(services, actor, conversation, accepted, suffix: str):
    now = datetime.now(UTC)
    snapshot = ResearchInputSnapshot(
        question="When shall the UE register?",
        objective="Establish the normative registration trigger.",
        conditions=("UE is deregistered",),
        requested_scope=SCOPE,
        answer_language="English",
        detail_level="standard",
    )
    run = ResearchRun(
        run_id="RUN-" + suffix * 32,
        thread_id="THREAD-" + suffix * 32,
        workspace_id=actor.workspace_id,
        owner_user_id=actor.user_id,
        question=snapshot.question,
        release=17,
        created_at=now,
        corpus_version_policy="exact_version",
        corpus_versions=(VERSION,),
        conversation_id=conversation.conversation_id,
        origin_message_id=accepted.message.message_id,
        origin_turn_id=accepted.turn.turn_id,
        submitted_user_sequence=accepted.message.sequence,
        submission_operation_key=f"research-{suffix}",
        input_snapshot=snapshot,
    )
    services.runs.put(run)
    link = ConversationResearchLink(
        link_id="LINK-" + suffix * 32,
        conversation_id=conversation.conversation_id,
        origin_message_id=accepted.message.message_id,
        origin_turn_id=accepted.turn.turn_id,
        tool_call_id="CALL-" + suffix * 32,
        run_id=run.run_id,
        submitted_user_sequence=accepted.message.sequence,
        task_status="queued",
        created_at=now,
        updated_at=now,
    )
    services.conversation_lifecycle_store.create_link(link)
    return run, link


def _finish(services, run: ResearchRun, *, status: str = "completed"):
    now = datetime.now(UTC)
    terminal = run.model_copy(update={
        "status": status,
        "stop_reason": status if status != "timed_out" else "elapsed_time_budget",
        "started_at": run.created_at,
        "finished_at": now,
    })
    event = RunEvent(
        event_id="EVT-" + run.run_id[-32:],
        run_id=run.run_id,
        workspace_id=run.workspace_id,
        sequence=1,
        event_type="run_stopped",
        operation_key=f"run:stopped:{status}:attempt:0",
        occurred_at=now,
        payload={"reason": terminal.stop_reason},
    )
    outcome = None
    if status == "completed":
        outcome = ResearchOutcome(
            run_id=run.run_id,
            question=run.question,
            objective=run.input_snapshot.objective,
            conditions=run.input_snapshot.conditions,
            requested_scope=SCOPE,
            corpus_versions=(VERSION,),
            stop_reason="insufficient_evidence",
            created_at=now,
        )
        terminal = terminal.model_copy(update={
            "stop_reason": "insufficient_evidence"
        })
        event = event.model_copy(update={
            "operation_key": "run:stopped:completed:attempt:0",
            "payload": {"reason": "insufficient_evidence"},
        })
    services.runs.finalize(terminal, event, outcome)
    return terminal, event, outcome


def _complete_origin_turn(services, accepted) -> None:
    now = datetime.now(UTC)
    completed = accepted.turn.model_copy(update={
        "status": "completed",
        "started_at": accepted.turn.created_at,
        "finished_at": now,
        "reply_message_id": "MSG-" + "F" * 32,
    })
    with services.database.connection:
        services.database.connection.execute(
            "UPDATE conversation_turns SET status='completed', payload_json=?, "
            "updated_at=? WHERE turn_id=?",
            (
                completed.model_dump_json(),
                now.isoformat(),
                completed.turn_id,
            ),
        )
        services.database.connection.execute(
            "DELETE FROM conversation_turn_jobs WHERE turn_id=?",
            (completed.turn_id,),
        )


def _worker(services) -> ResearchLifecycleWorker:
    return ResearchLifecycleWorker(
        services.runs,
        services.research_results,
        services.conversation_lifecycle_store,
        services.completion_queue,
        retry_delay_seconds=0,
    )


def test_completion_defers_then_queues_one_idempotent_presentation(
    tmp_path: Path,
) -> None:
    services, actor, conversation, accepted = _setup(tmp_path / "defer.sqlite3")
    try:
        run, _ = _linked_run(services, actor, conversation, accepted, "A")
        terminal, event, outcome = _finish(services, run)
        worker = _worker(services)
        assert worker.process_one()
        pending = services.conversation_lifecycle_store.get_by_run(run.run_id)
        assert pending.presentation_status == "pending"
        assert services.completion_queue.pending_count() == 0

        _complete_origin_turn(services, accepted)
        queued = services.conversation_lifecycle_store.schedule_pending(
            conversation.conversation_id
        )
        assert queued.presentation_status == "queued"
        assert services.conversation_turn_queue.pending_count() == 1
        result = services.result_store.latest_for_run(run.run_id)
        assert result is not None
        assert result.evidence_status == "insufficient"

        events = services.conversation_store.events(
            conversation.conversation_id
        )
        assert [event.sequence for event in events] == list(
            range(1, len(events) + 1)
        )
        assert {event.event_type for event in events} >= {
            "run_linked",
            "run_status_changed",
            "result_available",
            "auto_reply_deferred",
            "auto_reply_queued",
        }
        event_count = len(events)
        services.conversation_lifecycle_store.reconcile(
            terminal, 0, "completed", result
        )
        assert len(services.conversation_store.events(
            conversation.conversation_id
        )) == event_count
        services.runs.finalize(terminal, event, outcome)
        assert services.completion_queue.pending_count() == 0
    finally:
        services.close()


def test_newer_user_message_skips_late_result(tmp_path: Path) -> None:
    services, actor, conversation, accepted = _setup(tmp_path / "skip.sqlite3")
    try:
        run, _ = _linked_run(services, actor, conversation, accepted, "B")
        services.conversations.append_user_message(
            actor,
            conversation.conversation_id,
            text="Let us discuss a different topic.",
            operation_key="message-2",
            expected_revision=1,
        )
        terminal, _, _ = _finish(services, run)
        assert _worker(services).process_one()

        link = services.conversation_lifecycle_store.get_by_run(run.run_id)
        assert link.presentation_status == "skipped"
        assert link.status_reason == "newer_user_message"
        assert services.result_store.latest_for_run(run.run_id) is not None
        assert not any(
            turn.trigger == "research_result_ready"
            for turn in services.conversation_store.turns(
                conversation.conversation_id
            )
        )
        result = services.result_store.latest_for_run(run.run_id)
        event_count = len(services.conversation_store.events(
            conversation.conversation_id
        ))
        replay = services.conversation_lifecycle_store.reconcile(
            terminal, 0, "completed", result
        )
        assert replay.presentation_status == "skipped"
        assert len(services.conversation_store.events(
            conversation.conversation_id
        )) == event_count
    finally:
        services.close()


def test_multiple_results_queue_only_one_and_new_message_cancels_it(
    tmp_path: Path,
) -> None:
    services, actor, conversation, accepted = _setup(tmp_path / "many.sqlite3")
    try:
        first, _ = _linked_run(services, actor, conversation, accepted, "C")
        second, _ = _linked_run(services, actor, conversation, accepted, "D")
        _complete_origin_turn(services, accepted)
        _finish(services, first)
        _finish(services, second)
        worker = _worker(services)
        assert worker.process_one()
        assert worker.process_one()

        links = services.conversation_lifecycle_store.list_for_conversation(
            conversation.conversation_id
        )
        assert sorted(link.presentation_status for link in links) == [
            "pending", "queued"
        ]
        auto_turns = [
            turn for turn in services.conversation_store.turns(
                conversation.conversation_id
            ) if turn.trigger == "research_result_ready"
        ]
        assert len(auto_turns) == 1

        services.conversations.append_user_message(
            actor,
            conversation.conversation_id,
            text="Change the conditions before presenting either result.",
            operation_key="message-after-results",
            expected_revision=1,
        )
        links = services.conversation_lifecycle_store.list_for_conversation(
            conversation.conversation_id
        )
        assert {link.presentation_status for link in links} == {"skipped"}
        assert all(
            link.status_reason == "newer_user_message" for link in links
        )
        stored_auto = next(
            turn for turn in services.conversation_store.turns(
                conversation.conversation_id
            ) if turn.turn_id == auto_turns[0].turn_id
        )
        assert stored_auto.status == "cancelled"
    finally:
        services.close()


def test_abnormal_completion_updates_link_without_result(tmp_path: Path) -> None:
    services, actor, conversation, accepted = _setup(tmp_path / "failed.sqlite3")
    try:
        run, _ = _linked_run(services, actor, conversation, accepted, "E")
        _finish(services, run, status="failed")
        assert _worker(services).process_one()

        link = services.conversation_lifecycle_store.get_by_run(run.run_id)
        assert link.task_status == "failed"
        assert link.latest_result_id is None
        assert link.presentation_status is None
        assert services.result_store.latest_for_run(run.run_id) is None
    finally:
        services.close()


def test_completion_waits_for_late_tool_call_link(tmp_path: Path) -> None:
    services, actor, conversation, accepted = _setup(tmp_path / "race.sqlite3")
    try:
        run, link = _linked_run(services, actor, conversation, accepted, "F")
        with services.database.connection:
            services.database.connection.execute(
                "DELETE FROM conversation_research_links WHERE run_id=?",
                (run.run_id,),
            )
        _finish(services, run)
        worker = _worker(services)
        assert worker.process_one()
        assert services.completion_queue.pending_count() == 1
        assert services.result_store.latest_for_run(run.run_id) is None

        services.conversation_lifecycle_store.create_link(link)
        _complete_origin_turn(services, accepted)
        assert worker.process_one()
        assert services.completion_queue.pending_count() == 0
        stored = services.conversation_lifecycle_store.get_by_run(run.run_id)
        assert stored.presentation_status == "queued"
        assert services.result_store.latest_for_run(run.run_id) is not None
    finally:
        services.close()

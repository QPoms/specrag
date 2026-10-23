from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from specrag.access import AuthorizationError
from specrag.infrastructure import InMemoryRunStore
from specrag.infrastructure.structured_agent import CodexAgentError
from specrag.models import (
    ActorContext,
    CorpusVersionRef,
    ResearchRequest,
    SpecificationScope,
)
from specrag.orchestration import RunHarness


NOW = datetime.now(timezone.utc)


def _context(user="alice", workspace="WA", role="member"):
    return ActorContext(
        user_id=user, workspace_id=workspace, role=role,
        session_id=f"session-{user}", authenticated_at=NOW,
        expires_at=NOW + timedelta(hours=12),
    )


class _Workflow:
    def __init__(self, control, *, crash=False):
        self.control = control
        self.crash = crash

    def resume(self, *, thread_id):
        from specrag.orchestration.application import RuntimeStopped
        harness = self.control.__self__
        run = next(item for item in harness.store._runs.values() if item.thread_id == thread_id)
        reason = self.control(run.run_id)
        if reason:
            raise RuntimeStopped(reason)
        return {"stop_reason": "completed"}

    def checkpoint(self, *, thread_id):
        return SimpleNamespace(values={"thread_id": thread_id})


class _Application:
    def __init__(self, control, *, crash=False):
        self.control = control
        self.workflow = _Workflow(control, crash=crash)

    def run(
        self, *, research_id, question, thread_id, budget,
        specfact_ids=(), memory_ids=(),
    ):
        reason = self.control(research_id)
        if reason:
            from specrag.orchestration.application import RuntimeStopped
            raise RuntimeStopped(reason)
        return {"stop_reason": "completed"}


def _factory(release, control, event_sink, research_context):
    return _Application(control)


class _ProcessKilled(BaseException):
    pass


class _CrashingApplication(_Application):
    def run(self, **kwargs):
        raise _ProcessKilled()


def _crashing_factory(release, control, event_sink, research_context):
    return _CrashingApplication(control)


class _ApiFailureApplication(_Application):
    def run(self, **kwargs):
        raise CodexAgentError(
            "sanitized provider failure",
            failure_kind="rate_limited",
            retryable=True,
            status_code=429,
            attempt_count=8,
        )


def _api_failure_factory(release, control, event_sink, research_context):
    return _ApiFailureApplication(control)


def test_run_content_is_owner_private_and_admin_gets_metadata_only() -> None:
    store = InMemoryRunStore()
    harness = RunHarness(store, _factory)
    result = harness.start(_context(), ResearchRequest(question="question"))
    run_id = result.run.run_id
    with pytest.raises(AuthorizationError):
        harness.inspect(_context("bob"), run_id)
    with pytest.raises(AuthorizationError):
        harness.inspect(_context("alice", "WB"), run_id)
    admin = _context("admin", role="admin")
    with pytest.raises(AuthorizationError):
        harness.inspect(admin, run_id)
    assert harness.metadata_status(admin, run_id) == result.run
    assert harness.list_runs(admin) == ()
    assert harness.list_run_metadata(admin) == (result.run,)


def test_cancel_is_checked_before_runtime_execution() -> None:
    store = InMemoryRunStore()
    harness = RunHarness(store, _factory)
    completed = harness.start(_context(), ResearchRequest(question="question"))
    assert harness.cancel(_context(), completed.run.run_id).status == "completed"

    original = completed.run.model_copy(update={
        "run_id": "RUN-" + "2" * 32,
        "thread_id": "THREAD-" + "2" * 32,
        "status": "queued",
        "stop_reason": None,
        "started_at": None,
        "finished_at": None,
    })
    store.put(original)
    harness.cancel(_context(), original.run_id)
    resumed = harness.resume(_context(), original.run_id)
    assert resumed.run.status == "cancelled"
    assert resumed.run.stop_reason == "cancelled"


def test_run_events_and_snapshot_ids_are_stable() -> None:
    store = InMemoryRunStore()

    def snapshot(context, question, release):
        return ("SF-" + "1" * 24,), ("UM-" + "1" * 32,), ({"hint": question},)

    harness = RunHarness(store, _factory, context_snapshot=snapshot)
    result = harness.start(_context(), ResearchRequest(
        question="question", configuration_memory_ids=("UM-" + "2" * 32,),
    ))
    assert result.run.specfact_ids == ("SF-" + "1" * 24,)
    assert result.run.memory_ids == (
        "UM-" + "1" * 32,
        "UM-" + "2" * 32,
    )
    assert [event.event_type for event in result.events] == ["run_started", "run_stopped"]
    assert len({event.operation_key for event in result.events}) == 2


def test_run_freezes_exact_corpus_versions_at_start() -> None:
    store = InMemoryRunStore()
    snapshots = [(
        CorpusVersionRef(specification="24.501", version="17.17.0"),
        CorpusVersionRef(specification="33.501", version="17.12.0"),
    )]
    harness = RunHarness(store, _factory, corpus_snapshot=lambda _scope: snapshots[0])

    result = harness.start(_context(), ResearchRequest(question="question"))
    snapshots[0] = (
        CorpusVersionRef(specification="24.501", version="17.18.0"),
        CorpusVersionRef(specification="33.501", version="17.12.0"),
    )

    persisted = harness.inspect(_context(), result.run.run_id).run
    assert persisted.corpus_version_policy == "mixed_versions"
    assert [item.version for item in persisted.corpus_versions] == [
        "17.17.0", "17.12.0"
    ]


def test_conversation_submission_freezes_input_and_is_idempotent() -> None:
    store = InMemoryRunStore()
    bound = CorpusVersionRef(specification="24.501", version="17.9.0")
    harness = RunHarness(store, _factory, corpus_snapshot=lambda _scope: (bound,))
    request = ResearchRequest(
        question="When shall the UE register?",
        objective="Determine the normative registration trigger.",
        conditions=("UE is in 5GMM-DEREGISTERED",),
        requested_scope=SpecificationScope(
            policy="exact_version", versions=(bound,)
        ),
        conversation_id="CONV-" + "A" * 32,
        origin_message_id="MESSAGE-1",
        origin_turn_id="TURN-" + "B" * 32,
        submitted_user_sequence=4,
        submission_operation_key="turn-B:research-1",
    )

    first = harness.submit(_context(), request)
    replay = harness.submit(_context(), request)

    assert replay.run.run_id == first.run.run_id
    assert first.run.input_snapshot is not None
    assert first.run.input_snapshot.objective == request.objective
    assert first.run.input_snapshot.conditions == request.conditions
    assert first.run.corpus_versions == (bound,)
    assert len(store._runs) == 1
    assert len(store.events(first.run.run_id)) == 1

    with pytest.raises(ValueError, match="different input"):
        harness.submit(
            _context(),
            request.model_copy(update={"conditions": ("UE is registered",)}),
        )


@pytest.mark.parametrize(
    "reason",
    [
        "iteration_budget",
        "tool_budget",
        "replan_budget",
        "insufficient_evidence",
    ],
)
def test_normal_research_stops_publish_outcome_and_completion_job(reason) -> None:
    class StateApplication(_Application):
        def run(self, **kwargs):
            return {"stop_reason": reason}

    def factory(release, control, event_sink, research_context):
        return StateApplication(control)

    store = InMemoryRunStore()
    result = RunHarness(store, factory).start(
        _context(), ResearchRequest(question="question")
    )

    assert result.run.status == "completed"
    assert result.run.stop_reason == reason
    outcome = store.get_outcome(result.run.run_id, 0)
    assert outcome is not None
    assert outcome.stop_reason == reason
    assert (result.run.run_id, 0) in store._completion_jobs


def test_elapsed_deadline_stops_resume() -> None:
    store = InMemoryRunStore()
    harness = RunHarness(store, _factory)
    completed = harness.start(_context(), ResearchRequest(question="question"))
    expired = completed.run.model_copy(update={
        "run_id": "RUN-" + "3" * 32,
        "thread_id": "THREAD-" + "3" * 32,
        "status": "running",
        "stop_reason": None,
        "finished_at": None,
        "deadline_at": NOW - timedelta(seconds=1),
    })
    store.put(expired)
    result = harness.resume(_context(), expired.run_id)
    assert result.run.status == "timed_out"
    assert result.run.stop_reason == "elapsed_time_budget"


def test_new_harness_can_resume_after_process_interruption() -> None:
    store = InMemoryRunStore()
    first_process = RunHarness(store, _crashing_factory)
    with pytest.raises(_ProcessKilled):
        first_process.start(_context(), ResearchRequest(question="question"))
    interrupted = next(iter(store._runs.values()))
    assert interrupted.status == "running"

    restarted_process = RunHarness(store, _factory)
    result = restarted_process.resume(_context(), interrupted.run_id)
    assert result.run.status == "completed"
    assert result.run.stop_reason == "completed"
    assert [event.event_type for event in result.events].count("run_stopped") == 1


def test_api_failure_is_persisted_as_safe_machine_readable_metadata() -> None:
    store = InMemoryRunStore()
    harness = RunHarness(store, _api_failure_factory)
    with pytest.raises(CodexAgentError):
        harness.start(_context(), ResearchRequest(question="question"))

    failed = next(iter(store._runs.values()))
    assert failed.status == "failed"
    assert failed.failure_kind == "rate_limited"
    assert failed.failure_retryable is True
    assert failed.failure_attempt_count == 8
    assert failed.failure_status_code == 429
    stopped = store.events(failed.run_id)[-1]
    assert stopped.event_type == "run_stopped"
    assert stopped.payload == {
        "reason": "failed",
        "failure_kind": "rate_limited",
        "retryable": True,
        "attempt_count": 8,
        "status_code": 429,
    }


def test_terminal_run_can_queue_resume_only_with_checkpoint() -> None:
    store = InMemoryRunStore()
    harness = RunHarness(store, _factory)
    completed = harness.start(_context(), ResearchRequest(question="question"))
    failed = completed.run.model_copy(update={
        "status": "failed",
        "stop_reason": "failed",
    })
    store.put(failed)

    queued = harness.queue_resume(
        _context(), failed.run_id, elapsed_time_seconds=120
    )
    duplicate = harness.queue_resume(
        _context(), failed.run_id, elapsed_time_seconds=120
    )

    assert queued.run.status == duplicate.run.status == "queued"
    assert queued.run.finished_at is None
    assert queued.run.cancel_requested_at is None
    assert [event.event_type for event in duplicate.events].count("resumed") == 1
    resumed = harness.execute_resume(_context(), failed.run_id)
    assert resumed.run.status == "completed"
    assert resumed.run.run_attempt == 1
    assert store.get_outcome(resumed.run.run_id, 0) is not None
    assert store.get_outcome(resumed.run.run_id, 1) is not None


def test_retry_creates_new_linked_run_with_fresh_lifecycle() -> None:
    store = InMemoryRunStore()
    harness = RunHarness(store, _factory)
    completed = harness.start(_context(), ResearchRequest(
        question="question",
        release=18,
        answer_language="Chinese",
        detail_level="detailed",
    ))

    retried = harness.retry(
        _context(), completed.run.run_id, elapsed_time_seconds=300
    )

    assert retried.run.run_id != completed.run.run_id
    assert retried.run.retry_of_run_id == completed.run.run_id
    assert retried.run.status == "queued"
    assert retried.run.question == completed.run.question
    assert retried.run.release == 18
    assert retried.run.answer_language == "Chinese"

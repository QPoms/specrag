from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from specrag.access import AuthorizationError
from specrag.infrastructure import InMemoryRunStore
from specrag.models import ActorContext, ResearchRequest
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


def test_run_access_is_owner_scoped_with_admin_override() -> None:
    store = InMemoryRunStore()
    harness = RunHarness(store, _factory)
    result = harness.start(_context(), ResearchRequest(question="question"))
    run_id = result.run.run_id
    with pytest.raises(AuthorizationError):
        harness.inspect(_context("bob"), run_id)
    with pytest.raises(AuthorizationError):
        harness.inspect(_context("alice", "WB"), run_id)
    assert harness.inspect(_context("admin", role="admin"), run_id).run == result.run


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

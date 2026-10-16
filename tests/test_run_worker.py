from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time

from specrag.infrastructure import InMemoryRunStore
from specrag.models import ActorContext, ResearchRequest
from specrag.orchestration import RunHarness, RunWorker


class _Queue:
    def __init__(self) -> None:
        self.jobs: dict[str, tuple[str, str, int]] = {}

    def enqueue(self, run_id: str, action: str) -> None:
        existing = self.jobs.get(run_id)
        if existing is not None and existing[1] == "leased":
            return
        self.jobs[run_id] = (action, "queued", existing[2] if existing else 0)

    def ensure(self, run_id: str, action: str) -> None:
        self.jobs.setdefault(run_id, (action, "queued", 0))

    def claim(self, worker_id: str, *, lease_seconds: float):
        for run_id, (action, state, attempts) in tuple(self.jobs.items()):
            if state == "queued":
                self.jobs[run_id] = (worker_id, "leased", attempts + 1)
                return run_id, action, attempts + 1
        return None

    def renew(self, run_id: str, worker_id: str, *, lease_seconds: float) -> bool:
        item = self.jobs.get(run_id)
        return bool(item and item[1] == "leased" and item[0] == worker_id)

    def complete(self, run_id: str, worker_id: str) -> bool:
        item = self.jobs.get(run_id)
        if item and item[1] == "leased" and item[0] == worker_id:
            del self.jobs[run_id]
            return True
        return False

    def release(self, run_id: str, worker_id: str, *, delay_seconds: float) -> bool:
        item = self.jobs.get(run_id)
        if item and item[1] == "leased" and item[0] == worker_id:
            self.jobs[run_id] = (item[0], "queued", item[2])
            return True
        return False

    def pending_count(self) -> int:
        return len(self.jobs)


class _Application:
    def run(self, **kwargs):
        return {"stop_reason": "completed"}


def _factory(release, stop_requested, event_sink, research_context):
    return _Application()


def _context() -> ActorContext:
    now = datetime.now(timezone.utc)
    return ActorContext(
        user_id="USR-1",
        workspace_id="WS-1",
        role="member",
        session_id="session-1",
        authenticated_at=now,
        expires_at=now + timedelta(hours=1),
    )


def test_worker_restart_recovers_persisted_queued_run() -> None:
    store = InMemoryRunStore()
    harness = RunHarness(store, _factory)
    queue = _Queue()
    first = RunWorker(harness, queue, poll_seconds=0.01)
    submitted = first.submit(_context(), ResearchRequest(question="recover me"))
    assert submitted.run.status == "queued"

    restarted = RunWorker(harness, queue, poll_seconds=0.01)
    restarted.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if store.get(submitted.run.run_id).status == "completed":
                break
            time.sleep(0.01)
        assert store.get(submitted.run.run_id).status == "completed"
        assert queue.pending_count() == 0
    finally:
        restarted.stop()

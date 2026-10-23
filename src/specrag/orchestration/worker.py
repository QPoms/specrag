"""Durable bounded Worker for asynchronous Run execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Event, Thread
from typing import Literal, Protocol
from uuid import uuid4

from specrag.models import ActorContext, ResearchRequest, ResearchRun

from .harness import RunHarness, RunInspection


RunJobAction = Literal["execute", "resume"]


class RunQueue(Protocol):
    def enqueue(self, run_id: str, action: RunJobAction) -> None: ...
    def ensure(self, run_id: str, action: RunJobAction) -> None: ...
    def claim(
        self, worker_id: str, *, lease_seconds: float,
    ) -> tuple[str, RunJobAction, int] | None: ...
    def renew(
        self, run_id: str, worker_id: str, *, lease_seconds: float,
    ) -> bool: ...
    def complete(self, run_id: str, worker_id: str) -> bool: ...
    def release(
        self, run_id: str, worker_id: str, *, delay_seconds: float,
    ) -> bool: ...
    def pending_count(self) -> int: ...


@dataclass(frozen=True)
class RunJob:
    run_id: str
    action: RunJobAction
    attempt: int


def _worker_context(run: ResearchRun) -> ActorContext:
    """Rebuild a trusted internal actor for an already-authorized Run."""

    now = datetime.now(timezone.utc)
    return ActorContext(
        user_id=run.owner_user_id,
        workspace_id=run.workspace_id,
        role="member",
        session_id=f"worker-{run.run_id}",
        authenticated_at=now,
        expires_at=now + timedelta(days=1),
    )


class RunWorker:
    """Poll durable jobs, lease them, and execute the existing RunHarness."""

    def __init__(
        self,
        harness: RunHarness,
        queue: RunQueue,
        *,
        poll_seconds: float = 0.1,
        lease_seconds: float = 30.0,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        if poll_seconds <= 0 or lease_seconds <= 0 or retry_delay_seconds < 0:
            raise ValueError("Worker timing values must be positive")
        self.harness = harness
        self.queue = queue
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.retry_delay_seconds = retry_delay_seconds
        self.worker_id = "WORKER-" + uuid4().hex.upper()
        self._stopping = Event()
        self._wake = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self.recover()
        self._thread = Thread(
            target=self._run,
            name="specrag-run-worker",
            daemon=True,
        )
        self._thread.start()

    def run_forever(self) -> None:
        self._stopping.clear()
        self.recover()
        self._run()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def recover(self) -> int:
        """Repair durable jobs for active Runs left by an older process."""

        recovered = 0
        for run in self.harness.recoverable_runs():
            events = self.harness.store.events(run.run_id)
            latest_started = max(
                (item.sequence for item in events if item.event_type == "run_started"),
                default=0,
            )
            latest_resumed = max(
                (item.sequence for item in events if item.event_type == "resumed"),
                default=0,
            )
            action: RunJobAction = (
                "resume"
                if run.status == "running" or latest_resumed > latest_started
                else "execute"
            )
            self.queue.ensure(run.run_id, action)
            recovered += 1
        if recovered:
            self._wake.set()
        return recovered

    def submit(
        self, context: ActorContext, request: ResearchRequest,
    ) -> RunInspection:
        inspection = self.harness.submit(context, request)
        if inspection.run.status == "queued":
            self.queue.enqueue(inspection.run.run_id, "execute")
            self._wake.set()
        return inspection

    def resume(
        self,
        context: ActorContext,
        run_id: str,
        *,
        elapsed_time_seconds: int = 900,
    ) -> RunInspection:
        inspection = self.harness.queue_resume(
            context,
            run_id,
            elapsed_time_seconds=elapsed_time_seconds,
        )
        self.queue.enqueue(run_id, "resume")
        self._wake.set()
        return inspection

    def retry(
        self,
        context: ActorContext,
        run_id: str,
        *,
        elapsed_time_seconds: int = 900,
    ) -> RunInspection:
        inspection = self.harness.retry(
            context,
            run_id,
            elapsed_time_seconds=elapsed_time_seconds,
        )
        self.queue.enqueue(inspection.run.run_id, "execute")
        self._wake.set()
        return inspection

    def _run(self) -> None:
        while not self._stopping.is_set():
            if not self.process_one():
                self._wake.wait(self.poll_seconds)
                self._wake.clear()

    def process_one(self) -> bool:
        """Execute at most one leased job for deterministic embedding/tests."""

        claimed = self.queue.claim(
            self.worker_id, lease_seconds=self.lease_seconds
        )
        if claimed is None:
            return False
        job = RunJob(*claimed)
        run = self.harness.store.get(job.run_id)
        if run is None:
            self.queue.complete(job.run_id, self.worker_id)
            return True

        heartbeat_stop = Event()
        heartbeat = Thread(
            target=self._heartbeat,
            args=(job.run_id, heartbeat_stop),
            name=f"specrag-lease-{job.run_id[-8:]}",
            daemon=True,
        )
        heartbeat.start()
        try:
            context = _worker_context(run)
            if job.action == "resume":
                self.harness.execute_resume(context, job.run_id)
            else:
                self.harness.execute_queued(context, job.run_id)
        except Exception:
            current = self.harness.store.get(job.run_id)
            if current is not None and current.status in {
                "completed", "failed", "cancelled", "timed_out"
            }:
                self.queue.complete(job.run_id, self.worker_id)
            else:
                self.queue.release(
                    job.run_id,
                    self.worker_id,
                    delay_seconds=self.retry_delay_seconds,
                )
        else:
            self.queue.complete(job.run_id, self.worker_id)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=self.poll_seconds + 0.5)
        return True

    def _heartbeat(self, run_id: str, stopping: Event) -> None:
        interval = max(0.05, self.lease_seconds / 3)
        while not stopping.wait(interval):
            if not self.queue.renew(
                run_id, self.worker_id, lease_seconds=self.lease_seconds
            ):
                return


__all__ = ["RunJob", "RunQueue", "RunWorker"]

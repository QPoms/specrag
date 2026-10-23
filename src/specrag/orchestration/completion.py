"""Durable lifecycle worker for terminal Run reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from typing import Protocol
from uuid import uuid4

from specrag.infrastructure.conversation_lifecycle_store import ResearchLinkPending
from specrag.models import (
    ConversationResearchLink,
    ResearchResult,
    ResearchRun,
    RunStatus,
)

from .results import ResearchResultService
from .run_contracts import RunStore


class CompletionQueue(Protocol):
    def claim(
        self, worker_id: str, *, lease_seconds: float
    ) -> tuple[str, int, int] | None: ...
    def renew(
        self,
        run_id: str,
        run_attempt: int,
        worker_id: str,
        *,
        lease_seconds: float,
    ) -> bool: ...
    def complete(
        self, run_id: str, run_attempt: int, worker_id: str
    ) -> bool: ...
    def release(
        self,
        run_id: str,
        run_attempt: int,
        worker_id: str,
        *,
        delay_seconds: float,
    ) -> bool: ...
    def pending_count(self) -> int: ...


class CompletionLifecycleStore(Protocol):
    def reconcile(
        self,
        run: ResearchRun,
        run_attempt: int,
        task_status: RunStatus,
        result: ResearchResult | None,
        *,
        now=None,
    ) -> ConversationResearchLink | None: ...


@dataclass(frozen=True)
class CompletionJob:
    run_id: str
    run_attempt: int
    attempt: int


class ResearchLifecycleWorker:
    def __init__(
        self,
        runs: RunStore,
        results: ResearchResultService,
        lifecycle: CompletionLifecycleStore,
        queue: CompletionQueue,
        *,
        poll_seconds: float = 0.1,
        lease_seconds: float = 30.0,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        if poll_seconds <= 0 or lease_seconds <= 0 or retry_delay_seconds < 0:
            raise ValueError("Worker timing values must be positive")
        self.runs = runs
        self.results = results
        self.lifecycle = lifecycle
        self.queue = queue
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.retry_delay_seconds = retry_delay_seconds
        self.worker_id = "LIFECYCLE-" + uuid4().hex.upper()
        self._stopping = Event()
        self._wake = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = Thread(
            target=self._run,
            name="specrag-lifecycle-worker",
            daemon=True,
        )
        self._thread.start()

    def run_forever(self) -> None:
        self._stopping.clear()
        self._run()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def process_one(self) -> bool:
        claimed = self.queue.claim(
            self.worker_id, lease_seconds=self.lease_seconds
        )
        if claimed is None:
            return False
        job = CompletionJob(*claimed)
        run = self.runs.get(job.run_id)
        if run is None:
            self.queue.complete(job.run_id, job.run_attempt, self.worker_id)
            return True
        heartbeat_stop = Event()
        heartbeat = Thread(
            target=self._heartbeat,
            args=(job, heartbeat_stop),
            name=f"specrag-completion-lease-{job.run_id[-8:]}",
            daemon=True,
        )
        heartbeat.start()
        try:
            status = self._attempt_status(run, job.run_attempt)
            result = (
                self.results.build(job.run_id, job.run_attempt)
                if status == "completed"
                else None
            )
            if status == "completed" and result is None:
                raise RuntimeError("completed Run attempt has no Research Outcome")
            self.lifecycle.reconcile(
                run, job.run_attempt, status, result
            )
        except ResearchLinkPending:
            self.queue.release(
                job.run_id,
                job.run_attempt,
                self.worker_id,
                delay_seconds=self.retry_delay_seconds,
            )
        except Exception:
            self.queue.release(
                job.run_id,
                job.run_attempt,
                self.worker_id,
                delay_seconds=self.retry_delay_seconds,
            )
        else:
            self.queue.complete(job.run_id, job.run_attempt, self.worker_id)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=self.poll_seconds + 0.5)
        return True

    def _run(self) -> None:
        while not self._stopping.is_set():
            if not self.process_one():
                self._wake.wait(self.poll_seconds)
                self._wake.clear()

    def _heartbeat(self, job: CompletionJob, stopping: Event) -> None:
        interval = max(0.05, self.lease_seconds / 3)
        while not stopping.wait(interval):
            if not self.queue.renew(
                job.run_id,
                job.run_attempt,
                self.worker_id,
                lease_seconds=self.lease_seconds,
            ):
                return

    def _attempt_status(self, run: ResearchRun, run_attempt: int) -> RunStatus:
        if run.run_attempt == run_attempt and run.status in {
            "completed", "failed", "cancelled", "timed_out"
        }:
            return run.status
        suffix = f":attempt:{run_attempt}"
        for event in reversed(self.runs.events(run.run_id)):
            operation = event.operation_key or ""
            if event.event_type != "run_stopped" or not operation.endswith(suffix):
                continue
            parts = operation.split(":")
            if len(parts) >= 5 and parts[2] in {
                "completed", "failed", "cancelled", "timed_out"
            }:
                return parts[2]  # type: ignore[return-value]
        raise RuntimeError("completion job has no terminal Run attempt")


__all__ = [
    "CompletionJob",
    "CompletionLifecycleStore",
    "CompletionQueue",
    "ResearchLifecycleWorker",
]

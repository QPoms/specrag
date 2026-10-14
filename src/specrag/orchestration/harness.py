"""Authorized durable lifecycle for research runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from uuid import uuid4

from specrag.access import AccessPolicy
from specrag.models import (
    ActorContext,
    ResearchRequest,
    ResearchRun,
    RunEvent,
)

from .application import ResearchApplication, RuntimeStopped
from .state import ResearchState


class RunStore(Protocol):
    def put(self, run: ResearchRun) -> None: ...
    def get(self, run_id: str) -> ResearchRun | None: ...
    def is_cancel_requested(self, run_id: str) -> bool: ...
    def append_event(self, event: RunEvent) -> RunEvent: ...
    def events(self, run_id: str) -> tuple[RunEvent, ...]: ...


ApplicationFactory = Callable[
    [int, Callable[[str], str | None], Callable[..., None], tuple[dict[str, object], ...]],
    ResearchApplication,
]
ContextSnapshot = Callable[
    [ActorContext, str, int], tuple[tuple[str, ...], tuple[str, ...], tuple[dict[str, object], ...]]
]
ContextLoader = Callable[[ActorContext, ResearchRun], tuple[dict[str, object], ...]]
ProposalWriter = Callable[[ActorContext, ResearchRun, ResearchState], tuple[str, ...]]


@dataclass(frozen=True)
class RunInspection:
    run: ResearchRun
    events: tuple[RunEvent, ...]
    checkpoint: dict[str, Any] | None
    stage_latency_ms: dict[str, float]
    input_tokens: int
    output_tokens: int


class RunHarness:
    def __init__(
        self,
        store: RunStore,
        application_factory: ApplicationFactory,
        *,
        policy: AccessPolicy | None = None,
        context_snapshot: ContextSnapshot | None = None,
        context_loader: ContextLoader | None = None,
        proposal_writer: ProposalWriter | None = None,
    ) -> None:
        self.store = store
        self.application_factory = application_factory
        self.policy = policy or AccessPolicy()
        self.context_snapshot = context_snapshot
        self.context_loader = context_loader
        self.proposal_writer = proposal_writer

    def start(self, context: ActorContext, request: ResearchRequest) -> RunInspection:
        self.policy.require(context, "research.create", workspace_id=context.workspace_id)
        now = datetime.now(timezone.utc)
        run = ResearchRun(
            run_id="RUN-" + uuid4().hex.upper(),
            thread_id="THREAD-" + uuid4().hex.upper(),
            workspace_id=context.workspace_id,
            owner_user_id=context.user_id,
            question=request.question,
            release=request.release,
            answer_language=request.answer_language,
            detail_level=request.detail_level,
            development_question_id=request.development_question_id,
            created_at=now,
            deadline_at=now + timedelta(seconds=request.elapsed_time_seconds),
        )
        if self.context_snapshot is not None:
            facts, memories, _ = self.context_snapshot(
                context, request.question, request.release
            )
            run = run.model_copy(update={"specfact_ids": facts, "memory_ids": memories})
        self.store.put(run)
        self._event(run, "run_started", "run:start", payload={"release": run.release})
        return self._execute(context, run, request=request, resume=False)

    def resume(self, context: ActorContext, run_id: str) -> RunInspection:
        run = self._authorized_run(context, run_id, "run.resume")
        if run.status in {"completed", "cancelled", "timed_out"}:
            return self.inspect(context, run_id)
        self._event(run, "resumed", f"run:resume:{len(self.store.events(run_id))}")
        return self._execute(context, run, request=None, resume=True)

    def inspect(self, context: ActorContext, run_id: str) -> RunInspection:
        run = self._authorized_run(context, run_id, "run.read")
        events = self.store.events(run_id)
        checkpoint = None
        try:
            app = self._application(context, run)
            snapshot = app.workflow.checkpoint(thread_id=run.thread_id)
            checkpoint = dict(snapshot.values) if getattr(snapshot, "values", None) else None
        except (LookupError, RuntimeError, ValueError):
            checkpoint = None
        latency: dict[str, float] = {}
        input_tokens = 0
        output_tokens = 0
        for event in events:
            if event.stage:
                latency[event.stage] = latency.get(event.stage, 0.0) + float(
                    event.payload.get("latency_ms", 0.0)
                )
            input_tokens += int(event.payload.get("input_tokens", 0))
            output_tokens += int(event.payload.get("output_tokens", 0))
        return RunInspection(run, events, checkpoint, latency, input_tokens, output_tokens)

    def cancel(self, context: ActorContext, run_id: str) -> ResearchRun:
        run = self._authorized_run(context, run_id, "run.cancel")
        if run.status in {"completed", "failed", "cancelled", "timed_out"}:
            return run
        now = datetime.now(timezone.utc)
        updated = run.model_copy(update={"cancel_requested_at": now})
        self.store.put(updated)
        self._event(updated, "cancel_requested", "run:cancel")
        return updated

    def _execute(
        self,
        context: ActorContext,
        run: ResearchRun,
        *,
        request: ResearchRequest | None,
        resume: bool,
    ) -> RunInspection:
        now = datetime.now(timezone.utc)
        active = run.model_copy(update={
            "status": "running",
            "started_at": run.started_at or now,
        })
        self.store.put(active)
        app = self._application(context, active)
        try:
            if resume:
                state = app.workflow.resume(thread_id=active.thread_id)
            else:
                assert request is not None
                state = app.run(
                    research_id=active.run_id,
                    question=active.question,
                    thread_id=active.thread_id,
                    budget=request.budget,
                    specfact_ids=active.specfact_ids,
                    memory_ids=active.memory_ids,
                )
        except RuntimeStopped as exc:
            status = "cancelled" if exc.reason == "cancelled" else "timed_out"
            terminal = active.model_copy(update={
                "status": status,
                "stop_reason": exc.reason,
                "finished_at": datetime.now(timezone.utc),
            })
        except Exception:
            terminal = active.model_copy(update={
                "status": "failed",
                "stop_reason": "failed",
                "finished_at": datetime.now(timezone.utc),
            })
            self.store.put(terminal)
            self._event(terminal, "run_stopped", "run:failed", payload={"reason": "failed"})
            raise
        else:
            reason = state.get("stop_reason") or "completed"
            status = (
                "cancelled" if reason == "cancelled" else
                "timed_out" if reason == "elapsed_time_budget" else
                "completed" if reason == "completed" else "failed"
            )
            terminal = active.model_copy(update={
                "status": status,
                "stop_reason": reason if reason in {
                    "completed", "cancelled", "elapsed_time_budget", "iteration_budget",
                    "tool_budget", "replan_budget", "insufficient_evidence",
                } else "failed",
                "finished_at": datetime.now(timezone.utc),
            })
            if status == "completed" and self.proposal_writer is not None:
                proposal_ids = self.proposal_writer(context, terminal, state)
                terminal = terminal.model_copy(update={
                    "specfact_proposal_ids": tuple(sorted(set(proposal_ids)))
                })
                app.workflow.update(
                    thread_id=terminal.thread_id,
                    update={"specfact_proposal_ids": terminal.specfact_proposal_ids},
                )
        self.store.put(terminal)
        self._event(
            terminal, "run_stopped", f"run:stopped:{terminal.status}",
            payload={"reason": terminal.stop_reason},
        )
        return self.inspect(context, terminal.run_id)

    def _application(self, context: ActorContext, run: ResearchRun) -> ResearchApplication:
        research_context: tuple[dict[str, object], ...] = ()
        if self.context_loader is not None:
            research_context = self.context_loader(context, run)
        elif self.context_snapshot is not None:
            _, _, research_context = self.context_snapshot(context, run.question, run.release)
        research_context = (*research_context, {
            "kind": "run_configuration",
            "release": run.release,
            "answer_language": run.answer_language,
            "detail_level": run.detail_level,
        })
        return self.application_factory(
            run.release, self._stop_requested, self._node_event(run), research_context
        )

    def _stop_requested(self, run_id: str) -> str | None:
        run = self.store.get(run_id)
        if run is None:
            return "cancelled"
        if run.cancel_requested_at is not None:
            return "cancelled"
        if run.deadline_at is not None and datetime.now(timezone.utc) >= run.deadline_at:
            return "elapsed_time_budget"
        return None

    def _node_event(self, run: ResearchRun) -> Callable[..., None]:
        def emit(name: str, state: ResearchState, update: dict[str, object], latency_ms: float) -> None:
            event_type = {
                "researcher": "task_dispatched",
                "source_validator": "source_validated",
                "sufficiency_judge": "judge_completed",
                "replanner": "replanned",
                "synthesizer": "synthesized",
            }.get(name, "node_transition")
            operation_key = (
                f"node:{name}:iteration:{state['iteration']}:"
                f"calls:{len(state['tool_calls'])}:gaps:{len(state['gaps'])}"
            )
            old_receipts = len(state.get("model_receipts", ()))
            new_receipts = tuple(update.get("model_receipts", ()))[old_receipts:]
            for receipt in new_receipts:
                self._event(
                    run,
                    "model_receipt",
                    f"receipt:{receipt.get('response_sha256', '')}",
                    stage=name,
                    payload={
                        "model": receipt.get("model"),
                        "purpose": receipt.get("purpose"),
                        "prompt_version": receipt.get("prompt_version"),
                        "request_sha256": receipt.get("request_sha256"),
                        "response_sha256": receipt.get("response_sha256"),
                        "latency_ms": receipt.get("latency_ms", 0),
                        "input_tokens": receipt.get("input_tokens") or 0,
                        "output_tokens": receipt.get("output_tokens") or 0,
                    },
                )
            old_calls = len(state.get("tool_calls", ()))
            for call in tuple(update.get("tool_calls", ()))[old_calls:]:
                self._event(
                    run, "tool_call", f"tool:{call.call_id}",
                    stage="researcher", task_id=call.task_id,
                    payload={
                        "tool_name": call.tool_name,
                        "status": call.status,
                        "evidence_ids": list(call.evidence_ids),
                    },
                )
            old_selections = len(state.get("researcher_evidence_selections", ()))
            for selection in tuple(update.get("researcher_evidence_selections", ()))[old_selections:]:
                self._event(
                    run, "evidence_selected", f"selection:{selection.selection_id}",
                    stage="researcher", task_id=selection.task_id,
                    payload={"evidence_ids": list(selection.retained_evidence_ids)},
                )
            old_gaps = {gap.gap_id for gap in state.get("gaps", ())}
            for gap in update.get("gaps", ()):
                if gap.gap_id not in old_gaps:
                    self._event(
                        run, "gap_created", f"gap:{gap.gap_id}",
                        stage="sufficiency_judge", task_id=gap.task_id,
                        payload={"kind": gap.kind, "description": gap.description},
                    )
            self._event(run, event_type, operation_key, stage=name, payload={
                "latency_ms": round(latency_ms, 3),
                "tool_calls": len(update.get("tool_calls", ())),
                "evidence": len(update.get("candidate_evidence", ())),
                "stop_reason": update.get("stop_reason"),
            })
        return emit

    def _event(
        self,
        run: ResearchRun,
        event_type: str,
        operation_key: str,
        *,
        stage: str | None = None,
        task_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> RunEvent:
        sequence = len(self.store.events(run.run_id)) + 1
        event = RunEvent(
            event_id="EVT-" + uuid4().hex.upper(),
            run_id=run.run_id,
            workspace_id=run.workspace_id,
            sequence=sequence,
            event_type=event_type,
            occurred_at=datetime.now(timezone.utc),
            stage=stage,
            task_id=task_id,
            operation_key=operation_key,
            payload=payload or {},
        )
        return self.store.append_event(event)

    def _authorized_run(
        self, context: ActorContext, run_id: str, permission: str,
    ) -> ResearchRun:
        run = self.store.get(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        self.policy.require(
            context, permission, workspace_id=run.workspace_id,
            owner_user_id=run.owner_user_id,
        )
        return run


__all__ = ["RunHarness", "RunInspection", "RunStore"]

"""Authorized durable lifecycle for research runs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from specrag.access import AccessPolicy
from specrag.models import (
    ActorContext,
    ResearchInputSnapshot,
    ResearchOutcome,
    ResearchRequest,
    ResearchRun,
    SpecificationScope,
)

from .application import ResearchApplication, RuntimeStopped
from .failure import failure_metadata
from .lifecycle import reset_for_resume, retry_request
from .outcomes import build_research_outcome
from .run_contracts import (
    ApplicationFactory,
    ContextLoader,
    ContextSnapshot,
    ContextWriter,
    CorpusBindingWriter,
    CorpusSnapshot,
    ProposalWriter,
    ResultLookup,
    RunInspection,
    RunStore,
)
from .run_events import RunEventRecorder
from .run_access import RunAccessMixin
from .state import ResearchState


class RunHarness(RunAccessMixin):
    def __init__(
        self,
        store: RunStore,
        application_factory: ApplicationFactory,
        *,
        policy: AccessPolicy | None = None,
        context_snapshot: ContextSnapshot | None = None,
        context_loader: ContextLoader | None = None,
        proposal_writer: ProposalWriter | None = None,
        corpus_snapshot: CorpusSnapshot | None = None,
        corpus_binding_writer: CorpusBindingWriter | None = None,
        context_writer: ContextWriter | None = None,
        result_lookup: ResultLookup | None = None,
    ) -> None:
        self.store = store
        self.application_factory = application_factory
        self.policy = policy or AccessPolicy()
        self.context_snapshot = context_snapshot
        self.context_loader = context_loader
        self.proposal_writer = proposal_writer
        self.corpus_snapshot = corpus_snapshot
        self.corpus_binding_writer = corpus_binding_writer
        self.context_writer = context_writer
        self.result_lookup = result_lookup
        self.event_recorder = RunEventRecorder(store)
    def start(self, context: ActorContext, request: ResearchRequest) -> RunInspection:
        run = self._prepare(context, request)
        return self._execute(context, run, request=request, resume=False)

    def submit(self, context: ActorContext, request: ResearchRequest) -> RunInspection:
        """Persist a queued Run without executing its research graph."""

        run = self._prepare(context, request)
        return self._queued_inspection(run)

    def execute_queued(self, context: ActorContext, run_id: str) -> RunInspection:
        """Execute one previously submitted Run from a trusted worker."""

        run = self.store.get(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if run.workspace_id != context.workspace_id or run.owner_user_id != context.user_id:
            raise PermissionError("queued Run context mismatch")
        if run.status in {"completed", "cancelled", "timed_out"}:
            return self._queued_inspection(run)
        if run.status == "running":
            return self.resume(context, run_id)
        if run.cancel_requested_at is not None:
            return self._finish_cancelled(run)
        return self._execute(
            context, run, request=self._request_from_run(run), resume=False
        )

    def recoverable_runs(self) -> tuple[ResearchRun, ...]:
        """Return persisted Runs that require trusted Worker recovery."""

        return self.store.list_active_runs()

    def _prepare(
        self,
        context: ActorContext,
        request: ResearchRequest,
        *,
        retry_of_run_id: str | None = None,
    ) -> ResearchRun:
        self.policy.require(context, "research.create", workspace_id=context.workspace_id)
        requested_scope = request.requested_scope or SpecificationScope(
            policy="highest_version_within_release",
            release=request.release,
        )
        input_snapshot = ResearchInputSnapshot(
            question=request.question,
            objective=request.objective or request.question,
            conditions=request.conditions,
            requested_scope=requested_scope,
            prior_results=request.prior_results,
            answer_language=request.answer_language,
            detail_level=request.detail_level,
        )
        if (
            request.conversation_id is not None
            and request.submission_operation_key is not None
        ):
            existing = self.store.get_by_submission(
                request.conversation_id, request.submission_operation_key
            )
            if existing is not None:
                if (
                    existing.input_snapshot != input_snapshot
                    or existing.origin_message_id != request.origin_message_id
                    or existing.origin_turn_id != request.origin_turn_id
                    or existing.submitted_user_sequence
                    != request.submitted_user_sequence
                ):
                    raise ValueError(
                        "submission operation key was reused with different input"
                    )
                return existing
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
            budget=request.budget,
            memory_ids=request.configuration_memory_ids,
            development_question_id=request.development_question_id,
            retry_of_run_id=retry_of_run_id,
            conversation_id=request.conversation_id,
            origin_message_id=request.origin_message_id,
            origin_turn_id=request.origin_turn_id,
            submitted_user_sequence=request.submitted_user_sequence,
            submission_operation_key=request.submission_operation_key,
            input_snapshot=input_snapshot,
            created_at=now,
            deadline_at=now + timedelta(seconds=request.elapsed_time_seconds),
        )
        if self.corpus_snapshot is not None:
            versions = self.corpus_snapshot(requested_scope)
            if not versions:
                raise ValueError(
                    "no active specification versions satisfy the requested scope"
                )
            run = run.model_copy(update={
                "corpus_version_policy": (
                    "exact_version" if len(versions) == 1 else "mixed_versions"
                ),
                "corpus_release": None,
                "corpus_versions": versions,
            })
        elif requested_scope.versions:
            run = run.model_copy(update={
                "corpus_version_policy": requested_scope.policy,
                "corpus_release": None,
                "corpus_versions": requested_scope.versions,
            })
        else:
            run = run.model_copy(update={
                "corpus_version_policy": requested_scope.policy,
                "corpus_release": requested_scope.release,
            })
        if self.context_snapshot is not None:
            facts, memories, _ = self.context_snapshot(
                context, request.question, request.release
            )
            run = run.model_copy(update={
                "specfact_ids": facts,
                "memory_ids": tuple(sorted(set((*run.memory_ids, *memories)))),
            })
        self.store.put(run)
        if self.context_writer is not None:
            self.context_writer(context, run)
        if self.corpus_binding_writer is not None:
            self.corpus_binding_writer(run)
        self.event_recorder.append(run, "run_started", "run:start", payload={
            "release": run.release,
            "corpus_version_policy": run.corpus_version_policy,
            "corpus_versions": [
                item.model_dump(mode="json") for item in run.corpus_versions
            ],
            "retry_of_run_id": run.retry_of_run_id,
        })
        return run

    def _queued_inspection(self, run: ResearchRun) -> RunInspection:
        return RunInspection(
            run=run,
            events=self.store.events(run.run_id),
            checkpoint=None,
            stage_latency_ms={},
            input_tokens=0,
            output_tokens=0,
            result_id=(self.result_lookup(run.run_id) if self.result_lookup else None),
        )

    @staticmethod
    def _request_from_run(run: ResearchRun) -> ResearchRequest:
        remaining = 900
        if run.deadline_at is not None:
            remaining = max(
                1, int((run.deadline_at - datetime.now(timezone.utc)).total_seconds())
            )
        return ResearchRequest(
            question=run.question,
            objective=(
                run.input_snapshot.objective if run.input_snapshot else run.question
            ),
            conditions=(run.input_snapshot.conditions if run.input_snapshot else ()),
            requested_scope=(
                run.input_snapshot.requested_scope if run.input_snapshot else None
            ),
            prior_results=(
                run.input_snapshot.prior_results if run.input_snapshot else ()
            ),
            release=run.release,
            answer_language=run.answer_language,
            detail_level=run.detail_level,
            configuration_memory_ids=run.memory_ids,
            budget=run.budget,
            elapsed_time_seconds=remaining,
            development_question_id=run.development_question_id,
            conversation_id=run.conversation_id,
            origin_message_id=run.origin_message_id,
            origin_turn_id=run.origin_turn_id,
            submitted_user_sequence=run.submitted_user_sequence,
            submission_operation_key=run.submission_operation_key,
        )

    def _finish_cancelled(self, run: ResearchRun) -> RunInspection:
        now = datetime.now(timezone.utc)
        terminal = run.model_copy(update={
            "status": "cancelled",
            "stop_reason": "cancelled",
            "started_at": run.started_at or now,
            "finished_at": now,
        })
        event = self.event_recorder.build(
            terminal,
            "run_stopped",
            f"run:stopped:cancelled:attempt:{terminal.run_attempt}",
            payload={"reason": "cancelled"},
        )
        self.store.finalize(terminal, event, None)
        return RunInspection(
            run=terminal,
            events=self.store.events(terminal.run_id),
            checkpoint=None,
            stage_latency_ms={},
            input_tokens=0,
            output_tokens=0,
            result_id=(
                self.result_lookup(terminal.run_id) if self.result_lookup else None
            ),
        )

    def queue_resume(
        self,
        context: ActorContext,
        run_id: str,
        *,
        elapsed_time_seconds: int = 900,
    ) -> RunInspection:
        run = self._authorized_run(context, run_id, "run.content.read")
        resumed_before = any(
            event.event_type == "resumed" for event in self.store.events(run_id)
        )
        if run.status in {"queued", "running"} and resumed_before:
            return self.inspect(context, run_id)
        if run.status not in {"failed", "cancelled", "timed_out"}:
            raise ValueError("only failed, cancelled, or timed-out Runs can resume")
        inspection = self.inspect(context, run_id)
        if not inspection.checkpoint:
            raise ValueError("Run has no durable checkpoint to resume")
        queued = reset_for_resume(
            run, elapsed_time_seconds=elapsed_time_seconds
        )
        self.store.put(queued)
        attempt = run.run_attempt + 1
        self.event_recorder.append(
            queued,
            "resumed",
            f"run:resume:{attempt}",
            payload={"status": "queued", "elapsed_time_seconds": elapsed_time_seconds},
        )
        return self._queued_inspection(queued)

    def execute_resume(
        self, context: ActorContext, run_id: str,
    ) -> RunInspection:
        run = self._authorized_run(context, run_id, "run.content.read")
        if run.status in {"completed", "failed", "cancelled", "timed_out"}:
            return self.inspect(context, run_id)
        if run.status == "running":
            return self.resume(context, run_id)
        if run.cancel_requested_at is not None:
            return self._finish_cancelled(run)
        if run.status != "queued" or not any(
            event.event_type == "resumed" for event in self.store.events(run_id)
        ):
            raise ValueError("Run is not queued for resume")
        return self._execute(context, run, request=None, resume=True)

    def resume(
        self,
        context: ActorContext,
        run_id: str,
        *,
        elapsed_time_seconds: int = 900,
    ) -> RunInspection:
        run = self._authorized_run(context, run_id, "run.content.read")
        if run.cancel_requested_at is not None and run.status in {"queued", "running"}:
            return self._finish_cancelled(run)
        if run.status == "completed":
            return self.inspect(context, run_id)
        if run.status in {"failed", "cancelled", "timed_out"}:
            self.queue_resume(
                context, run_id, elapsed_time_seconds=elapsed_time_seconds
            )
            return self.execute_resume(context, run_id)
        if run.status == "queued":
            return self.execute_resume(context, run_id)
        attempt = run.run_attempt + 1
        resumed = run.model_copy(update={"run_attempt": attempt})
        self.store.put(resumed)
        self.event_recorder.append(resumed, "resumed", f"run:resume:{attempt}")
        return self._execute(context, resumed, request=None, resume=True)

    def retry(
        self,
        context: ActorContext,
        run_id: str,
        *,
        elapsed_time_seconds: int = 900,
    ) -> RunInspection:
        source = self._authorized_run(context, run_id, "run.content.read")
        if source.status not in {"completed", "failed", "cancelled", "timed_out"}:
            raise ValueError("only terminal Runs can be retried")
        request = retry_request(
            source, elapsed_time_seconds=elapsed_time_seconds
        )
        run = self._prepare(context, request, retry_of_run_id=source.run_id)
        return self._queued_inspection(run)

    def inspect(self, context: ActorContext, run_id: str) -> RunInspection:
        run = self._authorized_run(context, run_id, "run.content.read")
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
        return RunInspection(
            run,
            events,
            checkpoint,
            latency,
            input_tokens,
            output_tokens,
            self.result_lookup(run_id) if self.result_lookup else None,
        )

    def cancel(self, context: ActorContext, run_id: str) -> ResearchRun:
        run = self._authorized_run(context, run_id, "run.lifecycle.manage")
        if run.status in {"completed", "failed", "cancelled", "timed_out"}:
            return run
        now = datetime.now(timezone.utc)
        updated = run.model_copy(update={"cancel_requested_at": now})
        self.store.put(updated)
        self.event_recorder.append(updated, "cancel_requested", "run:cancel")
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
        except Exception as exc:  # noqa: BLE001 - persisted metadata is sanitized
            failure = failure_metadata(exc)
            terminal = active.model_copy(update={
                "status": "failed",
                "stop_reason": "failed",
                "finished_at": datetime.now(timezone.utc),
                **failure,
            })
            event = self.event_recorder.build(
                terminal,
                "run_stopped",
                f"run:stopped:failed:attempt:{terminal.run_attempt}",
                payload={
                    "reason": "failed",
                    "failure_kind": terminal.failure_kind,
                    "retryable": terminal.failure_retryable,
                    "attempt_count": terminal.failure_attempt_count,
                    "status_code": terminal.failure_status_code,
                },
            )
            self.store.finalize(terminal, event, None)
            raise
        else:
            reason = state.get("stop_reason") or "completed"
            status = (
                "cancelled" if reason == "cancelled" else
                "timed_out" if reason == "elapsed_time_budget" else
                "failed" if reason == "unrecoverable_error" else
                "completed"
            )
            terminal = active.model_copy(update={
                "status": status,
                "stop_reason": reason if reason in {
                    "completed", "cancelled", "elapsed_time_budget", "iteration_budget",
                    "tool_budget", "replan_budget", "insufficient_evidence",
                } else "failed",
                "finished_at": datetime.now(timezone.utc),
            })
            if (
                status == "completed"
                and state.get("synthesis") is not None
                and self.proposal_writer is not None
            ):
                proposal_ids = self.proposal_writer(context, terminal, state)
                terminal = terminal.model_copy(update={
                    "specfact_proposal_ids": tuple(sorted(set(proposal_ids)))
                })
                app.workflow.update(
                    thread_id=terminal.thread_id,
                    update={"specfact_proposal_ids": terminal.specfact_proposal_ids},
                )
        outcome = (
            build_research_outcome(terminal, state)
            if terminal.status == "completed" else None
        )
        event = self.event_recorder.build(
            terminal,
            "run_stopped",
            f"run:stopped:{terminal.status}:attempt:{terminal.run_attempt}",
            payload={"reason": terminal.stop_reason},
        )
        self.store.finalize(terminal, event, outcome)
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
            "objective": (
                run.input_snapshot.objective if run.input_snapshot else run.question
            ),
            "conditions": list(
                run.input_snapshot.conditions if run.input_snapshot else ()
            ),
            "requested_scope": (
                run.input_snapshot.requested_scope.model_dump(mode="json")
                if run.input_snapshot else None
            ),
            "prior_results": [
                item.model_dump(mode="json")
                for item in (
                    run.input_snapshot.prior_results if run.input_snapshot else ()
                )
            ],
            "corpus_version_policy": run.corpus_version_policy,
            "corpus_versions": [
                item.model_dump(mode="json") for item in run.corpus_versions
            ],
        })
        return self.application_factory(
            run.release,
            self._stop_requested,
            self.event_recorder.node_sink(run),
            research_context,
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

__all__ = ["RunHarness", "RunInspection", "RunStore"]

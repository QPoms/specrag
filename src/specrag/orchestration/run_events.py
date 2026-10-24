"""Durable Run Event construction and workflow-node audit projection."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from specrag.models import ResearchRun, RunEvent

from .state import ResearchState


class RunEventStore(Protocol):
    def append_event(self, event: RunEvent) -> RunEvent: ...
    def events(self, run_id: str) -> tuple[RunEvent, ...]: ...


class RunEventRecorder:
    def __init__(self, store: RunEventStore) -> None:
        self.store = store

    def append(
        self,
        run: ResearchRun,
        event_type: str,
        operation_key: str,
        *,
        stage: str | None = None,
        task_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> RunEvent:
        return self.store.append_event(
            self.build(
                run,
                event_type,
                operation_key,
                stage=stage,
                task_id=task_id,
                payload=payload,
            )
        )

    def build(
        self,
        run: ResearchRun,
        event_type: str,
        operation_key: str,
        *,
        stage: str | None = None,
        task_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> RunEvent:
        return RunEvent(
            event_id="EVT-" + uuid4().hex.upper(),
            run_id=run.run_id,
            workspace_id=run.workspace_id,
            sequence=len(self.store.events(run.run_id)) + 1,
            event_type=event_type,
            occurred_at=datetime.now(timezone.utc),
            stage=stage,
            task_id=task_id,
            operation_key=operation_key,
            payload=payload or {},
        )

    def node_sink(self, run: ResearchRun) -> Callable[..., None]:
        def emit(
            name: str,
            state: ResearchState,
            update: dict[str, object],
            latency_ms: float,
        ) -> None:
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
            old_execution_ids = {
                item.execution_id for item in state.get("task_executions", ())
            }
            for execution in update.get("task_executions", ()):
                if execution.execution_id in old_execution_ids:
                    continue
                self.append(
                    run,
                    "task_dispatched",
                    f"task-execution:{execution.execution_id}",
                    stage="researcher",
                    task_id=execution.task_id,
                    payload={
                        "execution_id": execution.execution_id,
                        "attempt": execution.attempt,
                        "phase": execution.phase,
                    },
                )
            old_receipts = len(state.get("model_receipts", ()))
            new_receipts = tuple(update.get("model_receipts", ()))[old_receipts:]
            for offset, receipt in enumerate(new_receipts, start=old_receipts + 1):
                receipt_stage = str(receipt.get("stage") or name)
                self.append(
                    run,
                    "model_receipt",
                    (
                        f"receipt:{offset}:"
                        f"{receipt.get('request_sha256', '')}:"
                        f"{receipt.get('response_sha256', '')}"
                    ),
                    stage=receipt_stage,
                    payload={
                        "model": receipt.get("model"),
                        "purpose": receipt.get("purpose"),
                        "prompt_version": receipt.get("prompt_version"),
                        "request_sha256": receipt.get("request_sha256"),
                        "response_sha256": receipt.get("response_sha256"),
                        "latency_ms": receipt.get("latency_ms", 0),
                        "input_tokens": receipt.get("input_tokens"),
                        "output_tokens": receipt.get("output_tokens"),
                        "estimated_input_tokens": receipt.get("estimated_input_tokens"),
                        "input_ceiling_tokens": receipt.get("input_ceiling_tokens"),
                        "outcome": receipt.get("outcome", "completed"),
                        "failure_kind": receipt.get("failure_kind"),
                    },
                )
            if name == "fan_in":
                processed = int(state.get("researcher_results_processed", 0))
                for result in state.get("researcher_results", ())[processed:]:
                    self.append(
                        run,
                        "node_transition",
                        f"researcher-worker:{result.execution_id}",
                        stage="researcher",
                        task_id=result.task_id,
                        payload={
                            "latency_ms": round(result.latency_ms, 3),
                            "parallel_worker": True,
                            "wave": state["iteration"] + 1,
                            "execution_id": result.execution_id,
                            "execution_status": result.execution_status,
                        },
                    )
            old_calls = len(state.get("tool_calls", ()))
            for call in tuple(update.get("tool_calls", ()))[old_calls:]:
                self.append(
                    run,
                    "tool_call",
                    f"tool:{call.call_id}",
                    stage="researcher",
                    task_id=call.task_id,
                    payload={
                        "tool_name": call.tool_name,
                        "status": call.status,
                        "evidence_ids": list(call.evidence_ids),
                    },
                )
            old_selections = len(state.get("researcher_evidence_selections", ()))
            selections = tuple(update.get("researcher_evidence_selections", ()))
            for selection in selections[old_selections:]:
                self.append(
                    run,
                    "evidence_selected",
                    f"selection:{selection.selection_id}",
                    stage="researcher",
                    task_id=selection.task_id,
                    payload={"evidence_ids": list(selection.retained_evidence_ids)},
                )
            old_gaps = {gap.gap_id for gap in state.get("gaps", ())}
            for gap in update.get("gaps", ()):
                if gap.gap_id not in old_gaps:
                    self.append(
                        run,
                        "gap_created",
                        f"gap:{gap.gap_id}",
                        stage="sufficiency_judge",
                        task_id=gap.task_id,
                        payload={"kind": gap.kind, "description": gap.description},
                    )
            self.append(
                run,
                event_type,
                operation_key,
                stage=name,
                payload={
                    "latency_ms": round(latency_ms, 3),
                    "tool_calls": len(update.get("tool_calls", ())),
                    "evidence": len(update.get("candidate_evidence", ())),
                    "stop_reason": update.get("stop_reason"),
                },
            )

        return emit


__all__ = ["RunEventRecorder", "RunEventStore"]

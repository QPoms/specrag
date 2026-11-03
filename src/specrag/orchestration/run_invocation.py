"""Trace-aware invocation boundary for initial and resumed research Runs."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any
from uuid import uuid4

from specrag.models import ResearchRequest, ResearchRun

from .application import ResearchApplication
from .tracing import WorkflowTraceContext


def invoke_run_application(
    app: Any,
    run: ResearchRun,
    request: ResearchRequest | None,
    *,
    resume: bool,
) -> Any:
    """Invoke one Run while preserving trace identity across resume boundaries."""

    trace = _run_trace_context(run, resume=resume) if isinstance(
        app, ResearchApplication
    ) else None
    if resume:
        tracing = getattr(app, "tracing", None)
        activation = tracing.activate() if tracing is not None else nullcontext()
        with activation:
            options = {"trace_context": trace} if trace is not None else {}
            return app.workflow.resume(thread_id=run.thread_id, **options)

    assert request is not None
    options = {"trace_context": trace} if trace is not None else {}
    return app.run(
        research_id=run.run_id,
        question=run.question,
        thread_id=run.thread_id,
        budget=request.budget,
        specfact_ids=run.specfact_ids,
        memory_ids=run.memory_ids,
        **options,
    )


def _run_trace_context(
    run: ResearchRun,
    *,
    resume: bool,
) -> WorkflowTraceContext:
    phase = "resume" if resume else "initial"
    return WorkflowTraceContext(
        run_id=uuid4(),
        run_name=f"workflow:{run.run_id}:{phase}",
        tags=("specrag:research", "phase:workflow", f"invocation:{phase}"),
        metadata={
            "phase": "workflow",
            "local_run_id": run.run_id,
            "thread_id": run.thread_id,
            "resume": resume,
        },
    )


__all__ = ["invoke_run_application"]

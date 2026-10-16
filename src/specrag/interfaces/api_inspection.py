"""Safe HTTP projection of durable Run inspection state."""

from __future__ import annotations

from specrag.models import RunFailureResponse, RunInspectionResponse
from specrag.orchestration import RunInspection


_STOP_SUMMARIES = {
    "failed": "Research execution failed before a final answer was produced.",
    "elapsed_time_budget": "The Run reached its elapsed-time limit.",
    "iteration_budget": "The Run used its configured iteration allowance.",
    "tool_budget": "The Run used its configured retrieval-tool allowance.",
    "replan_budget": "The Run used its configured re-planning allowance.",
    "insufficient_evidence": "The available evidence was not sufficient for a grounded answer.",
    "cancelled": "Cancellation was requested before the Run completed.",
}


def inspection_response(value: RunInspection) -> RunInspectionResponse:
    checkpoint = value.checkpoint or {}

    def read(item, key: str, default=None):
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    candidates = tuple(read(checkpoint, "candidate_evidence", ()) or ())
    validations = tuple(read(checkpoint, "validated_evidence", ()) or ())
    supported_ids = {
        read(item, "evidence_id")
        for item in validations
        if read(item, "source_valid", False)
        and read(item, "supports_target", False)
        and read(item, "validation_status") == "supported"
    }
    synthesis = read(checkpoint, "synthesis")
    claims = tuple(read(synthesis, "claims", ()) or ())
    claims = tuple(
        claim for claim in claims
        if all(
            evidence_id in supported_ids
            for evidence_id in read(claim, "evidence_ids", ())
        )
    )
    claim_evidence_ids = {
        evidence_id
        for claim in claims
        for evidence_id in read(claim, "evidence_ids", ())
    }
    evidence = tuple(
        item
        for item in candidates
        if read(item, "evidence_id") in supported_ids
        and read(item, "evidence_id") in claim_evidence_ids
    )
    answer = read(synthesis, "answer") or read(checkpoint, "final_answer")
    failure = _failure(value)
    return RunInspectionResponse(
        run=value.run,
        events=value.events,
        stage_latency_ms=value.stage_latency_ms,
        input_tokens=value.input_tokens,
        output_tokens=value.output_tokens,
        checkpoint_available=value.checkpoint is not None,
        answer=answer,
        claims=claims,
        evidence=evidence,
        failure=failure,
    )


def _failure(value: RunInspection) -> RunFailureResponse | None:
    run = value.run
    if run.status not in {"failed", "cancelled", "timed_out"}:
        return None
    reason = run.stop_reason or "failed"
    last_event = value.events[-1] if value.events else None
    last_stage = next(
        (event.stage for event in reversed(value.events) if event.stage),
        None,
    )
    return RunFailureResponse(
        summary=_STOP_SUMMARIES.get(reason, _STOP_SUMMARIES["failed"]),
        stop_reason=reason,
        last_stage=last_stage,
        last_event_type=last_event.event_type if last_event else None,
        last_event_at=last_event.occurred_at if last_event else None,
    )


__all__ = ["inspection_response"]

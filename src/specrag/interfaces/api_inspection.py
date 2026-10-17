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

_FAILURE_SUMMARIES = {
    "retryable_transient": "The model provider had a temporary connection failure; retry later.",
    "rate_limited": "The model provider rate-limited this request; retry later.",
    "timeout": "The model provider did not respond in time; retry later.",
    "provider_unavailable": "The model provider is temporarily unavailable; retry later.",
    "authentication_or_configuration": "The model API credentials or endpoint configuration must be checked.",
    "invalid_request": "The model API rejected the request configuration.",
    "structured_output": "The model did not return the required structured output.",
    "model_refusal": "The model refused to return the requested output.",
    "internal_error": "The service encountered an internal error.",
    "unknown": "The model API call failed for an unknown reason.",
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
    cited_ids = tuple(read(synthesis, "evidence_ids", ()) or ())
    if not cited_ids:
        cited_ids = tuple(dict.fromkeys(
            evidence_id
            for claim in tuple(read(synthesis, "claims", ()) or ())
            for evidence_id in read(claim, "evidence_ids", ())
        ))
    cited_ids = tuple(
        evidence_id for evidence_id in cited_ids if evidence_id in supported_ids
    )
    cited_id_set = set(cited_ids)
    evidence = tuple(
        item
        for item in candidates
        if read(item, "evidence_id") in supported_ids
        and read(item, "evidence_id") in cited_id_set
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
        evidence_ids=cited_ids,
        evidence=evidence,
        failure=failure,
    )


def _failure(value: RunInspection) -> RunFailureResponse | None:
    run = value.run
    if run.status not in {"failed", "cancelled", "timed_out"}:
        return None
    reason = run.stop_reason or "failed"
    kind = run.failure_kind
    last_event = value.events[-1] if value.events else None
    last_stage = next(
        (event.stage for event in reversed(value.events) if event.stage),
        None,
    )
    return RunFailureResponse(
        summary=_FAILURE_SUMMARIES.get(
            kind,
            _STOP_SUMMARIES.get(reason, _STOP_SUMMARIES["failed"]),
        ),
        stop_reason=reason,
        failure_kind=kind,
        retryable=bool(run.failure_retryable),
        attempt_count=run.failure_attempt_count,
        status_code=run.failure_status_code,
        last_stage=last_stage,
        last_event_type=last_event.event_type if last_event else None,
        last_event_at=last_event.occurred_at if last_event else None,
    )


__all__ = ["inspection_response"]

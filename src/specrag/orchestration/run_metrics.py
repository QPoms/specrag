"""Deterministic inspection metrics derived from durable Run Events."""

from __future__ import annotations

from specrag.models import RunEvent


def aggregate_run_event_metrics(
    events: tuple[RunEvent, ...],
) -> tuple[dict[str, float], int | None, int | None]:
    """Return critical-path stage latency and complete provider Token totals."""

    latency: dict[str, float] = {}
    parallel_researcher_waves: dict[int, float] = {}
    input_tokens = 0
    output_tokens = 0
    receipt_count = 0
    token_usage_complete = True
    for event in events:
        if event.stage and event.event_type != "model_receipt":
            event_latency = float(event.payload.get("latency_ms", 0.0))
            if (
                event.stage == "researcher"
                and event.payload.get("parallel_worker") is True
            ):
                wave = int(event.payload.get("wave", 0))
                parallel_researcher_waves[wave] = max(
                    parallel_researcher_waves.get(wave, 0.0),
                    event_latency,
                )
                continue
            latency[event.stage] = latency.get(event.stage, 0.0) + event_latency
        if event.event_type == "model_receipt":
            receipt_count += 1
            raw_input = event.payload.get("input_tokens")
            raw_output = event.payload.get("output_tokens")
            if not isinstance(raw_input, int) or not isinstance(raw_output, int):
                token_usage_complete = False
            else:
                input_tokens += raw_input
                output_tokens += raw_output
    if parallel_researcher_waves:
        latency["researcher"] = latency.get("researcher", 0.0) + sum(
            parallel_researcher_waves.values()
        )
    reported_input_tokens = (
        input_tokens if receipt_count and token_usage_complete else None
    )
    reported_output_tokens = (
        output_tokens if receipt_count and token_usage_complete else None
    )
    return latency, reported_input_tokens, reported_output_tokens


__all__ = ["aggregate_run_event_metrics"]

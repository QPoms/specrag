"""Pure Run lifecycle transitions shared by Harness control paths."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from specrag.models import ResearchRequest, ResearchRun, RunEvent


def resume_attempt(events: tuple[RunEvent, ...]) -> int:
    return sum(event.event_type == "resumed" for event in events)


def reset_for_resume(
    run: ResearchRun, *, elapsed_time_seconds: int,
) -> ResearchRun:
    now = datetime.now(timezone.utc)
    return ResearchRun.model_validate({
        **run.model_dump(),
        "status": "queued",
        "stop_reason": None,
        "failure_kind": None,
        "failure_retryable": None,
        "failure_attempt_count": None,
        "failure_status_code": None,
        "finished_at": None,
        "deadline_at": now + timedelta(seconds=elapsed_time_seconds),
        "cancel_requested_at": None,
    })


def retry_request(
    run: ResearchRun, *, elapsed_time_seconds: int,
) -> ResearchRequest:
    return ResearchRequest(
        question=run.question,
        release=run.release,
        answer_language=run.answer_language,
        detail_level=run.detail_level,
        budget=run.budget,
        elapsed_time_seconds=elapsed_time_seconds,
        development_question_id=run.development_question_id,
    )


__all__ = ["reset_for_resume", "resume_attempt", "retry_request"]

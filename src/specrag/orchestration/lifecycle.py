"""Pure Run lifecycle transitions shared by Harness control paths."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from specrag.models import (
    RESEARCH_FINALIZATION_RESERVE_SECONDS,
    RESEARCH_PERSISTENCE_RESERVE_SECONDS,
    ResearchRequest,
    ResearchRun,
    RunEvent,
)


def phase_deadlines(
    now: datetime,
    *,
    elapsed_time_seconds: int,
) -> tuple[datetime, datetime, datetime]:
    """Calculate ordered research, finalization, and hard Run deadlines."""

    if elapsed_time_seconds < 1:
        raise ValueError("elapsed_time_seconds must be positive")
    hard = now + timedelta(seconds=elapsed_time_seconds)
    finalization_reserve = min(
        RESEARCH_FINALIZATION_RESERVE_SECONDS,
        elapsed_time_seconds,
    )
    persistence_reserve = min(
        RESEARCH_PERSISTENCE_RESERVE_SECONDS,
        elapsed_time_seconds,
    )
    research = hard - timedelta(seconds=finalization_reserve)
    finalization = hard - timedelta(seconds=persistence_reserve)
    return research, finalization, hard


def phase_deadline_context(run: ResearchRun) -> dict[str, str | None]:
    """Project persisted phase deadlines into trusted application context."""

    return {
        "research_deadline_at": (
            run.research_deadline_at.isoformat()
            if run.research_deadline_at is not None
            else None
        ),
        "finalization_deadline_at": (
            run.finalization_deadline_at.isoformat()
            if run.finalization_deadline_at is not None
            else None
        ),
        "deadline_at": (
            run.deadline_at.isoformat() if run.deadline_at is not None else None
        ),
    }


def resume_attempt(events: tuple[RunEvent, ...]) -> int:
    return sum(event.event_type == "resumed" for event in events)


def reset_for_resume(
    run: ResearchRun,
    *,
    elapsed_time_seconds: int | None,
) -> ResearchRun:
    now = datetime.now(timezone.utc)
    deadlines = (
        phase_deadlines(now, elapsed_time_seconds=elapsed_time_seconds)
        if elapsed_time_seconds is not None
        else (None, None, None)
    )
    research_deadline, finalization_deadline, hard_deadline = deadlines
    return ResearchRun.model_validate(
        {
            **run.model_dump(),
            "status": "queued",
            "stop_reason": None,
            "failure_kind": None,
            "failure_retryable": None,
            "failure_attempt_count": None,
            "failure_status_code": None,
            "finished_at": None,
            "research_deadline_at": research_deadline,
            "finalization_deadline_at": finalization_deadline,
            "deadline_at": hard_deadline,
            "cancel_requested_at": None,
            "run_attempt": run.run_attempt + 1,
        }
    )


def retry_request(
    run: ResearchRun,
    *,
    elapsed_time_seconds: int | None,
) -> ResearchRequest:
    return ResearchRequest(
        question=run.question,
        objective=(
            run.input_snapshot.objective if run.input_snapshot else run.question
        ),
        conditions=(run.input_snapshot.conditions if run.input_snapshot else ()),
        requested_scope=(
            run.input_snapshot.requested_scope if run.input_snapshot else None
        ),
        prior_results=(run.input_snapshot.prior_results if run.input_snapshot else ()),
        release=run.release,
        answer_language=run.answer_language,
        detail_level=run.detail_level,
        budget=run.budget,
        elapsed_time_seconds=elapsed_time_seconds,
        development_question_id=run.development_question_id,
    )


def queued_request(run: ResearchRun) -> ResearchRequest:
    """Rebuild the frozen request for execution of the same queued Run."""

    remaining: int | None = None
    if run.deadline_at is not None:
        remaining = max(
            1,
            int((run.deadline_at - datetime.now(timezone.utc)).total_seconds()),
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
        prior_results=(run.input_snapshot.prior_results if run.input_snapshot else ()),
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


__all__ = [
    "phase_deadlines",
    "phase_deadline_context",
    "queued_request",
    "reset_for_resume",
    "resume_attempt",
    "retry_request",
]

"""Pure conversion from terminal workflow state to a Research Outcome."""

from __future__ import annotations

from datetime import datetime, timezone

from specrag.models import (
    ResearchInputSnapshot,
    ResearchOutcome,
    ResearchRun,
    SpecificationScope,
)

from .state import ResearchState


def build_research_outcome(
    run: ResearchRun, state: ResearchState
) -> ResearchOutcome:
    snapshot = run.input_snapshot or ResearchInputSnapshot(
        question=run.question,
        objective=run.question,
        requested_scope=SpecificationScope(
            policy=(
                "highest_version_within_release"
                if run.corpus_version_policy is None
                else run.corpus_version_policy
            ),
            release=(
                run.release
                if run.corpus_version_policy in {
                    None, "highest_version_within_release"
                } else None
            ),
            versions=run.corpus_versions,
        ),
        answer_language=run.answer_language,
        detail_level=run.detail_level,
    )
    return ResearchOutcome(
        run_id=run.run_id,
        run_attempt=run.run_attempt,
        question=snapshot.question,
        objective=snapshot.objective,
        conditions=snapshot.conditions,
        requested_scope=snapshot.requested_scope,
        corpus_versions=run.corpus_versions,
        synthesis=state.get("synthesis"),
        research_tasks=tuple(state.get("research_tasks", ())),
        candidate_evidence=tuple(state.get("candidate_evidence", ())),
        validated_evidence=tuple(state.get("validated_evidence", ())),
        assessments=tuple(state.get("sufficiency_assessments", ())),
        gaps=tuple(state.get("gaps", ())),
        stop_reason=state.get("stop_reason") or "completed",
        prior_results=snapshot.prior_results,
        created_at=run.finished_at or datetime.now(timezone.utc),
    )


__all__ = ["build_research_outcome"]

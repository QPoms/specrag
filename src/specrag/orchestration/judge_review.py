"""Deterministic construction of task-local Sufficiency Judge inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from specrag.models import (
    CandidateEvidence,
    JudgeReviewRecord,
    ResearchTask,
    SourceValidationResult,
)

from .state import ResearchState


@dataclass(frozen=True)
class JudgeReviewSet:
    task: ResearchTask
    candidates: tuple[CandidateEvidence, ...]
    source_validations: tuple[SourceValidationResult, ...]
    record: JudgeReviewRecord


def build_judge_review_sets(
    state: ResearchState,
    tasks: tuple[ResearchTask, ...],
) -> tuple[JudgeReviewSet, ...]:
    """Build stable review sets without conflating them with candidate history."""

    candidates_by_task: dict[str, list[CandidateEvidence]] = {}
    for candidate in state["candidate_evidence"]:
        candidates_by_task.setdefault(candidate.task_id, []).append(candidate)
    validation_by_id = {
        item.evidence_id: item for item in state["source_validations"]
    }
    assessment_by_task = {
        item.task_id: item for item in state["sufficiency_assessments"]
    }
    active_by_task = {
        item.task_id: item.active_evidence_ids
        for item in state.get("task_evidence_sets", ())
    }

    result: list[JudgeReviewSet] = []
    for task in tasks:
        task_candidates = tuple(candidates_by_task.get(task.task_id, ()))
        if task.status == "pending" and not task_candidates:
            continue
        latest_active = active_by_task.get(task.task_id)
        if latest_active is None:
            # Compatibility for checkpoints and direct handler callers created
            # before the latest-active state was introduced.
            latest_active = tuple(item.evidence_id for item in task_candidates)
        prior_selected = (
            assessment_by_task[task.task_id].supporting_evidence_ids
            if task.task_id in assessment_by_task
            else ()
        )
        requested_ids = set((*prior_selected, *latest_active))
        review_candidates = tuple(
            item
            for item in task_candidates
            if item.evidence_id in requested_ids
            and (
                (validation := validation_by_id.get(item.evidence_id)) is not None
                and validation.valid
            )
        )
        review_validations = tuple(
            validation_by_id[item.evidence_id] for item in review_candidates
        )
        payload = {
            "original_question": state["user_question"],
            "task_id": task.task_id,
            "task_question": task.question,
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "document_sha256": item.document_sha256,
                    "source_path": item.source_path,
                    "line_start": item.line_start,
                    "line_end": item.line_end,
                }
                for item in review_candidates
            ],
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        result.append(JudgeReviewSet(
            task=task,
            candidates=review_candidates,
            source_validations=review_validations,
            record=JudgeReviewRecord(
                task_id=task.task_id,
                review_sha256=fingerprint,
                evidence_ids=tuple(
                    item.evidence_id for item in review_candidates
                ),
                iteration=state["iteration"],
            ),
        ))
    return tuple(result)


__all__ = ["JudgeReviewSet", "build_judge_review_sets"]

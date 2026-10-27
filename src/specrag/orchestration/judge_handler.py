"""Parallel, incremental orchestration adapter for evidence sufficiency."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

from specrag.evidence import (
    EvidenceSufficiencyBatch,
    EvidenceSufficiencyService,
    publication_decision,
)
from specrag.models import ValidatedEvidence

from .judge_review import JudgeReviewSet, build_judge_review_sets
from .state import ResearchState


@dataclass(frozen=True)
class EvidenceSufficiencyHandler:
    """Judge task-local review sets concurrently and merge deterministically."""

    service: EvidenceSufficiencyService
    max_parallel_tasks: int = 4
    receipt_capture: Callable[
        [], AbstractContextManager[list[Any]]
    ] | None = None
    receipt_recorder: Callable[[Any], None] | None = None

    def __call__(self, state: ResearchState) -> dict[str, object]:
        if self.max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be positive")
        assessment_by_task = {
            item.task_id: item for item in state["sufficiency_assessments"]
        }
        completed_task_ids = {
            task.task_id
            for task in state["research_tasks"]
            if task.status == "completed"
            and (assessment := assessment_by_task.get(task.task_id)) is not None
            and assessment.status == "sufficient"
        }
        active_tasks = tuple(
            task
            for task in state["research_tasks"]
            if task.task_id not in completed_task_ids
        )
        review_sets = build_judge_review_sets(state, active_tasks)
        previous_review_by_task = {
            item.task_id: item for item in state.get("judge_review_records", ())
        }
        skipped = tuple(
            item.task.task_id
            for item in review_sets
            if item.task.task_id in assessment_by_task
            and (
                previous := previous_review_by_task.get(item.task.task_id)
            ) is not None
            and previous.review_sha256 == item.record.review_sha256
        )
        skipped_ids = set(skipped)
        pending_reviews = tuple(
            item for item in review_sets if item.task.task_id not in skipped_ids
        )
        assessed = self._assess_reviews(
            pending_reviews,
            original_question=state["user_question"],
        )

        updated_task_by_id = {
            item.task_id: item for item in state["research_tasks"]
        }
        updated_assessment_by_task = dict(assessment_by_task)
        validated_by_id = {
            item.evidence_id: item for item in state["validated_evidence"]
        }
        reassessed_task_ids: set[str] = set()
        new_gaps = []
        receipts = list(state["model_receipts"])
        for batch, batch_receipts in assessed:
            reassessed_task_ids.update(item.task_id for item in batch.tasks)
            updated_task_by_id.update(
                {item.task_id: item for item in batch.tasks}
            )
            updated_assessment_by_task.update(
                {item.task_id: item for item in batch.assessments}
            )
            validated_by_id.update(
                {item.evidence_id: item for item in batch.validated_evidence}
            )
            new_gaps.extend(batch.gaps)
            receipts.extend(batch_receipts)

        source_validation_by_id = {
            item.evidence_id: item for item in state["source_validations"]
        }
        for candidate in state["candidate_evidence"]:
            if candidate.evidence_id in validated_by_id:
                continue
            validation = source_validation_by_id.get(candidate.evidence_id)
            if validation is None:
                continue
            if validation.valid:
                rationale = (
                    "Not included in the latest Researcher-active or prior "
                    "Judge-selected review set."
                )
            else:
                rationale = (
                    "; ".join(validation.errors)
                    or "Source provenance validation failed."
                )
            validated_by_id[candidate.evidence_id] = ValidatedEvidence(
                evidence_id=candidate.evidence_id,
                task_id=candidate.task_id,
                source_valid=validation.valid,
                supports_target=False,
                validation_status="rejected",
                rationale=rationale,
            )

        tasks = tuple(
            updated_task_by_id[task.task_id] for task in state["research_tasks"]
        )
        review_by_task = dict(previous_review_by_task)
        review_by_task.update({item.task.task_id: item.record for item in review_sets})
        preserved_gaps = tuple(
            gap
            for gap in state["gaps"]
            if gap.task_id not in reassessed_task_ids
            and gap.task_id not in completed_task_ids
        )
        ordered_validated = tuple(
            validated_by_id[candidate.evidence_id]
            for candidate in state["candidate_evidence"]
            if candidate.evidence_id in validated_by_id
        )
        ordered_assessments = tuple(
            updated_assessment_by_task[task.task_id]
            for task in tasks
            if task.task_id in updated_assessment_by_task
        )
        decision = publication_decision(
            tasks,
            ordered_validated,
            ordered_assessments,
        )
        return {
            "research_tasks": tasks,
            "validated_evidence": ordered_validated,
            "sufficiency_assessments": ordered_assessments,
            "gaps": (*preserved_gaps, *new_gaps),
            "judge_review_records": tuple(
                review_by_task[task.task_id]
                for task in tasks
                if task.task_id in review_by_task
            ),
            "judge_skipped_task_ids": skipped,
            "model_receipts": tuple(receipts),
            "publication_path": decision.path,
            "evidence_sufficient": bool(tasks)
            and all(task.status == "completed" for task in tasks),
        }

    def _assess_reviews(
        self,
        reviews: tuple[JudgeReviewSet, ...],
        *,
        original_question: str,
    ) -> tuple[
        tuple[EvidenceSufficiencyBatch, tuple[dict[str, object], ...]], ...
    ]:
        if not reviews:
            return ()

        def assess(review: JudgeReviewSet) -> tuple[
            EvidenceSufficiencyBatch | Exception,
            tuple[dict[str, object], ...],
            tuple[Any, ...],
        ]:
            capture = (
                self.receipt_capture()
                if self.receipt_capture is not None
                else nullcontext([])
            )
            try:
                with capture as receipts:
                    outcome: EvidenceSufficiencyBatch | Exception = (
                        self.service.evaluate(
                            (review.task,),
                            review.candidates,
                            review.source_validations,
                            original_question=original_question,
                        )
                    )
            except Exception as exc:  # preserve failed receipts across threads
                outcome = exc
            receipt_tuple = tuple(receipts)
            payloads = tuple(
                {
                    **receipt.model_dump(mode="json"),
                    "stage": "sufficiency_judge",
                }
                if hasattr(receipt, "model_dump")
                else {**dict(receipt), "stage": "sufficiency_judge"}
                for receipt in receipt_tuple
            )
            return outcome, payloads, receipt_tuple

        with ThreadPoolExecutor(
            max_workers=min(self.max_parallel_tasks, len(reviews)),
            thread_name_prefix="specrag-judge",
        ) as executor:
            outcomes = tuple(executor.map(assess, reviews))
        assessed: list[
            tuple[EvidenceSufficiencyBatch, tuple[dict[str, object], ...]]
        ] = []
        for outcome, payloads, captured_receipts in outcomes:
            if self.receipt_recorder is not None:
                for receipt in captured_receipts:
                    self.receipt_recorder(receipt)
            if isinstance(outcome, Exception):
                raise outcome
            assessed.append((outcome, payloads))
        return tuple(assessed)


__all__ = ["EvidenceSufficiencyHandler"]

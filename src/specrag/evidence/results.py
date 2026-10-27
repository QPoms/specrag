"""Deterministic materialization of validated Research Outcomes."""

from __future__ import annotations

import hashlib
import json

from pydantic_core import to_jsonable_python

from specrag.models import (
    EvidenceStatus,
    ResearchOutcome,
    ResearchResult,
    ResearchResultEvidence,
    ResearchRun,
)


def _canonical(value: object) -> str:
    return json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class ResearchResultMaterializer:
    """Build one safe immutable Result without consulting a checkpoint."""

    def materialize(
        self,
        run: ResearchRun,
        outcome: ResearchOutcome,
        *,
        supersedes_result_id: str | None = None,
    ) -> ResearchResult:
        if run.status != "completed":
            raise ValueError("final Research Result requires a completed Run")
        if outcome.run_id != run.run_id or outcome.run_attempt != run.run_attempt:
            raise ValueError("Research Outcome does not match the Run attempt")

        validations = {
            item.evidence_id: item for item in outcome.validated_evidence
        }
        copied = tuple(
            ResearchResultEvidence(
                source=candidate,
                validation=validations[candidate.evidence_id],
            )
            for candidate in outcome.candidate_evidence
            if candidate.evidence_id in validations
            and validations[candidate.evidence_id].source_valid
        )
        copied_by_id = {item.source.evidence_id: item for item in copied}
        assessments = tuple(
            item.model_copy(update={
                "supporting_evidence_ids": tuple(
                    evidence_id
                    for evidence_id in item.supporting_evidence_ids
                    if evidence_id in copied_by_id
                )
            })
            for item in outcome.assessments
        )
        synthesis = outcome.synthesis
        cited_ids: tuple[str, ...] = ()
        synthesis_is_safe = synthesis is not None
        if synthesis is not None:
            cited_ids = synthesis.evidence_ids
            synthesis_is_safe = bool(cited_ids) and all(
                evidence_id in copied_by_id
                and copied_by_id[evidence_id].validation.supports_target
                and copied_by_id[evidence_id].validation.validation_status in (
                    {"supported"}
                    if synthesis.mode == "complete"
                    else {"supported", "partial"}
                )
                for evidence_id in cited_ids
            )
        evidence_status = self._evidence_status(outcome, synthesis_is_safe)
        if synthesis is not None and synthesis_is_safe:
            answer = synthesis.answer
        else:
            cited_ids = ()
            answer = self._safe_answer(evidence_status)
        unresolved = tuple(dict.fromkeys(
            gap.description for gap in outcome.gaps if not gap.resolved
        ))
        outcome_hash = _digest(outcome)
        result_id = "RESULT-" + hashlib.sha256(
            f"{run.run_id}:{run.run_attempt}:final".encode("utf-8")
        ).hexdigest()[:32].upper()
        values = {
            "result_id": result_id,
            "run_id": run.run_id,
            "run_attempt": run.run_attempt,
            "revision": run.run_attempt + 1,
            "workspace_id": run.workspace_id,
            "owner_user_id": run.owner_user_id,
            "conversation_id": run.conversation_id,
            "publication_kind": "final",
            "evidence_status": evidence_status,
            "answer": answer,
            "evidence": copied,
            "evidence_ids": cited_ids,
            "research_tasks": outcome.research_tasks,
            "task_assessments": assessments,
            "conditions": outcome.conditions,
            "requested_scope": outcome.requested_scope,
            "corpus_versions": outcome.corpus_versions,
            "unresolved_questions": unresolved,
            "gaps": outcome.gaps,
            "prior_results": outcome.prior_results,
            "source_outcome_sha256": outcome_hash,
            "supersedes_result_id": supersedes_result_id,
            "created_at": outcome.created_at,
        }
        content_hash = _digest(values)
        return ResearchResult.model_validate({
            **values,
            "content_sha256": content_hash,
        })

    @staticmethod
    def _evidence_status(
        outcome: ResearchOutcome, synthesis_is_safe: bool
    ) -> EvidenceStatus:
        statuses = {item.status for item in outcome.assessments}
        if "conflict" in statuses or any(
            item.validation_status == "conflict"
            for item in outcome.validated_evidence
            if item.source_valid
        ):
            return "conflict"
        if (
            synthesis_is_safe
            and statuses
            and statuses <= {"sufficient"}
            and outcome.stop_reason != "insufficient_evidence"
        ):
            return "sufficient"
        if (
            "partial" in statuses
            or any(
                item.source_valid
                and item.validation_status in {"supported", "partial"}
                for item in outcome.validated_evidence
            )
        ):
            return "partial"
        return "insufficient"

    @staticmethod
    def _safe_answer(status: EvidenceStatus) -> str:
        return {
            "conflict": (
                "The research completed, but the available specification evidence "
                "contains an unresolved conflict; no definitive conclusion is published."
            ),
            "partial": (
                "The research completed with partial specification support; the "
                "unresolved conditions must be verified before relying on a conclusion."
            ),
            "insufficient": (
                "The research completed without sufficient specification evidence to "
                "support a conclusion."
            ),
            "sufficient": (
                "The validated synthesis was unavailable, so no conclusion is published."
            ),
        }[status]


__all__ = ["ResearchResultMaterializer"]

"""Binary end-to-end task success and workflow efficiency evaluation."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .component_models import ReviewableModel
from .efficiency import EfficiencyEvaluation, extract_efficiency


FailureReason = Literal[
    "runtime_failure",
    "missing_core_conclusion",
    "material_error",
    "material_unsupported_claim",
    "missing_core_citation",
    "incorrect_core_citation",
    "version_scope_violation",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskSuccessSemanticDraft(_StrictModel):
    """Provider-neutral semantic proposal before deterministic gates/review."""

    core_conclusion_answered: bool
    no_material_error: bool
    no_material_unsupported_conclusion: bool
    core_citations_correct: bool
    rationale: str = Field(min_length=1)


class TaskSuccessReviewDecision(_StrictModel):
    """Auditable human decision over only the four semantic success gates."""

    schema_version: Literal["specrag.task-success-review/v1"] = (
        "specrag.task-success-review/v1"
    )
    question_id: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    core_conclusion_answered: bool
    no_material_error: bool
    no_material_unsupported_conclusion: bool
    core_citations_correct: bool
    rationale: str = Field(min_length=1)
    reviewed_by: str = Field(min_length=1)
    reviewed_at: datetime

    @model_validator(mode="after")
    def _timezone_aware(self) -> "TaskSuccessReviewDecision":
        if self.reviewed_at.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return self


class TaskSuccessEvaluation(ReviewableModel):
    schema_version: Literal["specrag.task-success-evaluation/v1"] = (
        "specrag.task-success-evaluation/v1"
    )
    question_id: str = Field(min_length=1)
    source_artifact: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normal_final_answer_produced: bool
    core_conclusion_answered: bool
    no_material_error: bool
    no_material_unsupported_conclusion: bool
    cited_evidence_present: bool
    citation_integrity: bool
    core_citations_correct: bool
    version_scope_compliant: bool
    task_success: bool
    failure_reasons: tuple[FailureReason, ...] = ()
    rationale: str = Field(min_length=1)
    judge_rationale: str = Field(min_length=1)
    judge_prompt_version: str | None = None
    judge_model: str | None = None
    efficiency: EfficiencyEvaluation

    @model_validator(mode="after")
    def _coherent_success(self) -> "TaskSuccessEvaluation":
        expected = all((
            self.normal_final_answer_produced,
            self.core_conclusion_answered,
            self.no_material_error,
            self.no_material_unsupported_conclusion,
            self.core_citations_correct,
            self.version_scope_compliant,
        ))
        if self.task_success != expected:
            raise ValueError("task_success must equal the accepted success gates")
        if self.core_citations_correct and not self.citation_integrity:
            raise ValueError(
                "core citations cannot pass deterministic citation integrity"
            )
        if len(self.failure_reasons) != len(set(self.failure_reasons)):
            raise ValueError("failure reasons must be unique")
        if self.task_success and self.failure_reasons:
            raise ValueError("successful task cannot contain failure reasons")
        if not self.task_success and not self.failure_reasons:
            raise ValueError("failed task requires at least one failure reason")
        if self.efficiency.question_id != self.question_id:
            raise ValueError("efficiency question ID must match task judgment")
        return self


def apply_task_success_review(
    evaluation: TaskSuccessEvaluation,
    decision: TaskSuccessReviewDecision,
) -> TaskSuccessEvaluation:
    """Promote one proposal while preserving all deterministic gates."""

    if evaluation.question_id != decision.question_id:
        raise ValueError("review question ID does not match evaluation")
    if evaluation.source_artifact_sha256 != decision.source_artifact_sha256:
        raise ValueError("review source artifact hash does not match evaluation")
    if evaluation.review_status != "proposed":
        raise ValueError("only proposed Task Success evaluations can be reviewed")

    core_citations_correct = (
        decision.core_citations_correct and evaluation.citation_integrity
    )
    task_success = all((
        evaluation.normal_final_answer_produced,
        decision.core_conclusion_answered,
        decision.no_material_error,
        decision.no_material_unsupported_conclusion,
        core_citations_correct,
        evaluation.version_scope_compliant,
    ))
    reasons = _failure_reasons(
        normal_final_answer_produced=evaluation.normal_final_answer_produced,
        core_conclusion_answered=decision.core_conclusion_answered,
        no_material_error=decision.no_material_error,
        no_material_unsupported_conclusion=(
            decision.no_material_unsupported_conclusion
        ),
        core_citations_correct=core_citations_correct,
        version_scope_compliant=evaluation.version_scope_compliant,
        cited_ids_present=evaluation.cited_evidence_present,
    )
    return TaskSuccessEvaluation.model_validate({
        **evaluation.model_dump(mode="python"),
        "review_status": "reviewed",
        "reviewed_by": decision.reviewed_by,
        "reviewed_at": decision.reviewed_at,
        "core_conclusion_answered": decision.core_conclusion_answered,
        "no_material_error": decision.no_material_error,
        "no_material_unsupported_conclusion": (
            decision.no_material_unsupported_conclusion
        ),
        "core_citations_correct": core_citations_correct,
        "task_success": task_success,
        "failure_reasons": reasons,
        "rationale": decision.rationale,
    })


def _failure_reasons(
    *,
    normal_final_answer_produced: bool,
    core_conclusion_answered: bool,
    no_material_error: bool,
    no_material_unsupported_conclusion: bool,
    core_citations_correct: bool,
    version_scope_compliant: bool,
    cited_ids_present: bool,
) -> tuple[FailureReason, ...]:
    if not normal_final_answer_produced:
        return ("runtime_failure",)
    reasons: list[FailureReason] = []
    if not core_conclusion_answered:
        reasons.append("missing_core_conclusion")
    if not no_material_error:
        reasons.append("material_error")
    if not no_material_unsupported_conclusion:
        reasons.append("material_unsupported_claim")
    if not core_citations_correct:
        reasons.append(
            "incorrect_core_citation"
            if cited_ids_present else "missing_core_citation"
        )
    if not version_scope_compliant:
        reasons.append("version_scope_violation")
    return tuple(reasons)


def evaluate_task_success(
    artifact_path: Path,
    qa_record: Mapping[str, Any],
    semantic: TaskSuccessSemanticDraft,
    *,
    judge_prompt_version: str | None = None,
    judge_model: str | None = None,
) -> TaskSuccessEvaluation:
    """Combine independent semantic review with deterministic runtime gates."""

    raw_bytes = artifact_path.read_bytes()
    artifact = json.loads(raw_bytes)
    question_id = str(qa_record["id"])
    artifact_question_value = artifact.get("question", {})
    artifact_question = (
        artifact_question_value
        if isinstance(artifact_question_value, Mapping) else {}
    )
    if str(artifact_question.get("id") or "") != question_id:
        raise ValueError("Run artifact question ID does not match QA record")
    expected_question = qa_record.get("question")
    if (
        expected_question is not None
        and str(artifact_question.get("text") or "") != str(expected_question)
    ):
        raise ValueError("Run artifact question text does not match QA record")
    state_value = artifact.get("state", {})
    state = state_value if isinstance(state_value, Mapping) else {}
    synthesis_value = state.get("synthesis") or artifact.get("synthesis") or {}
    synthesis = synthesis_value if isinstance(synthesis_value, Mapping) else {}
    answer = str(synthesis.get("answer") or state.get("final_answer") or "").strip()
    completed = state.get("stop_reason") == "completed" and bool(answer)

    candidates = {
        str(item["evidence_id"]): item
        for item in state.get("candidate_evidence", ())
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    validated = {
        str(item["evidence_id"]): item
        for item in state.get("validated_evidence", ())
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    cited_ids = tuple(str(value) for value in synthesis.get("evidence_ids", ()))
    expected_release = int(qa_record["release"])
    scope_value = artifact.get("retrieval_scope", {})
    scope = scope_value if isinstance(scope_value, Mapping) else {}
    scope_release = scope.get("release")
    scope_documents = tuple(
        item for item in scope.get("documents", ())
        if isinstance(item, Mapping)
    )
    allowed_documents = {
        (
            str(item.get("source_path") or ""),
            str(item.get("document_sha256") or "").upper(),
            str(item.get("version") or ""),
            int(item.get("release", -1)),
        )
        for item in scope_documents
    }
    citation_integrity = bool(cited_ids) and all(
        evidence_id in candidates
        and evidence_id in validated
        and validated[evidence_id].get("source_valid") is True
        and validated[evidence_id].get("validation_status") == "supported"
        for evidence_id in cited_ids
    )
    version_scope_compliant = (
        scope_release == expected_release
        and bool(allowed_documents)
        and bool(cited_ids)
        and all(
            (
                str(candidates[evidence_id].get("source_path") or ""),
                str(
                    candidates[evidence_id].get("document_sha256") or ""
                ).upper(),
                str(candidates[evidence_id].get("version") or ""),
                int(candidates[evidence_id].get("release", -1)),
            ) in allowed_documents
            for evidence_id in cited_ids
            if evidence_id in candidates
        )
        and all(evidence_id in candidates for evidence_id in cited_ids)
    )
    core_citations_correct = semantic.core_citations_correct and citation_integrity

    gates = (
        completed,
        semantic.core_conclusion_answered,
        semantic.no_material_error,
        semantic.no_material_unsupported_conclusion,
        core_citations_correct,
        version_scope_compliant,
    )
    task_success = all(gates)
    reasons = _failure_reasons(
        normal_final_answer_produced=completed,
        core_conclusion_answered=semantic.core_conclusion_answered,
        no_material_error=semantic.no_material_error,
        no_material_unsupported_conclusion=(
            semantic.no_material_unsupported_conclusion
        ),
        core_citations_correct=core_citations_correct,
        version_scope_compliant=version_scope_compliant,
        cited_ids_present=bool(cited_ids),
    )

    return TaskSuccessEvaluation(
        question_id=question_id,
        source_artifact=artifact_path.as_posix(),
        source_artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        review_status="proposed",
        normal_final_answer_produced=completed,
        core_conclusion_answered=semantic.core_conclusion_answered,
        no_material_error=semantic.no_material_error,
        no_material_unsupported_conclusion=(
            semantic.no_material_unsupported_conclusion
        ),
        cited_evidence_present=bool(cited_ids),
        citation_integrity=citation_integrity,
        core_citations_correct=core_citations_correct,
        version_scope_compliant=version_scope_compliant,
        task_success=task_success,
        failure_reasons=reasons,
        rationale=semantic.rationale,
        judge_rationale=semantic.rationale,
        judge_prompt_version=judge_prompt_version,
        judge_model=judge_model,
        efficiency=extract_efficiency(artifact, question_id=question_id),
    )


def _statistics(values: tuple[float, ...]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "sum": None, "mean": None, "p95": None}
    ordered = tuple(sorted(values))
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "sum": round(sum(values), 3),
        "mean": round(sum(values) / len(values), 3),
        "p95": round(ordered[index], 3),
    }


def summarize_end_to_end(
    evaluations: tuple[TaskSuccessEvaluation, ...],
) -> dict[str, Any]:
    """Summarize reviewed task success plus transparent efficiency telemetry."""

    reviewed = tuple(item for item in evaluations if item.review_status == "reviewed")
    if len(reviewed) != len(evaluations):
        proposed = sorted(
            item.question_id for item in evaluations
            if item.review_status != "reviewed"
        )
        raise ValueError(
            "end-to-end summary requires reviewed judgments: "
            + ", ".join(proposed)
        )
    token_complete = tuple(
        item for item in reviewed if item.efficiency.token_telemetry_complete
    )
    token_statistics = _statistics(tuple(
        float(item.efficiency.total_tokens)
        for item in token_complete
        if item.efficiency.total_tokens is not None
    ))
    if len(token_complete) != len(reviewed):
        token_statistics.update({"sum": None, "mean": None, "p95": None})
    return {
        "schema_version": "specrag.end-to-end-evaluation-summary/v1",
        "question_count": len(evaluations),
        "reviewed_count": len(reviewed),
        "successful_count": sum(item.task_success for item in reviewed),
        "task_success_rate": (
            round(sum(item.task_success for item in reviewed) / len(reviewed), 4)
            if reviewed else 0.0
        ),
        "telemetry_complete_count": len(token_complete),
        "efficiency": {
            "latency_ms": _statistics(tuple(
                item.efficiency.latency_ms for item in reviewed
            )),
            "total_tokens": token_statistics,
            "llm_call_count": _statistics(tuple(
                float(item.efficiency.llm_call_count) for item in reviewed
            )),
            "tool_call_count": _statistics(tuple(
                float(item.efficiency.tool_call_count) for item in reviewed
            )),
        },
    }


__all__ = [
    "EfficiencyEvaluation",
    "FailureReason",
    "TaskSuccessEvaluation",
    "TaskSuccessReviewDecision",
    "TaskSuccessSemanticDraft",
    "apply_task_success_review",
    "evaluate_task_success",
    "extract_efficiency",
    "summarize_end_to_end",
]

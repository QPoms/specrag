"""Deterministic metrics for the component evidence path."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .component_evidence_models import (
    FinalEvidenceJudgment,
    ResearcherEvidenceJudgment,
    RetrievalCallJudgment,
    SynthesizerJudgment,
)
from .component_evidence_support import (
    artifact_metadata as _artifact_metadata,
    candidate_evidence as _candidate_evidence,
    deduplicated_final_evidence as _deduplicated_final_evidence,
    locator_equals as _locator_equals,
    locator_matches as _locator_matches,
    ndcg as _ndcg,
    optional_text as _optional_text,
    ratio as _ratio,
    read_artifact as _read_artifact,
    require_assessed_units as _require_assessed_units,
    require_evidence_unit_references as _require_evidence_unit_references,
    require_fixed_gold_task as _require_fixed_gold_task,
    require_question as _require_question,
    set_difference as _set_difference,
    state as _state,
    synthesis_citations as _synthesis_citations,
    target_recall as _target_recall,
    task_evidence_units as _task_evidence_units,
    unique_evidence_ids as _unique_evidence_ids,
)
from .component_models import ComponentQuestionAnnotation, ReviewStatus


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalEvaluation(_StrictModel):
    schema_version: str = "specrag.retrieval-evaluation/v1"
    question_id: str
    call_id: str
    judgment_schema_version: str
    artifact_schema_version: str | None = None
    judgment_review_status: ReviewStatus
    judgment_reviewed_by: str | None = None
    judgment_reviewed_at: datetime | None = None
    result_count: int = Field(ge=0)
    target_count: int = Field(ge=1)
    unjudged_result_count: int = Field(ge=0)
    target_recall_at_5: float = Field(ge=0, le=1)
    target_recall_at_12: float = Field(ge=0, le=1)
    ndcg_at_5: float = Field(ge=0, le=1)
    ndcg_at_12: float = Field(ge=0, le=1)
    ranked_relevances: tuple[int, ...]
    corpus_scope: dict[str, Any]
    tool_name: str
    retrieval_request: dict[str, Any]
    judge_prompt_version: str | None = None
    judge_model: str | None = None
    artifact_path: str
    artifact_sha256: str


class ResearcherEvaluation(_StrictModel):
    schema_version: str = "specrag.researcher-evaluation/v1"
    question_id: str
    task_id: str
    annotation_schema_version: str
    judgment_schema_version: str
    artifact_schema_version: str | None = None
    annotation_review_status: ReviewStatus
    judgment_review_status: ReviewStatus
    annotation_reviewed_by: str | None = None
    annotation_reviewed_at: datetime | None = None
    judgment_reviewed_by: str | None = None
    judgment_reviewed_at: datetime | None = None
    evidence_unit_count: int = Field(ge=1)
    covered_evidence_unit_count: int = Field(ge=0)
    submitted_evidence_count: int = Field(ge=0)
    precise_evidence_count: int = Field(ge=0)
    evidence_coverage: float = Field(ge=0, le=1)
    full_evidence_coverage: bool
    evidence_precision: float = Field(ge=0, le=1)
    diagnostic_tags: tuple[str, ...]
    stop_reason: str | None = None
    runtime_prompt_version: str | None = None
    runtime_model: str | None = None
    judge_prompt_version: str | None = None
    judge_model: str | None = None
    artifact_path: str
    artifact_sha256: str


class FinalEvidenceEvaluation(_StrictModel):
    schema_version: str = "specrag.final-evidence-evaluation/v1"
    question_id: str
    annotation_schema_version: str
    judgment_schema_version: str
    artifact_schema_version: str | None = None
    annotation_review_status: ReviewStatus
    judgment_review_status: ReviewStatus
    annotation_reviewed_by: str | None = None
    annotation_reviewed_at: datetime | None = None
    judgment_reviewed_by: str | None = None
    judgment_reviewed_at: datetime | None = None
    evidence_unit_count: int = Field(ge=1)
    covered_evidence_unit_count: int = Field(ge=0)
    final_evidence_count: int = Field(ge=0)
    precise_evidence_count: int = Field(ge=0)
    redundant_evidence_count: int = Field(ge=0)
    final_evidence_coverage: float = Field(ge=0, le=1)
    full_final_evidence_coverage: bool
    final_evidence_precision: float = Field(ge=0, le=1)
    evidence_overlap_rate: float = Field(ge=0, le=1)
    deduplicated_evidence_ids: tuple[str, ...]
    diagnostic_tags: tuple[str, ...]
    runtime_prompt_version: str | None = None
    runtime_model: str | None = None
    judge_prompt_version: str | None = None
    judge_model: str | None = None
    artifact_path: str
    artifact_sha256: str


class SynthesizerEvaluation(_StrictModel):
    schema_version: str = "specrag.synthesizer-evaluation/v1"
    question_id: str
    annotation_schema_version: str
    judgment_schema_version: str
    artifact_schema_version: str | None = None
    annotation_review_status: ReviewStatus
    judgment_review_status: ReviewStatus
    annotation_reviewed_by: str | None = None
    annotation_reviewed_at: datetime | None = None
    judgment_reviewed_by: str | None = None
    judgment_reviewed_at: datetime | None = None
    answer_unit_count: int = Field(ge=1)
    fully_covered_answer_unit_count: int = Field(ge=0)
    claim_count: int = Field(ge=1)
    supported_claim_count: int = Field(ge=0)
    evidence_requiring_claim_count: int = Field(ge=0)
    correctly_cited_claim_count: int = Field(ge=0)
    answer_coverage: float = Field(ge=0, le=1)
    full_answer_coverage: bool
    answer_groundedness: float = Field(ge=0, le=1)
    citation_correctness: float = Field(ge=0, le=1)
    diagnostic_tags: tuple[str, ...]
    runtime_prompt_version: str | None = None
    runtime_model: str | None = None
    judge_prompt_version: str | None = None
    judge_model: str | None = None
    artifact_path: str
    artifact_sha256: str


def evaluate_retrieval_artifact(
    artifact_path: Path,
    judgment: RetrievalCallJudgment,
) -> RetrievalEvaluation:
    raw_bytes, artifact = _read_artifact(artifact_path)
    _require_question(artifact, judgment.question_id)
    state = _state(artifact)
    matches = tuple(
        item
        for item in state.get("tool_calls", ())
        if isinstance(item, Mapping) and item.get("call_id") == judgment.call_id
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one retrieval call {judgment.call_id}, found {len(matches)}"
        )
    call = matches[0]
    if judgment.task_id is not None and call.get("task_id") != judgment.task_id:
        raise ValueError("retrieval call task does not match fixed GoldResearchTask")
    tool_name = str(call.get("tool_name", ""))
    if tool_name not in {"hybrid_search", "semantic_search", "keyword_search"}:
        raise ValueError(f"{tool_name!r} is not a ranked-search tool")
    results = tuple(
        item for item in call.get("observed_evidence", ()) if isinstance(item, Mapping)
    )
    relevances = tuple(
        max(
            (
                qrel.relevance
                for qrel in judgment.qrels
                if not qrel.target
                and _locator_equals(qrel.locator.model_dump(), result)
            ),
            default=0,
        )
        for result in results
    )
    targets = tuple(item for item in judgment.qrels if item.target)
    unjudged = sum(
        not any(
            not qrel.target
            and _locator_equals(qrel.locator.model_dump(), result)
            for qrel in judgment.qrels
        )
        for result in results
    )
    artifact_scope = artifact.get("retrieval_scope", {})
    if artifact_scope and artifact_scope != judgment.corpus_scope:
        raise ValueError(
            "retrieval artifact corpus scope does not match fixed judgment scope"
        )
    return RetrievalEvaluation(
        question_id=judgment.question_id,
        call_id=judgment.call_id,
        judgment_schema_version=judgment.schema_version,
        artifact_schema_version=_optional_text(artifact.get("schema_version")),
        judgment_review_status=judgment.review_status,
        judgment_reviewed_by=judgment.reviewed_by,
        judgment_reviewed_at=judgment.reviewed_at,
        result_count=len(results),
        target_count=len(targets),
        unjudged_result_count=unjudged,
        target_recall_at_5=_target_recall(results, targets, 5),
        target_recall_at_12=_target_recall(results, targets, 12),
        ndcg_at_5=_ndcg(
            relevances,
            tuple(item.relevance for item in judgment.qrels if not item.target),
            5,
        ),
        ndcg_at_12=_ndcg(
            relevances,
            tuple(item.relevance for item in judgment.qrels if not item.target),
            12,
        ),
        ranked_relevances=relevances,
        corpus_scope=judgment.corpus_scope,
        tool_name=tool_name,
        retrieval_request={
            key: call.get(key)
            for key in (
                "task_id",
                "lexical_query",
                "semantic_query",
                "target_specifications",
            )
            if call.get(key) not in (None, "", [])
        },
        judge_prompt_version=judgment.judge_prompt_version,
        judge_model=judgment.judge_model,
        artifact_path=artifact_path.as_posix(),
        artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def evaluate_researcher_artifact(
    artifact_path: Path,
    annotation: ComponentQuestionAnnotation,
    judgment: ResearcherEvidenceJudgment,
) -> ResearcherEvaluation:
    raw_bytes, artifact = _read_artifact(artifact_path)
    _require_question(artifact, judgment.question_id)
    if annotation.question_id != judgment.question_id:
        raise ValueError("Researcher annotation and judgment question IDs differ")
    _require_fixed_gold_task(artifact, annotation, judgment.task_id)
    unit_ids = _task_evidence_units(annotation, judgment.task_id)
    _require_assessed_units(unit_ids, judgment.unit_assessments)
    submitted = tuple(
        item
        for item in _candidate_evidence(artifact)
        if str(item.get("task_id", "")) == judgment.task_id
    )
    submitted_ids = _unique_evidence_ids(submitted)
    assessed_ids = {item.evidence_id for item in judgment.item_assessments}
    if assessed_ids != set(submitted_ids):
        raise ValueError(_set_difference("Researcher evidence", set(submitted_ids), assessed_ids))
    _require_evidence_unit_references(judgment.item_assessments, unit_ids)
    covered = sum(item.status == "covered" for item in judgment.unit_assessments)
    precise = sum(
        item.contribution in {"direct_support", "necessary_chain"}
        for item in judgment.item_assessments
    )
    return ResearcherEvaluation(
        question_id=judgment.question_id,
        task_id=judgment.task_id,
        annotation_schema_version=annotation.schema_version,
        judgment_schema_version=judgment.schema_version,
        artifact_schema_version=_optional_text(artifact.get("schema_version")),
        annotation_review_status=annotation.review_status,
        judgment_review_status=judgment.review_status,
        annotation_reviewed_by=annotation.reviewed_by,
        annotation_reviewed_at=annotation.reviewed_at,
        judgment_reviewed_by=judgment.reviewed_by,
        judgment_reviewed_at=judgment.reviewed_at,
        evidence_unit_count=len(unit_ids),
        covered_evidence_unit_count=covered,
        submitted_evidence_count=len(submitted_ids),
        precise_evidence_count=precise,
        evidence_coverage=covered / len(unit_ids),
        full_evidence_coverage=covered == len(unit_ids),
        evidence_precision=_ratio(precise, len(submitted_ids)),
        diagnostic_tags=judgment.diagnostic_tags,
        stop_reason=_optional_text(_state(artifact).get("last_researcher_stop_reason")),
        runtime_prompt_version=_artifact_metadata(artifact, "prompt_version"),
        runtime_model=_artifact_metadata(artifact, "model"),
        judge_prompt_version=judgment.judge_prompt_version,
        judge_model=judgment.judge_model,
        artifact_path=artifact_path.as_posix(),
        artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def evaluate_final_evidence_artifact(
    artifact_path: Path,
    annotation: ComponentQuestionAnnotation,
    judgment: FinalEvidenceJudgment,
) -> FinalEvidenceEvaluation:
    raw_bytes, artifact = _read_artifact(artifact_path)
    _require_question(artifact, judgment.question_id)
    if annotation.question_id != judgment.question_id:
        raise ValueError("Final Evidence annotation and judgment question IDs differ")
    if artifact.get("schema_version") == "specrag.component-final-evidence-run/v1":
        contract = artifact.get("candidate_pool_contract", {})
        if not isinstance(contract, Mapping) or contract.get("gold_sufficient") is not True:
            raise ValueError("Final Evidence run requires a gold-sufficient candidate pool")
    unit_ids = {item.unit_id for item in annotation.evidence_units}
    if not unit_ids:
        raise ValueError("Final Evidence annotation has no EvidenceUnits")
    _require_assessed_units(unit_ids, judgment.unit_assessments)
    selected = _deduplicated_final_evidence(artifact)
    selected_ids = _unique_evidence_ids(selected)
    assessed_ids = {item.evidence_id for item in judgment.item_assessments}
    if assessed_ids != set(selected_ids):
        raise ValueError(_set_difference("Final Evidence", set(selected_ids), assessed_ids))
    _require_evidence_unit_references(judgment.item_assessments, unit_ids)
    covered = sum(item.status == "covered" for item in judgment.unit_assessments)
    precise = sum(
        item.contribution in {"direct_support", "necessary_chain"}
        for item in judgment.item_assessments
    )
    redundant = len(judgment.redundant_evidence_ids)
    return FinalEvidenceEvaluation(
        question_id=judgment.question_id,
        annotation_schema_version=annotation.schema_version,
        judgment_schema_version=judgment.schema_version,
        artifact_schema_version=_optional_text(artifact.get("schema_version")),
        annotation_review_status=annotation.review_status,
        judgment_review_status=judgment.review_status,
        annotation_reviewed_by=annotation.reviewed_by,
        annotation_reviewed_at=annotation.reviewed_at,
        judgment_reviewed_by=judgment.reviewed_by,
        judgment_reviewed_at=judgment.reviewed_at,
        evidence_unit_count=len(unit_ids),
        covered_evidence_unit_count=covered,
        final_evidence_count=len(selected_ids),
        precise_evidence_count=precise,
        redundant_evidence_count=redundant,
        final_evidence_coverage=covered / len(unit_ids),
        full_final_evidence_coverage=covered == len(unit_ids),
        final_evidence_precision=_ratio(precise, len(selected_ids)),
        evidence_overlap_rate=_ratio(redundant, len(selected_ids)),
        deduplicated_evidence_ids=selected_ids,
        diagnostic_tags=judgment.diagnostic_tags,
        runtime_prompt_version=_artifact_metadata(artifact, "sufficiency_prompt_version"),
        runtime_model=_artifact_metadata(artifact, "model"),
        judge_prompt_version=judgment.judge_prompt_version,
        judge_model=judgment.judge_model,
        artifact_path=artifact_path.as_posix(),
        artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def evaluate_synthesizer_artifact(
    artifact_path: Path,
    annotation: ComponentQuestionAnnotation,
    judgment: SynthesizerJudgment,
) -> SynthesizerEvaluation:
    raw_bytes, artifact = _read_artifact(artifact_path)
    _require_question(artifact, judgment.question_id)
    if artifact.get("schema_version") == "specrag.component-synthesis-run/v1":
        contract = artifact.get("input_contract")
        if not isinstance(contract, Mapping) or not (
            contract.get("judge_independent") is True
            and contract.get("gold_sufficient") is True
        ):
            raise ValueError(
                "formal Synthesizer evaluation requires a judge-independent, "
                "gold-sufficient input contract"
            )
    if annotation.question_id != judgment.question_id:
        raise ValueError("Synthesizer annotation and judgment question IDs differ")
    answer_unit_ids = {item.unit_id for item in annotation.answer_units}
    if not answer_unit_ids:
        raise ValueError("Synthesizer annotation has no AnswerUnits")
    assessed_unit_ids = {item.unit_id for item in judgment.answer_unit_assessments}
    if assessed_unit_ids != answer_unit_ids:
        raise ValueError(_set_difference("AnswerUnits", answer_unit_ids, assessed_unit_ids))
    synthesis = artifact.get("synthesis")
    if not isinstance(synthesis, Mapping):
        synthesis = _state(artifact).get("synthesis")
    if not isinstance(synthesis, Mapping) or not str(synthesis.get("answer", "")).strip():
        raise ValueError("Synthesizer artifact contains no answer")
    actual_citations = _synthesis_citations(synthesis)
    judged_citations = {
        evidence_id
        for claim in judgment.claim_assessments
        for evidence_id in claim.cited_evidence_ids
    }
    if judged_citations != actual_citations:
        raise ValueError(
            _set_difference("Synthesizer citations", actual_citations, judged_citations)
        )
    weights = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
    coverage = sum(
        weights[item.status] for item in judgment.answer_unit_assessments
    ) / len(answer_unit_ids)
    fully_covered = sum(
        item.status == "covered" for item in judgment.answer_unit_assessments
    )
    claims = judgment.claim_assessments
    supported = sum(item.support_status == "supported" for item in claims)
    requiring = tuple(item for item in claims if item.requires_evidence)
    correct = sum(item.citation_status == "correct" for item in requiring)
    return SynthesizerEvaluation(
        question_id=judgment.question_id,
        annotation_schema_version=annotation.schema_version,
        judgment_schema_version=judgment.schema_version,
        artifact_schema_version=_optional_text(artifact.get("schema_version")),
        annotation_review_status=annotation.review_status,
        judgment_review_status=judgment.review_status,
        annotation_reviewed_by=annotation.reviewed_by,
        annotation_reviewed_at=annotation.reviewed_at,
        judgment_reviewed_by=judgment.reviewed_by,
        judgment_reviewed_at=judgment.reviewed_at,
        answer_unit_count=len(answer_unit_ids),
        fully_covered_answer_unit_count=fully_covered,
        claim_count=len(claims),
        supported_claim_count=supported,
        evidence_requiring_claim_count=len(requiring),
        correctly_cited_claim_count=correct,
        answer_coverage=coverage,
        full_answer_coverage=fully_covered == len(answer_unit_ids),
        answer_groundedness=supported / len(claims),
        citation_correctness=_ratio(correct, len(requiring)),
        diagnostic_tags=judgment.diagnostic_tags,
        runtime_prompt_version=_artifact_metadata(artifact, "synthesis_prompt_version"),
        runtime_model=_artifact_metadata(artifact, "model"),
        judge_prompt_version=judgment.judge_prompt_version,
        judge_model=judgment.judge_model,
        artifact_path=artifact_path.as_posix(),
        artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


__all__ = [
    "FinalEvidenceEvaluation",
    "ResearcherEvaluation",
    "RetrievalEvaluation",
    "SynthesizerEvaluation",
    "evaluate_final_evidence_artifact",
    "evaluate_researcher_artifact",
    "evaluate_retrieval_artifact",
    "evaluate_synthesizer_artifact",
]

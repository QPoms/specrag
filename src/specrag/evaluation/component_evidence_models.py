"""Structured judgments used by evidence-path component metrics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .component_models import ReviewableModel


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceLocator(_StrictModel):
    source_path: str = Field(min_length=1)
    document_sha256: str = Field(min_length=1)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered_lines(self) -> "EvidenceLocator":
        if self.line_end < self.line_start:
            raise ValueError("evidence locator line_end precedes line_start")
        return self


class RetrievalQrel(_StrictModel):
    qrel_id: str = Field(min_length=1)
    locator: EvidenceLocator
    relevance: int = Field(ge=0, le=2)
    target: bool = False
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _target_is_direct(self) -> "RetrievalQrel":
        if self.target and self.relevance != 2:
            raise ValueError("retrieval targets must have relevance 2")
        return self


class RetrievalCallJudgment(ReviewableModel):
    schema_version: Literal["specrag.retrieval-call-judgment/v1"] = (
        "specrag.retrieval-call-judgment/v1"
    )
    question_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    task_id: str | None = None
    corpus_scope: dict[str, object] = Field(min_length=1)
    qrels: tuple[RetrievalQrel, ...] = Field(min_length=1)
    judge_prompt_version: str | None = None
    judge_model: str | None = None

    @model_validator(mode="after")
    def _valid_qrels(self) -> "RetrievalCallJudgment":
        _unique((item.qrel_id for item in self.qrels), label="qrel IDs")
        if not any(item.target for item in self.qrels):
            raise ValueError("retrieval judgment requires at least one target qrel")
        return self


class RetrievalCandidateGrade(_StrictModel):
    candidate_id: str = Field(min_length=1)
    relevance: int = Field(ge=0, le=2)
    rationale: str = Field(min_length=1)


class RetrievalJudgmentDraft(_StrictModel):
    """Semantic grades over stable target and result IDs."""

    target_locator_ids: tuple[str, ...] = Field(min_length=1)
    candidate_grades: tuple[RetrievalCandidateGrade, ...] = Field(min_length=1)
    diagnostic_tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _unique_entries(self) -> "RetrievalJudgmentDraft":
        _unique(self.target_locator_ids, label="target locator IDs")
        _unique(
            (item.candidate_id for item in self.candidate_grades),
            label="retrieval candidate grades",
        )
        _unique(self.diagnostic_tags, label="diagnostic tags")
        return self


EvidenceUnitStatus = Literal["covered", "partial", "missing"]
EvidenceContribution = Literal[
    "direct_support",
    "necessary_chain",
    "background",
    "analogous",
    "wrong_branch",
    "duplicate",
]


class EvidenceUnitAssessment(_StrictModel):
    unit_id: str = Field(min_length=1)
    status: EvidenceUnitStatus
    evidence_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_evidence(self) -> "EvidenceUnitAssessment":
        _unique(self.evidence_ids, label=f"{self.unit_id} evidence IDs")
        if self.status == "missing" and self.evidence_ids:
            raise ValueError("missing EvidenceUnit cannot cite evidence")
        return self


class EvidenceItemAssessment(_StrictModel):
    evidence_id: str = Field(min_length=1)
    contribution: EvidenceContribution
    evidence_unit_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_contribution(self) -> "EvidenceItemAssessment":
        _unique(
            self.evidence_unit_ids,
            label=f"{self.evidence_id} EvidenceUnit IDs",
        )
        if self.contribution in {"direct_support", "necessary_chain"}:
            if not self.evidence_unit_ids:
                raise ValueError("contributing evidence must map to an EvidenceUnit")
        elif self.evidence_unit_ids:
            raise ValueError("non-contributing evidence cannot map to EvidenceUnits")
        return self


class ResearcherEvidenceJudgment(ReviewableModel):
    schema_version: Literal["specrag.researcher-evidence-judgment/v1"] = (
        "specrag.researcher-evidence-judgment/v1"
    )
    question_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    unit_assessments: tuple[EvidenceUnitAssessment, ...] = Field(min_length=1)
    item_assessments: tuple[EvidenceItemAssessment, ...] = ()
    diagnostic_tags: tuple[str, ...] = ()
    judge_prompt_version: str | None = None
    judge_model: str | None = None

    @model_validator(mode="after")
    def _unique_entries(self) -> "ResearcherEvidenceJudgment":
        _unique(
            (item.unit_id for item in self.unit_assessments),
            label="EvidenceUnit assessments",
        )
        _unique(
            (item.evidence_id for item in self.item_assessments),
            label="evidence-item assessments",
        )
        _unique(self.diagnostic_tags, label="diagnostic tags")
        return self


class TaskEvidenceJudgmentDraft(_StrictModel):
    """Provider-neutral evidence assessment for one fixed ResearchTask."""

    task_id: str = Field(min_length=1)
    unit_assessments: tuple[EvidenceUnitAssessment, ...] = Field(min_length=1)
    item_assessments: tuple[EvidenceItemAssessment, ...] = ()
    diagnostic_tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _valid_entries(self) -> "TaskEvidenceJudgmentDraft":
        ResearcherEvidenceJudgment(
            question_id="draft-validation",
            task_id=self.task_id,
            review_status="proposed",
            unit_assessments=self.unit_assessments,
            item_assessments=self.item_assessments,
            diagnostic_tags=self.diagnostic_tags,
        )
        return self


class ResearcherQuestionJudgmentDraft(_StrictModel):
    """All task-level evidence assessments for one question artifact."""

    task_judgments: tuple[TaskEvidenceJudgmentDraft, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_tasks(self) -> "ResearcherQuestionJudgmentDraft":
        _unique(
            (item.task_id for item in self.task_judgments),
            label="task judgments",
        )
        return self


class FinalEvidenceJudgment(ReviewableModel):
    schema_version: Literal["specrag.final-evidence-judgment/v1"] = (
        "specrag.final-evidence-judgment/v1"
    )
    question_id: str = Field(min_length=1)
    unit_assessments: tuple[EvidenceUnitAssessment, ...] = Field(min_length=1)
    item_assessments: tuple[EvidenceItemAssessment, ...] = ()
    redundant_evidence_ids: tuple[str, ...] = ()
    diagnostic_tags: tuple[str, ...] = ()
    judge_prompt_version: str | None = None
    judge_model: str | None = None

    @model_validator(mode="after")
    def _unique_entries(self) -> "FinalEvidenceJudgment":
        _unique(
            (item.unit_id for item in self.unit_assessments),
            label="EvidenceUnit assessments",
        )
        evidence_ids = tuple(item.evidence_id for item in self.item_assessments)
        _unique(evidence_ids, label="evidence-item assessments")
        _unique(self.redundant_evidence_ids, label="redundant evidence IDs")
        unknown = set(self.redundant_evidence_ids) - set(evidence_ids)
        if unknown:
            raise ValueError(f"redundant evidence IDs are not assessed: {sorted(unknown)}")
        _unique(self.diagnostic_tags, label="diagnostic tags")
        return self


class FinalEvidenceJudgmentDraft(_StrictModel):
    """Provider-neutral whole-question Final Evidence assessment."""

    unit_assessments: tuple[EvidenceUnitAssessment, ...] = Field(min_length=1)
    item_assessments: tuple[EvidenceItemAssessment, ...] = ()
    redundant_evidence_ids: tuple[str, ...] = ()
    diagnostic_tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _valid_entries(self) -> "FinalEvidenceJudgmentDraft":
        FinalEvidenceJudgment(
            question_id="draft-validation",
            review_status="proposed",
            unit_assessments=self.unit_assessments,
            item_assessments=self.item_assessments,
            redundant_evidence_ids=self.redundant_evidence_ids,
            diagnostic_tags=self.diagnostic_tags,
        )
        return self


class AnswerUnitAssessment(_StrictModel):
    unit_id: str = Field(min_length=1)
    status: EvidenceUnitStatus
    rationale: str = Field(min_length=1)


ClaimSupportStatus = Literal["supported", "partial", "unsupported", "contradicted"]
CitationStatus = Literal[
    "correct",
    "missing",
    "unknown",
    "irrelevant",
    "incomplete",
    "not_required",
]


class AtomicClaimAssessment(_StrictModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    support_status: ClaimSupportStatus
    requires_evidence: bool = True
    citation_status: CitationStatus
    cited_evidence_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_citation_status(self) -> "AtomicClaimAssessment":
        _unique(self.cited_evidence_ids, label=f"{self.claim_id} citation IDs")
        if self.requires_evidence and self.citation_status == "not_required":
            raise ValueError("evidence-requiring claim cannot use not_required")
        if not self.requires_evidence and self.citation_status != "not_required":
            raise ValueError("non-evidentiary claim must use not_required")
        if self.citation_status == "missing" and self.cited_evidence_ids:
            raise ValueError("missing citation status cannot contain citation IDs")
        return self


class SynthesizerJudgment(ReviewableModel):
    schema_version: Literal["specrag.synthesizer-judgment/v1"] = (
        "specrag.synthesizer-judgment/v1"
    )
    question_id: str = Field(min_length=1)
    answer_unit_assessments: tuple[AnswerUnitAssessment, ...] = Field(min_length=1)
    claim_assessments: tuple[AtomicClaimAssessment, ...] = Field(min_length=1)
    diagnostic_tags: tuple[str, ...] = ()
    judge_prompt_version: str | None = None
    judge_model: str | None = None

    @model_validator(mode="after")
    def _unique_entries(self) -> "SynthesizerJudgment":
        _unique(
            (item.unit_id for item in self.answer_unit_assessments),
            label="AnswerUnit assessments",
        )
        _unique(
            (item.claim_id for item in self.claim_assessments),
            label="claim assessments",
        )
        _unique(self.diagnostic_tags, label="diagnostic tags")
        return self


class SynthesizerJudgmentDraft(_StrictModel):
    """Provider-neutral answer, claim-support, and citation assessment."""

    answer_unit_assessments: tuple[AnswerUnitAssessment, ...] = Field(min_length=1)
    claim_assessments: tuple[AtomicClaimAssessment, ...] = Field(min_length=1)
    diagnostic_tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _valid_entries(self) -> "SynthesizerJudgmentDraft":
        SynthesizerJudgment(
            question_id="draft-validation",
            review_status="proposed",
            answer_unit_assessments=self.answer_unit_assessments,
            claim_assessments=self.claim_assessments,
            diagnostic_tags=self.diagnostic_tags,
        )
        return self


def _unique(values, *, label: str) -> None:
    entries = tuple(values)
    if len(entries) != len(set(entries)):
        raise ValueError(f"{label} must be unique")


__all__ = [
    "AnswerUnitAssessment",
    "AtomicClaimAssessment",
    "EvidenceItemAssessment",
    "EvidenceLocator",
    "EvidenceUnitAssessment",
    "FinalEvidenceJudgmentDraft",
    "FinalEvidenceJudgment",
    "ResearcherQuestionJudgmentDraft",
    "ResearcherEvidenceJudgment",
    "RetrievalCallJudgment",
    "RetrievalCandidateGrade",
    "RetrievalJudgmentDraft",
    "RetrievalQrel",
    "SynthesizerJudgment",
    "SynthesizerJudgmentDraft",
    "TaskEvidenceJudgmentDraft",
]

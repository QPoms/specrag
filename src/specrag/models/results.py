"""Immutable research input, outcome, and published Result contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .corpus import CorpusVersionRef
from .research import (
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchGap,
    ResearchTask,
    StopReason,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


EvidenceStatus = Literal["sufficient", "partial", "insufficient", "conflict"]
ResultPublicationKind = Literal["final", "validated_partial"]


class SpecificationScope(_StrictModel):
    """User-requested corpus scope before exact versions are bound."""

    policy: Literal[
        "latest_per_spec",
        "highest_version_within_release",
        "exact_version",
        "mixed_versions",
    ]
    release: int | None = Field(default=None, ge=1)
    versions: tuple[CorpusVersionRef, ...] = ()

    @model_validator(mode="after")
    def _coherent_selection(self) -> "SpecificationScope":
        if self.policy == "latest_per_spec":
            if self.release is not None or self.versions:
                raise ValueError("latest scope cannot select a Release or versions")
        elif self.policy == "highest_version_within_release":
            if self.release is None or self.versions:
                raise ValueError("Release scope requires one Release and no versions")
        elif self.policy == "exact_version":
            if self.release is not None or len(self.versions) != 1:
                raise ValueError("exact scope requires exactly one version")
        elif self.release is not None or len(self.versions) < 2:
            raise ValueError("mixed scope requires at least two exact versions")
        if self.versions:
            ordered = tuple(sorted(
                self.versions,
                key=lambda item: (item.specification, item.version),
            ))
            if len({item.specification for item in ordered}) != len(ordered):
                raise ValueError("scope selects a specification more than once")
            object.__setattr__(self, "versions", ordered)
        return self


class PriorResultRef(_StrictModel):
    """Authorized immutable Result dependency captured at Run submission."""

    result_id: str = Field(pattern=r"^RESULT-[0-9A-F]{32}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    relationship: Literal[
        "follow_up", "condition_change", "revalidation", "continuation"
    ]


class ResearchInputSnapshot(_StrictModel):
    """Immutable semantic input captured before a Research Run is queued."""

    schema_version: Literal["research-input/v1"] = "research-input/v1"
    question: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    conditions: tuple[str, ...] = ()
    requested_scope: SpecificationScope
    prior_results: tuple[PriorResultRef, ...] = ()
    answer_language: str = Field(min_length=1)
    detail_level: str = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_inputs(self) -> "ResearchInputSnapshot":
        if len(self.conditions) != len(set(self.conditions)):
            raise ValueError("research conditions must be unique")
        if len(self.prior_results) != len(
            {item.result_id for item in self.prior_results}
        ):
            raise ValueError("prior Result IDs must be unique")
        return self


class ResearchOutcome(_StrictModel):
    """Internal attempt output staged atomically with terminal Run state."""

    schema_version: Literal["research-outcome/v1"] = "research-outcome/v1"
    run_id: str = Field(pattern=r"^RUN-[0-9A-F]{32}$")
    run_attempt: int = Field(default=0, ge=0)
    question: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    conditions: tuple[str, ...] = ()
    requested_scope: SpecificationScope
    corpus_versions: tuple[CorpusVersionRef, ...]
    synthesis: EvidenceSynthesis | None = None
    research_tasks: tuple[ResearchTask, ...] = ()
    candidate_evidence: tuple[CandidateEvidence, ...] = ()
    validated_evidence: tuple[ValidatedEvidence, ...] = ()
    assessments: tuple[TaskSufficiencyAssessment, ...] = ()
    gaps: tuple[ResearchGap, ...] = ()
    stop_reason: StopReason
    prior_results: tuple[PriorResultRef, ...] = ()
    created_at: datetime

    @model_validator(mode="after")
    def _coherent_evidence(self) -> "ResearchOutcome":
        candidate_ids = [item.evidence_id for item in self.candidate_evidence]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("outcome candidate evidence IDs must be unique")
        validation_ids = [item.evidence_id for item in self.validated_evidence]
        if len(validation_ids) != len(set(validation_ids)):
            raise ValueError("outcome validation evidence IDs must be unique")
        if set(validation_ids) - set(candidate_ids):
            raise ValueError("outcome validation references unknown evidence")
        if self.synthesis is not None:
            unknown = set(self.synthesis.evidence_ids) - set(validation_ids)
            if unknown:
                raise ValueError("outcome synthesis references unvalidated evidence")
        task_ids = [item.task_id for item in self.research_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("outcome research task IDs must be unique")
        if len(self.conditions) != len(set(self.conditions)):
            raise ValueError("outcome conditions must be unique")
        if len(self.prior_results) != len(
            {item.result_id for item in self.prior_results}
        ):
            raise ValueError("outcome prior Result IDs must be unique")
        return self


class ResearchResultEvidence(_StrictModel):
    """Source-valid evidence copied into a durable Research Result."""

    source: CandidateEvidence
    validation: ValidatedEvidence

    @model_validator(mode="after")
    def _coherent_validation(self) -> "ResearchResultEvidence":
        if self.source.evidence_id != self.validation.evidence_id:
            raise ValueError("Result evidence and validation IDs must match")
        if self.source.task_id != self.validation.task_id:
            raise ValueError("Result evidence and validation tasks must match")
        if not self.validation.source_valid:
            raise ValueError("Result evidence must be source-valid")
        return self


class ResearchResult(_StrictModel):
    """Durable, independently readable evidence-backed research artifact."""

    schema_version: Literal["research-result/v1"] = "research-result/v1"
    result_id: str = Field(pattern=r"^RESULT-[0-9A-F]{32}$")
    run_id: str = Field(pattern=r"^RUN-[0-9A-F]{32}$")
    run_attempt: int = Field(default=0, ge=0)
    revision: int = Field(default=1, ge=1)
    workspace_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    conversation_id: str | None = Field(
        default=None, pattern=r"^CONV-[0-9A-F]{32}$"
    )
    publication_kind: ResultPublicationKind = "final"
    evidence_status: EvidenceStatus
    answer: str = Field(min_length=1)
    evidence: tuple[ResearchResultEvidence, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    research_tasks: tuple[ResearchTask, ...] = ()
    task_assessments: tuple[TaskSufficiencyAssessment, ...] = ()
    conditions: tuple[str, ...] = ()
    requested_scope: SpecificationScope
    corpus_versions: tuple[CorpusVersionRef, ...]
    unresolved_questions: tuple[str, ...] = ()
    gaps: tuple[ResearchGap, ...] = ()
    prior_results: tuple[PriorResultRef, ...] = ()
    source_outcome_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supersedes_result_id: str | None = Field(
        default=None, pattern=r"^RESULT-[0-9A-F]{32}$"
    )
    created_at: datetime

    @model_validator(mode="after")
    def _coherent_publication(self) -> "ResearchResult":
        evidence_by_id = {
            item.source.evidence_id: item for item in self.evidence
        }
        if len(evidence_by_id) != len(self.evidence):
            raise ValueError("Result evidence IDs must be unique")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Result cited evidence IDs must be unique")
        if set(self.evidence_ids) - set(evidence_by_id):
            raise ValueError("Result cites evidence not copied into the Result")
        for evidence_id in self.evidence_ids:
            validation = evidence_by_id[evidence_id].validation
            if validation.validation_status == "rejected":
                raise ValueError("Result cannot cite rejected evidence")
        if self.evidence_status == "sufficient" and not self.evidence_ids:
            raise ValueError("sufficient Result requires cited evidence")
        task_ids = [item.task_id for item in self.research_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Result research task IDs must be unique")
        assessment_ids = [item.task_id for item in self.task_assessments]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("Result task assessment IDs must be unique")
        if task_ids and set(assessment_ids) - set(task_ids):
            raise ValueError("Result assessment references an unknown research task")
        assessment_evidence_ids = {
            evidence_id
            for item in self.task_assessments
            for evidence_id in item.supporting_evidence_ids
        }
        if assessment_evidence_ids - set(evidence_by_id):
            raise ValueError("Result assessment references evidence not copied into Result")
        if self.supersedes_result_id == self.result_id:
            raise ValueError("Result cannot supersede itself")
        for name, values in (
            ("conditions", self.conditions),
            ("unresolved questions", self.unresolved_questions),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Result {name} must be unique")
        if len(self.prior_results) != len(
            {item.result_id for item in self.prior_results}
        ):
            raise ValueError("Result prior Result IDs must be unique")
        return self


__all__ = [
    "EvidenceStatus",
    "PriorResultRef",
    "ResearchInputSnapshot",
    "ResearchOutcome",
    "ResearchResult",
    "ResearchResultEvidence",
    "ResultPublicationKind",
    "SpecificationScope",
]

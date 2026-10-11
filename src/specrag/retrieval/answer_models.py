"""Evidence selection, repair, and answer-summary contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..infrastructure.structured_agent import CodexAgentReceipt
from .execution_models import (
    EvidenceGroundedFollowup,
    GroundedRelationEvidence,
    PlannedRetrievalExecution,
    PlannedRetrievalTaskResult,
)
from .plan_models import DiscoveryRetrievalPlanV3, PLANNER_PROFILE
from .models import release_from_version


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


AnswerType = Literal[
    "protocol_message",
    "information_element",
    "container",
    "procedure",
    "boundary",
    "other",
]
AnswerSupportKind = Literal[
    "direct_statement",
    "table_relation",
    "combined_evidence",
    "negative_boundary",
]
AnswerCoverageStatus = Literal[
    "supported",
    "partial",
    "unsupported",
    "unresolved",
]


class EvidenceBackedAnswer(_StrictModel):
    answer_id: str = Field(min_length=1)
    hypothesis_id: str | None = Field(...)
    coverage_status: AnswerCoverageStatus = Field(...)
    answer_type: AnswerType
    statement: str = Field(min_length=1)
    support_kind: AnswerSupportKind
    evidence_ids: tuple[str, ...] = Field(max_length=8)


class EvidenceRepairSubQuery(_StrictModel):
    """One local evidence lookup within a bounded repair task."""

    specification: str = Field(min_length=1)
    lexical_query: str = Field(min_length=1)
    semantic_query: str = Field(min_length=1)
    required_terms: tuple[str, ...] = Field(min_length=1, max_length=3)

    @field_validator("specification", "lexical_query", "semantic_query")
    @classmethod
    def _non_blank_subquery_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("repair subquery text must be non-blank")
        return cleaned

    @field_validator("required_terms")
    @classmethod
    def _unique_subquery_terms(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("repair subquery terms must be non-blank")
        if len({value.casefold() for value in cleaned}) != len(cleaned):
            raise ValueError("repair subquery terms must be unique")
        return cleaned


class EvidenceRepairQuery(_StrictModel):
    repair_id: str = Field(min_length=1)
    hypothesis_ids: tuple[str, ...] = Field(max_length=6)
    requirement_id: str | None
    missing_claim: str = Field(min_length=1)
    evidence_queries: tuple[EvidenceRepairSubQuery, ...] = Field(
        min_length=1,
        max_length=3,
    )
    rationale: str = Field(min_length=1)

    @field_validator("requirement_id")
    @classmethod
    def _non_blank_optional_requirement_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("repair requirement ID must be non-blank")
        return cleaned

    @field_validator("repair_id", "missing_claim", "rationale")
    @classmethod
    def _non_blank_repair_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("repair query text must be non-blank")
        return cleaned

    @field_validator("hypothesis_ids")
    @classmethod
    def _non_blank_repair_hypothesis_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("repair hypothesis IDs must be non-blank")
        return cleaned

    @model_validator(mode="after")
    def _unique_evidence_queries(self) -> "EvidenceRepairQuery":
        keys = {
            (
                query.specification.casefold(),
                query.lexical_query.casefold(),
                query.semantic_query.casefold(),
            )
            for query in self.evidence_queries
        }
        if len(keys) != len(self.evidence_queries):
            raise ValueError("repair evidence queries must be unique")
        if len(set(self.hypothesis_ids)) != len(self.hypothesis_ids):
            raise ValueError("repair hypothesis IDs must be unique")
        return self


class AgentQuestionAnswerProposal(_StrictModel):
    query_id: str = Field(min_length=1)
    answers: tuple[EvidenceBackedAnswer, ...] = Field(max_length=10)


class AgentRepairableQuestionAnswerProposal(_StrictModel):
    query_id: str = Field(min_length=1)
    evidence_status: Literal["sufficient", "insufficient"]
    answers: tuple[EvidenceBackedAnswer, ...] = Field(max_length=10)
    repair_queries: tuple[EvidenceRepairQuery, ...] = Field(max_length=3)

    @model_validator(mode="after")
    def _repair_matches_status(self) -> "AgentRepairableQuestionAnswerProposal":
        if self.evidence_status == "sufficient" and self.repair_queries:
            raise ValueError("sufficient evidence cannot request repair")
        return self


class QuestionAnswerSummary(_StrictModel):
    query_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    angle: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    status: Literal["completed", "no_answers", "unavailable"]
    evidence_status: Literal[
        "sufficient",
        "insufficient",
        "not_assessed",
        "unavailable",
    ] = "not_assessed"
    answers: tuple[EvidenceBackedAnswer, ...] = ()
    repair_queries: tuple[EvidenceRepairQuery, ...] = ()
    warnings: tuple[str, ...] = ()


class EvidenceAnswerSummary(_StrictModel):
    schema_version: Literal["evidence-answer-summary/v1.5"] = (
        "evidence-answer-summary/v1.5"
    )
    status: Literal["completed", "partial", "unavailable"]
    max_answers_per_question: int = Field(ge=1, le=10)
    agent_receipts: tuple[CodexAgentReceipt, ...] = ()
    questions: tuple[QuestionAnswerSummary, ...] = ()
    errors: tuple[str, ...] = ()


class AgentEvidenceGroupProposal(_StrictModel):
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    hypothesis_ids: tuple[str, ...]


HypothesisSupportStatus = Literal[
    "supported",
    "partial",
    "unsupported",
    "unassessed",
]


class AgentHypothesisEvidenceAssessmentProposal(_StrictModel):
    hypothesis_id: str = Field(min_length=1)
    status: Literal["supported", "partial", "unsupported"]
    evidence_ids: tuple[str, ...] = Field(max_length=4)


class AgentIndependentEvidenceGroupProposal(_StrictModel):
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=4)


class AgentQuestionEvidenceSelectionProposal(_StrictModel):
    query_id: str = Field(min_length=1)
    hypothesis_assessments: tuple[AgentHypothesisEvidenceAssessmentProposal, ...]
    independent_evidence_groups: tuple[
        AgentIndependentEvidenceGroupProposal, ...
    ] = Field(max_length=10)
    evidence_groups: tuple[AgentEvidenceGroupProposal, ...] = Field(max_length=10)


EvidenceSourceStage = Literal[
    "first_stage",
    "hypothesis_route",
    "second_stage",
]


class SelectedSpecificationEvidence(_StrictModel):
    evidence_id: str = Field(min_length=1)
    source_stage: EvidenceSourceStage
    chunk_id: str = Field(min_length=1)
    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    version: str = Field(min_length=1)
    release: int | None = Field(default=None, ge=1)
    source_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")
    heading_path: tuple[str, ...] = ()
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    excerpt: str = Field(min_length=1)
    retrieval_sources: tuple[str, ...] = ()
    relation_kinds: tuple[str, ...] = ()
    bridge_terms: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _coherent_release(self) -> "SelectedSpecificationEvidence":
        expected = release_from_version(self.version)
        if self.release is not None and self.release != expected:
            raise ValueError("evidence Release does not match its exact version")
        object.__setattr__(self, "release", expected)
        return self


class SelectedEvidenceGroup(_StrictModel):
    group_id: str = Field(min_length=1)
    source_stages: tuple[EvidenceSourceStage, ...] = Field(min_length=1)
    hypothesis_ids: tuple[str, ...] = ()
    selection_basis: Literal[
        "hypothesis_assessment",
        "original_question_independent",
    ] = "hypothesis_assessment"
    evidence: tuple[SelectedSpecificationEvidence, ...] = Field(
        min_length=1,
        max_length=4,
    )


class HypothesisEvidenceAssessment(_StrictModel):
    hypothesis_id: str = Field(min_length=1)
    status: HypothesisSupportStatus
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=4)


class QuestionEvidenceSelection(_StrictModel):
    query_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    angle: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    status: Literal["completed", "no_evidence", "unavailable"]
    hypothesis_assessments: tuple[HypothesisEvidenceAssessment, ...] = ()
    evidence_groups: tuple[SelectedEvidenceGroup, ...] = ()
    warnings: tuple[str, ...] = ()


class DiscoveryEvidenceSelection(_StrictModel):
    schema_version: Literal["discovery-evidence-selection/v1.0"] = (
        "discovery-evidence-selection/v1.0"
    )
    status: Literal["completed", "partial", "unavailable"]
    max_evidence_groups_per_question: int = Field(ge=1, le=10)
    agent_receipts: tuple[CodexAgentReceipt, ...] = ()
    questions: tuple[QuestionEvidenceSelection, ...] = ()
    errors: tuple[str, ...] = ()


class EvidenceGroundedExpansionPlan(_StrictModel):
    schema_version: Literal["evidence-expansion-plan/v1.0"] = (
        "evidence-expansion-plan/v1.0"
    )
    planner_profile: Literal["protocol-agnostic-retrieval-planner/v1.0"] = (
        PLANNER_PROFILE
    )
    expansion_agent_receipts: tuple[CodexAgentReceipt, ...] = Field(min_length=1)
    retrievals: tuple[EvidenceGroundedFollowup, ...]


class DualStageRetrievalExecution(_StrictModel):
    schema_version: Literal["dual-stage-retrieval-execution/v1.0"] = (
        "dual-stage-retrieval-execution/v1.0"
    )
    status: Literal["completed", "partial", "unavailable"]
    resolved_first_stage_plan: DiscoveryRetrievalPlanV3 | None = None
    relation_evidence: tuple[GroundedRelationEvidence, ...] = ()
    expansion_plan: EvidenceGroundedExpansionPlan | None = None
    first_stage: PlannedRetrievalExecution
    second_stage: PlannedRetrievalExecution | None = None
    merged_tasks: tuple[PlannedRetrievalTaskResult, ...] = ()
    errors: tuple[str, ...] = ()


__all__ = [name for name in globals() if not name.startswith("_")]

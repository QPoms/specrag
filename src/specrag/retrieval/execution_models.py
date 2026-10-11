"""Retrieval execution, expansion, and curation result contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import (
    CorpusVersionPolicy,
    SpecificationVersion,
    release_from_version,
    validate_corpus_scope,
)
from .plan_models import (
    DiscoveryRetrievalPlanV3,
    GroundedTerm,
    PLANNER_PROFILE,
    RetrievalProbe,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlannedRetrievalHit(_StrictModel):
    hit_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    collection_rank: int = Field(ge=1)
    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    version: str = Field(min_length=1)
    release: int | None = Field(default=None, ge=1)
    source_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")
    heading_path: tuple[str, ...] = ()
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    excerpt: str = Field(min_length=1)
    evidence_roles: tuple[str, ...] = Field(min_length=1)
    probe_ids: tuple[str, ...] = Field(min_length=1)
    retrieval_sources: tuple[str, ...] = ()
    matched_anchors: tuple[str, ...] = ()
    retrieval_hop: Literal[1, 2] = 1
    reference_source_chunk_ids: tuple[str, ...] = ()
    semantic_rank: int | None = Field(default=None, ge=1)
    lexical_rank: int | None = Field(default=None, ge=1)
    rrf_score: float = Field(ge=0)
    route_score: float = Field(ge=0)
    anchor_score: float = Field(ge=0)
    heading_score: float = Field(ge=0)
    agreement_score: float = Field(ge=0)
    noise_penalty: float = Field(ge=0)
    deterministic_score: float | None = None
    reranker_score: float | None = None
    deterministic_score_normalized: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    reranker_score_normalized: float | None = Field(default=None, ge=0, le=1)
    reranker_model: str | None = None
    final_score: float

    @model_validator(mode="after")
    def _coherent_source_identity(self) -> "PlannedRetrievalHit":
        expected = release_from_version(self.version)
        if self.release is not None and self.release != expected:
            raise ValueError("hit Release does not match its exact version")
        object.__setattr__(self, "release", expected)
        return self


class PlannedRetrievalTaskResult(_StrictModel):
    retrieval_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    planner_action: Literal["planned", "fallback_full_corpus"]
    status: Literal["completed", "partial", "no_hits", "unavailable"]
    unavailable_specifications: tuple[str, ...] = ()
    followed_references: tuple[str, ...] = ()
    reranker_status: Literal["disabled", "applied", "fallback"] = "disabled"
    reranker_error: str | None = None
    hits: tuple[PlannedRetrievalHit, ...] = ()
    errors: tuple[str, ...] = ()


class PlannedRetrievalExecution(_StrictModel):
    schema_version: Literal["planned-retrieval-execution/v1.1"] = (
        "planned-retrieval-execution/v1.1"
    )
    planner_profile: Literal["protocol-agnostic-retrieval-planner/v1.0"] = (
        PLANNER_PROFILE
    )
    status: Literal["completed", "partial", "unavailable"]
    resolved_plan: DiscoveryRetrievalPlanV3 | None = None
    index_fingerprint: str | None = None
    index_manifest_path: str | None = None
    embedding_model: str
    reranker: dict[str, Any] | None = None
    corpus_version_policy: CorpusVersionPolicy
    corpus_release: int | None = Field(default=None, ge=1)
    corpus_versions: tuple[SpecificationVersion, ...] = ()
    tasks: tuple[PlannedRetrievalTaskResult, ...] = ()
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _coherent_corpus_scope(self) -> "PlannedRetrievalExecution":
        validate_corpus_scope(
            self.corpus_version_policy,
            self.corpus_release,
            self.corpus_versions,
        )
        return self


class ReleaseComparisonExecution(_StrictModel):
    """Paired evidence runs over two isolated, uniform Release snapshots."""

    schema_version: Literal["release-comparison-execution/v1.0"] = (
        "release-comparison-execution/v1.0"
    )
    status: Literal["completed", "partial", "unavailable"]
    base_release: int = Field(ge=1)
    target_release: int = Field(ge=1)
    base_execution: PlannedRetrievalExecution
    target_execution: PlannedRetrievalExecution
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _isolated_release_scopes(self) -> "ReleaseComparisonExecution":
        if self.base_release == self.target_release:
            raise ValueError("release comparison requires two different Releases")
        for label, release, execution in (
            ("base", self.base_release, self.base_execution),
            ("target", self.target_release, self.target_execution),
        ):
            if execution.corpus_version_policy != "highest_version_within_release":
                raise ValueError(f"{label} execution is not Release-scoped")
            if execution.corpus_release != release:
                raise ValueError(
                    f"{label} execution Release does not match comparison scope"
                )
        return self


ExpansionIntent = Literal[
    "definition",
    "representation",
    "field_structure",
    "encoding",
    "carrier_usage",
    "procedure_usage",
    "transport_path",
    "endpoint_interface",
    "direction",
    "authority",
    "security_protection",
    "state_condition",
    "timing_order",
    "prerequisites",
    "relationship",
    "resource_constraint",
    "explicit_reference",
]


class FollowupProbeProposal(_StrictModel):
    followup_id: str = Field(min_length=1)
    parent_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    intent: ExpansionIntent
    selected_term_ids: tuple[str, ...] = Field(min_length=1, max_length=8)


class QuestionExpansionProposal(_StrictModel):
    query_id: str = Field(min_length=1)
    probes: tuple[FollowupProbeProposal, ...] = Field(max_length=5)


class EvidenceExpansionBatch(_StrictModel):
    questions: tuple[QuestionExpansionProposal, ...]


class EvidenceGroundedFollowup(_StrictModel):
    retrieval_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    angle: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    status: Literal["expanded", "no_expansion"]
    warnings: tuple[str, ...] = ()
    retrieval_probes: tuple[RetrievalProbe, ...] = ()

    @model_validator(mode="after")
    def _same_role_second_stage(self) -> "EvidenceGroundedFollowup":
        if self.status == "expanded" and not self.retrieval_probes:
            raise ValueError("expanded follow-up must contain probes")
        if self.status == "no_expansion" and self.retrieval_probes:
            raise ValueError("no-expansion follow-up cannot contain probes")
        if any(probe.stage != 2 for probe in self.retrieval_probes):
            raise ValueError("follow-up probes must be second-stage probes")
        if any(probe.evidence_role != self.angle for probe in self.retrieval_probes):
            raise ValueError("follow-up probes must preserve the Discovery angle")
        return self


class GroundedRelationEvidence(_StrictModel):
    """Corpus-derived evidence made available to the second-stage agent."""

    evidence_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
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
    matched_original_anchors: tuple[str, ...] = ()
    bridge_terms: tuple[str, ...] = ()
    grounded_terms: tuple[GroundedTerm, ...] = ()
    relation_kinds: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_release(self) -> "GroundedRelationEvidence":
        expected = release_from_version(self.version)
        if self.release is not None and self.release != expected:
            raise ValueError("evidence Release does not match its exact version")
        object.__setattr__(self, "release", expected)
        return self


EvidenceRelationClass = Literal[
    "direct_relation",
    "supporting_relation",
    "context_only",
]


class CuratedEvidenceItem(_StrictModel):
    """One deduplicated candidate selected for downstream evidence review."""

    evidence_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
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
    relation_class: EvidenceRelationClass
    classification_reasons: tuple[str, ...] = Field(min_length=1)
    candidate_sources: tuple[str, ...] = Field(min_length=1)
    source_evidence_ids: tuple[str, ...] = Field(min_length=1)
    matched_original_anchors: tuple[str, ...] = ()
    grounded_terms: tuple[str, ...] = ()
    curation_score: float

    @model_validator(mode="after")
    def _coherent_release(self) -> "CuratedEvidenceItem":
        expected = release_from_version(self.version)
        if self.release is not None and self.release != expected:
            raise ValueError("evidence Release does not match its exact version")
        object.__setattr__(self, "release", expected)
        return self


class CuratedEvidenceTask(_StrictModel):
    query_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    angle: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    candidate_count: int = Field(ge=0)
    direct_candidate_count: int = Field(ge=0)
    supporting_candidate_count: int = Field(ge=0)
    context_candidate_count: int = Field(ge=0)
    selected_evidence: tuple[CuratedEvidenceItem, ...] = ()


class CuratedRetrievalExecution(_StrictModel):
    schema_version: Literal["curated-retrieval-execution/v1.0"] = (
        "curated-retrieval-execution/v1.0"
    )
    status: Literal["completed", "partial", "unavailable"]
    resolved_plan: DiscoveryRetrievalPlanV3 | None = None
    first_stage: PlannedRetrievalExecution
    relation_candidates: tuple[GroundedRelationEvidence, ...] = ()
    curated_tasks: tuple[CuratedEvidenceTask, ...] = ()
    errors: tuple[str, ...] = ()


class ExpandedRetrievalExecution(_StrictModel):
    schema_version: Literal["expanded-retrieval-execution/v1.0"] = (
        "expanded-retrieval-execution/v1.0"
    )
    status: Literal["completed", "partial", "unavailable"]
    resolved_plan: DiscoveryRetrievalPlanV3 | None = None
    first_stage: PlannedRetrievalExecution
    second_stage_candidates: tuple[GroundedRelationEvidence, ...] = ()
    errors: tuple[str, ...] = ()


__all__ = [
    "CuratedEvidenceItem",
    "CuratedEvidenceTask",
    "CuratedRetrievalExecution",
    "EvidenceExpansionBatch",
    "EvidenceGroundedFollowup",
    "EvidenceRelationClass",
    "ExpandedRetrievalExecution",
    "ExpansionIntent",
    "FollowupProbeProposal",
    "GroundedRelationEvidence",
    "PlannedRetrievalExecution",
    "PlannedRetrievalHit",
    "PlannedRetrievalTaskResult",
    "QuestionExpansionProposal",
    "ReleaseComparisonExecution",
]

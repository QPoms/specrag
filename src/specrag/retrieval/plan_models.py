"""Immutable contracts for extractive planning and specification routing."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..infrastructure.structured_agent import CodexAgentReceipt


PLANNER_PROFILE = "protocol-agnostic-retrieval-planner/v1.0"
AnchorType = Literal[
    "protocol_message",
    "information_element",
    "protocol_state",
    "timer",
    "specification",
    "literal_value",
    "technical_term",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSpan(_StrictModel):
    """Exact, half-open character range in the original Discovery question."""

    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def _valid_half_open_range(self) -> "SourceSpan":
        if self.end <= self.start:
            raise ValueError("source span end must be greater than start")
        if self.end - self.start != len(self.text):
            raise ValueError("source span offsets must match source span text")
        return self


class QueryAnchor(_StrictModel):
    surface: str = Field(min_length=1)
    canonical: str = Field(min_length=1)
    entity_type: AnchorType
    source_span: SourceSpan
    normalization: Literal["verbatim", "orthographic", "controlled_alias"]

    @model_validator(mode="after")
    def _surface_matches_span(self) -> "QueryAnchor":
        if self.surface != self.source_span.text:
            raise ValueError("anchor surface must equal source_span.text")
        return self


class ExplicitFacet(_StrictModel):
    """A role-specific information need explicitly stated in the question."""

    name: str = Field(min_length=1)
    source_span: SourceSpan


class EvidenceTermProposal(_StrictModel):
    """Verbatim term selected from one first-stage hit excerpt."""

    source_hit_id: str = Field(min_length=1)
    surface: str = Field(min_length=1)


class GroundedTerm(_StrictModel):
    """A code-assigned selectable term; the agent returns only its ID."""

    term_id: str = Field(min_length=1)
    source_evidence_id: str = Field(min_length=1)
    surface: str = Field(min_length=1)


class LLMAnchorProposal(_StrictModel):
    """An extractive proposal; normalization is performed by trusted code."""

    surface: str = Field(min_length=1)
    entity_type: AnchorType


class LLMQuestionAnchors(_StrictModel):
    query_id: str = Field(min_length=1)
    anchors: tuple[LLMAnchorProposal, ...]


class LLMAnchorBatch(_StrictModel):
    questions: tuple[LLMQuestionAnchors, ...]


class SpecificationRoute(_StrictModel):
    specification: str = Field(pattern=r"^\d{2}\.\d{3}$")
    weight: float = Field(gt=0, le=1)
    reason: str = Field(min_length=1)
    source: Literal[
        "document_router",
        "explicit_query",
        "reference",
        "agent_hypothesis",
    ]


class RetrievalProbe(_StrictModel):
    probe_id: str = Field(min_length=1)
    evidence_role: str = Field(min_length=1)
    lexical_query: str = Field(min_length=1)
    semantic_query: str = Field(min_length=1)
    routes: tuple[SpecificationRoute, ...] = ()
    anchors: tuple[str, ...] = ()
    facets: tuple[str, ...] = ()
    source_hit_ids: tuple[str, ...] = ()
    evidence_terms: tuple[EvidenceTermProposal, ...] = ()
    stage: Literal[1, 2] = 1
    weight: float = Field(default=1.0, gt=0)

    @field_validator("anchors", "facets", "source_hit_ids")
    @classmethod
    def _unique_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        folded = [item.casefold() for item in value]
        if len(folded) != len(set(folded)):
            raise ValueError("probe anchors and facets must be unique")
        return value


RetrievalSubquery = RetrievalProbe


class RoutingPolicy(_StrictModel):
    strategy: Literal["document_then_passage"] = "document_then_passage"
    max_candidate_specifications: int = Field(default=6, ge=1, le=12)
    fallback_to_full_corpus: bool = True


class ReferenceExpansionPolicy(_StrictModel):
    enabled: bool = True
    max_hops: Literal[1] = 1
    max_references: int = Field(default=3, ge=0, le=10)


class PlannedDiscoveryRetrieval(_StrictModel):
    retrieval_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    angle: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    planner_action: Literal["planned", "fallback_full_corpus"]
    anchors: tuple[QueryAnchor, ...] = ()
    explicit_facets: tuple[ExplicitFacet, ...] = ()
    context_dependent: bool = False
    warnings: tuple[str, ...] = ()
    routing_policy: RoutingPolicy = RoutingPolicy()
    reference_expansion: ReferenceExpansionPolicy = ReferenceExpansionPolicy()
    retrieval_probes: tuple[RetrievalProbe, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_plan(self) -> "PlannedDiscoveryRetrieval":
        identifiers = [item.probe_id for item in self.retrieval_probes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("probe IDs must be unique within a retrieval")
        if any(item.evidence_role != self.angle for item in self.retrieval_probes):
            raise ValueError("every probe must preserve the frozen Discovery angle")
        return self

    @property
    def subqueries(self) -> tuple[RetrievalProbe, ...]:
        return self.retrieval_probes


class DiscoveryRetrievalPlanV3(_StrictModel):
    schema_version: Literal["discovery-retrieval-plan/v3.0"] = (
        "discovery-retrieval-plan/v3.0"
    )
    planner_profile: Literal["protocol-agnostic-retrieval-planner/v1.0"] = (
        PLANNER_PROFILE
    )
    anchor_extraction_method: Literal[
        "deterministic_extractive",
        "codex_agent_extractive_validated",
    ] = "deterministic_extractive"
    planner_agent_receipt: CodexAgentReceipt | None = None
    retrievals: tuple[PlannedDiscoveryRetrieval, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_retrievals(self) -> "DiscoveryRetrievalPlanV3":
        retrieval_ids = [item.retrieval_id for item in self.retrievals]
        query_ids = [item.query_id for item in self.retrievals]
        if len(retrieval_ids) != len(set(retrieval_ids)):
            raise ValueError("planned retrieval IDs must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("each Discovery question must be planned once")
        return self


DiscoveryRetrievalPlanV2 = DiscoveryRetrievalPlanV3


__all__ = [
    "AnchorType",
    "DiscoveryRetrievalPlanV2",
    "DiscoveryRetrievalPlanV3",
    "EvidenceTermProposal",
    "ExplicitFacet",
    "GroundedTerm",
    "LLMAnchorBatch",
    "LLMAnchorProposal",
    "LLMQuestionAnchors",
    "PLANNER_PROFILE",
    "PlannedDiscoveryRetrieval",
    "QueryAnchor",
    "ReferenceExpansionPolicy",
    "RetrievalProbe",
    "RetrievalSubquery",
    "RoutingPolicy",
    "SourceSpan",
    "SpecificationRoute",
]

"""Protocol-agnostic, auditable planning for Discovery retrieval.

The Discovery question and its frozen angle are the semantic authority. The
planner may select and normalize spans present in the question, but it must not
add an unstated message, IE, interface, state, or other protocol fact.
Candidate specifications are resolved separately by corpus-level document
routing. Cross-specification expansion is permitted only when a first-hop
passage contains an explicit TS reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Literal, Mapping, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config import AppConfig, load_config
from .hybrid import (
    CorpusVersionPolicy,
    HybridMarkdownRetriever,
    MarkdownChunk,
)
from .index import searchable_text
from ..infrastructure.structured_agent import CodexAgentReceipt, CodexStructuredAgent
from .reranker import CrossEncoderReranker
from .normalization import (
    protocol_identifier_in_text,
    protocol_identifier_variants,
)


PLANNER_PROFILE = "protocol-agnostic-retrieval-planner/v1.0"
_SPEC_PATTERN = r"^\d{2}\.\d{3}$"
_LOW_VALUE_HEADINGS = (
    "change history",
    "handling of unknown, unforeseen",
    "handling of unknown unforeseen",
    "foreword",
    "references",
)
_DEICTIC_REFERENCE = re.compile(
    r"\b(?:this|these|those|such|aforementioned)"
    r"(?:\s+[a-z][a-z-]*){0,4}\s+"
    r"(?:claim|object|objects|message|messages|information|procedure|procedures)\b",
    re.I,
)
_SPEC_REFERENCE_PATTERNS = (
    re.compile(
        r"(?P<citation>(?:3GPP\s+)?TS\s+(?P<spec>\d{2}\.\d{3})"
        r"(?:\s*(?:\[[^\]]+\])?\s*,?\s*"
        r"(?:clause|subclause)\s+(?P<clause>\d+(?:\.\d+)*[A-Za-z]?))?)",
        re.I,
    ),
    re.compile(
        r"(?P<citation>(?:clause|subclause)\s+"
        r"(?P<clause>\d+(?:\.\d+)*[A-Za-z]?)\s+of\s+"
        r"(?:3GPP\s+)?TS\s+(?P<spec>\d{2}\.\d{3}))",
        re.I,
    ),
)

AnchorType = Literal[
    "protocol_message",
    "information_element",
    "protocol_state",
    "timer",
    "specification",
    "literal_value",
    "technical_term",
]
_T = TypeVar("_T")


def _resolve_evidence_id_aliases(
    evidence_ids: tuple[str, ...],
    available_ids: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Resolve only unique rank-shortened evidence IDs from the agent.

    Codex occasionally returns ``HB-D4:2`` instead of the supplied
    ``HB-D4:3gpp_markdown:2``.  Accept that form only when it maps to one and
    only one supplied candidate; ambiguous or unrelated IDs remain invalid.
    """

    resolved: list[str] = []
    unknown: list[str] = []
    aliases: list[tuple[str, str]] = []
    for value in evidence_ids:
        if value in available_ids:
            resolved.append(value)
            continue
        prefix, separator, rank = value.rpartition(":")
        matches = tuple(sorted(
            candidate
            for candidate in available_ids
            if separator
            and rank.isdigit()
            and candidate.startswith(prefix + ":")
            and candidate.rpartition(":")[2] == rank
        ))
        if len(matches) == 1:
            resolved.append(matches[0])
            aliases.append((value, matches[0]))
        else:
            unknown.append(value)
    return tuple(resolved), tuple(unknown), tuple(aliases)


# These words describe the requested evidence shape. They are facets, not
# protocol entities, and must never compete with concrete identifiers during
# lexical retrieval.
_GENERIC_FACET_TERMS = frozenset({
    "channel",
    "channels",
    "container",
    "containers",
    "endpoint",
    "endpoints",
    "field",
    "fields",
    "ie",
    "ies",
    "information element",
    "information elements",
    "interface",
    "interfaces",
    "message",
    "messages",
    "path",
    "paths",
    "procedure",
    "procedures",
    "protocol message",
    "protocol messages",
    "ue",
    "network",
})


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
    specification: str = Field(pattern=_SPEC_PATTERN)
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


# Compatibility name for callers of the experimental v0.1 module.
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
        """Read-only compatibility view; new artifacts use retrieval_probes."""

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


# Compatibility alias. The serialized schema version distinguishes artifacts.
DiscoveryRetrievalPlanV2 = DiscoveryRetrievalPlanV3


class PlannedRetrievalHit(_StrictModel):
    hit_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    collection_rank: int = Field(ge=1)
    specification: str = Field(pattern=_SPEC_PATTERN)
    version: str = Field(min_length=1)
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
    schema_version: Literal["planned-retrieval-execution/v1.0"] = (
        "planned-retrieval-execution/v1.0"
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
    tasks: tuple[PlannedRetrievalTaskResult, ...] = ()
    errors: tuple[str, ...] = ()


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
    selected_term_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
    )


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
    specification: str = Field(pattern=_SPEC_PATTERN)
    version: str = Field(min_length=1)
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
    specification: str = Field(pattern=_SPEC_PATTERN)
    version: str = Field(min_length=1)
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
    # Required in the structured response schema; null means an independent answer.
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

    @field_validator(
        "repair_id",
        "missing_claim",
        "rationale",
    )
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
    ] = (
        "not_assessed"
    )
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
    """Evidence IDs forming one distinct answer-bearing source group."""

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
    """Direct answer evidence found outside the hypothesis checklist."""

    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=4)


class AgentQuestionEvidenceSelectionProposal(_StrictModel):
    query_id: str = Field(min_length=1)
    hypothesis_assessments: tuple[
        AgentHypothesisEvidenceAssessmentProposal, ...
    ]
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
    """One unmodified specification excerpt selected for downstream work."""

    evidence_id: str = Field(min_length=1)
    source_stage: EvidenceSourceStage
    chunk_id: str = Field(min_length=1)
    specification: str = Field(pattern=_SPEC_PATTERN)
    version: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")
    heading_path: tuple[str, ...] = ()
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    excerpt: str = Field(min_length=1)
    retrieval_sources: tuple[str, ...] = ()
    relation_kinds: tuple[str, ...] = ()
    bridge_terms: tuple[str, ...] = ()


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


def _angle_value(question: Any) -> str:
    angle = question.angle
    return str(getattr(angle, "value", angle))


def _span(text: str, start: int, end: int) -> SourceSpan:
    return SourceSpan(text=text[start:end], start=start, end=end)


def _canonicalize(
    surface: str,
) -> tuple[str, Literal["verbatim", "orthographic"]]:
    canonical = re.sub(r"[\u2010-\u2015]", "-", surface)
    canonical = re.sub(r"\s+", " ", canonical).strip()
    return canonical, "verbatim" if canonical == surface else "orthographic"


def _classify_anchor(surface: str) -> AnchorType:
    folded = surface.casefold()
    if re.fullmatch(r"(?:3gpp\s+)?ts\s+\d{2}\.\d{3}", folded):
        return "specification"
    if re.fullmatch(r"t\d{3,5}", folded):
        return "timer"
    if re.fullmatch(r"#\d+", folded):
        return "literal_value"
    if folded.endswith((" ie", " information element")):
        return "information_element"
    if folded.endswith((
        " request",
        " response",
        " reject",
        " accept",
        " command",
        " complete",
        " indication",
        " notification",
        " message",
    )):
        return "protocol_message"
    if any(value in folded for value in ("-registered", "-idle", "-connected")):
        return "protocol_state"
    return "technical_term"


_TECHNICAL_GROUP = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Z0-9][A-Z0-9]*(?:[-\u2010-\u2015_/][A-Z0-9]+)*(?:\(s\))?)"
    r"(?:\s+(?:[A-Z0-9][A-Z0-9]*(?:[-\u2010-\u2015_/][A-Z0-9]+)*"
    r"(?:\(s\))?)){0,5}(?![A-Za-z0-9])"
)
_MESSAGE_PHRASE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*\s+){1,5}"
    r"(?:Request|Response|Reject|Accept|Command|Complete|Indication|"
    r"Notification|Message)\b"
)
_IE_PHRASE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*(?:\(s\))?\s+){1,5}IE\b"
)
_EXPLICIT_SPEC = re.compile(r"(?:3GPP\s+)?TS\s+\d{2}\.\d{3}", re.I)
_TIMER = re.compile(r"\bT\d{3,5}\b")
_LITERAL = re.compile(r"#\d+\b")
_ANCHOR_STOPWORDS = frozenset({
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BY", "CAN", "DOES",
    "FOR", "FROM", "HOW", "IF", "IN", "IS", "IT", "OF", "ON", "OR",
    "THE", "THIS", "TO", "UNDER", "WHAT", "WHEN", "WHETHER", "WHICH",
    "WITH", "WITHOUT",
})


def _extract_anchors(text: str) -> tuple[QueryAnchor, ...]:
    candidates: list[tuple[int, int]] = []
    for pattern in (
        _EXPLICIT_SPEC,
        _MESSAGE_PHRASE,
        _IE_PHRASE,
        _TIMER,
        _LITERAL,
        _TECHNICAL_GROUP,
    ):
        candidates.extend(
            (match.start(), match.end()) for match in pattern.finditer(text)
        )

    selected: list[tuple[int, int]] = []
    for start, end in sorted(
        candidates,
        key=lambda item: (item[0], -(item[1] - item[0])),
    ):
        surface = text[start:end].strip()
        if not surface:
            continue
        adjusted_start = text.find(surface, start, end)
        adjusted_end = adjusted_start + len(surface)
        tokens = surface.upper().split()
        if tokens and all(token in _ANCHOR_STOPWORDS for token in tokens):
            continue
        if surface.casefold() in _GENERIC_FACET_TERMS:
            continue
        if any(
            start0 <= adjusted_start and adjusted_end <= end0
            for start0, end0 in selected
        ):
            continue
        selected.append((adjusted_start, adjusted_end))

    anchors: list[QueryAnchor] = []
    seen: set[str] = set()
    for start, end in sorted(selected):
        surface = text[start:end]
        canonical, normalization = _canonicalize(surface)
        key = canonical.casefold()
        if key in seen:
            continue
        seen.add(key)
        anchors.append(QueryAnchor(
            surface=surface,
            canonical=canonical,
            entity_type=_classify_anchor(surface),
            source_span=_span(text, start, end),
            normalization=normalization,
        ))
    return tuple(anchors)


def _validated_llm_anchors(
    text: str,
    proposals: tuple[LLMAnchorProposal, ...],
) -> tuple[QueryAnchor, ...]:
    anchors: list[QueryAnchor] = []
    for proposal in proposals:
        start = text.find(proposal.surface)
        if start < 0:
            raise ValueError(
                "Codex anchor does not occur verbatim in the question"
            )
        end = start + len(proposal.surface)
        source = proposal.surface
        canonical, normalization = _canonicalize(source)
        anchors.append(QueryAnchor(
            surface=source,
            canonical=canonical,
            entity_type=proposal.entity_type,
            source_span=_span(text, start, end),
            normalization=normalization,
        ))
    return tuple(anchors)


def _merge_anchors(
    *groups: tuple[QueryAnchor, ...],
) -> tuple[QueryAnchor, ...]:
    # Prefer a longer extractive phrase to a contained token. This lets an LLM
    # identify natural-language protocol names while retaining strict spans.
    candidates = [anchor for group in groups for anchor in group]
    selected: list[QueryAnchor] = []
    for anchor in sorted(
        candidates,
        key=lambda item: (
            item.source_span.start,
            -(item.source_span.end - item.source_span.start),
        ),
    ):
        span = anchor.source_span
        if any(
            existing.source_span.start <= span.start
            and span.end <= existing.source_span.end
            for existing in selected
        ):
            continue
        if any(existing.canonical.casefold() == anchor.canonical.casefold() for existing in selected):
            continue
        selected.append(anchor)
    return tuple(sorted(selected, key=lambda item: item.source_span.start))


_FACET_CATALOG: dict[str, tuple[tuple[str, str], ...]] = {
    "format": (
        ("representation", r"\brepresent(?:ation|ed|s|ing)?\b"),
        ("format", r"\bformat\b"),
        ("structure", r"\bstructur(?:e|ed)\b"),
        ("fields", r"\bfields?\b"),
        ("encoding", r"\bencod(?:e|ed|ing)\b"),
        ("values", r"\bvalues?\b"),
    ),
    "carrier": (
        ("protocol_message", r"\b(?:protocol\s+)?messages?\b"),
        ("information_element", r"\binformation elements?\b|\bIEs?\b"),
        ("container", r"\bcontainers?\b"),
        ("procedure", r"\bprocedures?\b"),
        ("channel", r"\bchannels?\b"),
    ),
    "reachability": (
        ("endpoint", r"\bendpoints?\b"),
        ("interface", r"\binterfaces?\b"),
        ("direction", r"\bdirections?\b|\buplink\b|\bdownlink\b"),
        ("transport", r"\btransport\b"),
        ("path", r"\bpaths?\b"),
        ("encapsulation", r"\bencapsulation\b"),
        ("channel", r"\bchannels?\b"),
    ),
    "authority": (
        ("role", r"\broles?\b"),
        ("identity", r"\bidentity\b"),
        ("authorization", r"\bauthori[sz](?:ation|ed)\b"),
        ("entity", r"\bentit(?:y|ies)\b"),
    ),
    "protection": (
        ("integrity", r"\bintegrity(?:-protection| protection)?\b"),
        ("confidentiality", r"\bconfidentiality\b"),
        ("authentication", r"\bauthentication\b"),
        ("replay", r"\breplay\b"),
        ("freshness", r"\bfreshness\b"),
        ("security_context", r"\bsecurity(?:-| )context\b"),
        ("protection", r"\bprotection\b"),
    ),
    "state_timing": (
        ("state", r"\bstates?\b"),
        ("phase", r"\bphases?\b"),
        ("timing", r"\btiming\b"),
        ("timer", r"\btimers?\b"),
        ("condition", r"\bconditions?\b"),
        ("ordering", r"\bordering\b"),
        ("window", r"\bwindows?\b"),
        ("context", r"\bcontexts?\b"),
    ),
    "required_inputs": (
        ("prerequisite", r"\bprerequisites?\b"),
        ("mandatory", r"\bmandatory\b"),
        ("conditional", r"\bconditionally?\b"),
        ("input", r"\binputs?\b"),
    ),
    "relation": (
        ("mapping", r"\bmapping\b"),
        ("relationship", r"\brelationships?\b"),
        ("association", r"\bassociate(?:s|d)?\b|\bassociation\b"),
        ("link", r"\blinks?\b"),
    ),
    "resource_constraints": (
        ("repetition", r"\brepetition\b|\brepeated\b|\bsuccessive\b"),
        ("rate", r"\brate\b"),
        ("duration", r"\bduration\b"),
        ("constraint", r"\bconstraints?\b"),
        ("attempt", r"\battempts?\b"),
    ),
}


def _extract_facets(text: str, angle: str) -> tuple[ExplicitFacet, ...]:
    matches: list[tuple[int, int, str]] = []
    matched_names: set[str] = set()
    for name, pattern in _FACET_CATALOG.get(angle, ()):
        for match in re.finditer(pattern, text, re.I):
            if name in matched_names:
                break
            if any(
                start <= match.start() and match.end() <= end
                for start, end, _ in matches
            ):
                continue
            matches.append((match.start(), match.end(), name))
            matched_names.add(name)
    return tuple(
        ExplicitFacet(name=name, source_span=_span(text, start, end))
        for start, end, name in sorted(matches)
    )


def _partition(values: tuple[_T, ...], count: int) -> tuple[tuple[_T, ...], ...]:
    groups: list[list[_T]] = [[] for _ in range(count)]
    for index, value in enumerate(values):
        groups[index % count].append(value)
    return tuple(tuple(group) for group in groups if group)


def _compile_probes(
    text: str,
    angle: str,
    anchors: tuple[QueryAnchor, ...],
    facets: tuple[ExplicitFacet, ...],
) -> tuple[RetrievalProbe, ...]:
    if not anchors:
        return (RetrievalProbe(
            probe_id="P1",
            evidence_role=angle,
            lexical_query=text,
            semantic_query=text,
        ),)

    # One explicit aspect per probe keeps dense retrieval meaningfully
    # different across message, IE, container, path, and similar facets.
    facet_groups = tuple((facet,) for facet in facets[:5]) or ((),)
    anchor_groups = (anchors,) * len(facet_groups)

    probes: list[RetrievalProbe] = []
    for index, (anchor_group, facet_group) in enumerate(
        zip(anchor_groups, facet_groups, strict=True),
        1,
    ):
        lexical_parts = [anchor.surface for anchor in anchor_group]
        lexical_parts.extend(facet.source_span.text for facet in facet_group)
        lexical = " ".join(dict.fromkeys(lexical_parts)).strip() or text
        semantic = _facet_semantic_query(
            text,
            angle,
            anchor_group,
            facet_group,
        )
        probes.append(RetrievalProbe(
            probe_id=f"P{index}",
            evidence_role=angle,
            lexical_query=lexical,
            semantic_query=semantic,
            anchors=tuple(anchor.canonical for anchor in anchor_group),
            facets=tuple(facet.name for facet in facet_group),
        ))
    return tuple(probes)


def _facet_semantic_query(
    original_question: str,
    angle: str,
    anchors: tuple[QueryAnchor, ...],
    facets: tuple[ExplicitFacet, ...],
) -> str:
    """Build a source-faithful semantic query focused on one explicit facet."""

    if not facets:
        return original_question
    concrete = "; ".join(anchor.canonical for anchor in anchors)
    requested = "; ".join(facet.source_span.text for facet in facets)
    if not concrete:
        return original_question
    templates = {
        "carrier": "Which {facet} conveys or carries {anchors}?",
        "format": "What {facet} specifies the representation of {anchors}?",
        "reachability": "What {facet} describes the reachability of {anchors}?",
        "authority": "What {facet} specifies authority for {anchors}?",
        "protection": "What {facet} specifies protection of {anchors}?",
        "state_timing": "What {facet} specifies state or timing for {anchors}?",
        "required_inputs": "What {facet} specifies required inputs for {anchors}?",
        "relation": "What {facet} specifies the relationship of {anchors}?",
        "resource_constraints": "What {facet} specifies constraints on {anchors}?",
    }
    template = templates.get(angle)
    if template is None:
        return f"{concrete}. {requested}. {angle}."
    return template.format(facet=requested, anchors=concrete)


def _validate_source_grounding(item: PlannedDiscoveryRetrieval) -> None:
    text = item.original_question
    for anchor in item.anchors:
        span = anchor.source_span
        if text[span.start:span.end] != span.text:
            raise ValueError(f"ungrounded anchor in {item.query_id}: {anchor.surface}")
    for facet in item.explicit_facets:
        span = facet.source_span
        if text[span.start:span.end] != span.text:
            raise ValueError(f"ungrounded facet in {item.query_id}: {facet.name}")


def _has_unresolved_reference(
    text: str,
    anchors: tuple[QueryAnchor, ...],
) -> bool:
    for match in _DEICTIC_REFERENCE.finditer(text):
        if not any(
            match.start() <= anchor.source_span.start
            and anchor.source_span.end <= match.end()
            for anchor in anchors
        ):
            return True
    return False


def _compile_one(
    question: Any,
    llm_anchors: tuple[QueryAnchor, ...] = (),
) -> PlannedDiscoveryRetrieval:
    text = question.question
    angle = _angle_value(question)
    anchors = _merge_anchors(llm_anchors, _extract_anchors(text))
    facets = _extract_facets(text, angle)
    context_dependent = _has_unresolved_reference(text, anchors)
    warnings: list[str] = []
    if context_dependent:
        warnings.append(
            "question contains an unresolved reference; the planner did not "
            "import entities from another question or claim"
        )
    if not anchors:
        warnings.append(
            "no extractive protocol anchor was found; use full-corpus fallback"
        )
    action: Literal["planned", "fallback_full_corpus"] = (
        "planned" if anchors and not context_dependent else "fallback_full_corpus"
    )
    item = PlannedDiscoveryRetrieval(
        retrieval_id=f"P-{question.discovery_id}",
        query_id=question.discovery_id,
        claim_id=question.claim_id,
        angle=angle,
        original_question=text,
        planner_action=action,
        anchors=anchors,
        explicit_facets=facets,
        context_dependent=context_dependent,
        warnings=tuple(warnings),
        retrieval_probes=_compile_probes(text, angle, anchors, facets),
    )
    _validate_source_grounding(item)
    return item


def compile_discovery_retrieval_plan(
    discovery_questions: tuple[Any, ...],
    *,
    planner_agent: CodexStructuredAgent | None = None,
) -> DiscoveryRetrievalPlanV3:
    """Compile questions without importing unstated protocol facts.

    If a Codex agent is supplied, it is used once for batch extractive NER. Its
    surface terms are accepted only when they occur verbatim in the original
    question. Trusted code, not the agent, performs source locating,
    normalization, and probe construction.
    """

    llm_by_query: dict[str, tuple[QueryAnchor, ...]] = {}
    receipt: CodexAgentReceipt | None = None
    method: Literal[
        "deterministic_extractive",
        "codex_agent_extractive_validated",
    ] = "deterministic_extractive"
    if planner_agent is not None:
        response = planner_agent.parse(
            purpose="extract protocol anchors from Discovery questions",
            prompt_version="discovery-anchor-extraction/v1.0",
            instructions=(
                "Extract only protocol-relevant entities whose exact surface "
                "text occurs verbatim in each question. Do not infer or add a "
                "message, information element, interface, state, specification, "
                "relationship, or other fact absent from the question. Do not "
                "resolve pronouns from other questions or claims. Classify each "
                "surface using only the supplied schema. It is valid to return no "
                "anchors. Return every query_id exactly once."
            ),
            input_payload={
                "questions": [
                    {
                        "query_id": item.discovery_id,
                        "angle": _angle_value(item),
                        "question": item.question,
                    }
                    for item in discovery_questions
                ]
            },
            output_type=LLMAnchorBatch,
        )
        expected_ids = [item.discovery_id for item in discovery_questions]
        returned_ids = [item.query_id for item in response.output.questions]
        if returned_ids != expected_ids:
            raise ValueError(
                "Codex anchor extraction must preserve ordered Discovery query IDs"
            )
        question_by_id = {
            item.discovery_id: item.question for item in discovery_questions
        }
        llm_by_query = {
            item.query_id: _validated_llm_anchors(
                question_by_id[item.query_id],
                item.anchors,
            )
            for item in response.output.questions
        }
        receipt = response.receipt
        method = "codex_agent_extractive_validated"

    return DiscoveryRetrievalPlanV3(
        anchor_extraction_method=method,
        planner_agent_receipt=receipt,
        retrievals=tuple(
            _compile_one(item, llm_by_query.get(item.discovery_id, ()))
            for item in discovery_questions
        ),
    )


def _document_routes(
    retriever: HybridMarkdownRetriever,
    probe: RetrievalProbe,
    *,
    limit: int,
) -> tuple[SpecificationRoute, ...]:
    try:
        dense = retriever._dense(probe.semantic_query, "3gpp_markdown")
    except Exception:
        dense = []
    try:
        lexical = retriever.bm25.search(
            probe.lexical_query,
            source_collection="3gpp_markdown",
            limit=max(retriever.cfg.lexical_candidates_per_collection, limit * 4),
        )
    except Exception:
        lexical = []

    scores: dict[str, float] = {}
    for results in (dense, lexical):
        best_rank: dict[str, int] = {}
        for rank, (chunk_id, _) in enumerate(results, 1):
            specification = retriever.chunk_by_id[chunk_id].specification
            if specification is not None:
                best_rank[specification] = min(
                    best_rank.get(specification, rank),
                    rank,
                )
        for specification, rank in best_rank.items():
            scores[specification] = scores.get(specification, 0.0) + 1 / (
                retriever.cfg.rrf_k + rank
            )
    if not scores:
        return ()
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    maximum = ordered[0][1]
    return tuple(
        SpecificationRoute(
            specification=specification,
            weight=max(0.01, score / maximum),
            reason="hybrid document-level match for the extractive probe",
            source="document_router",
        )
        for specification, score in ordered
    )


def resolve_discovery_retrieval_plan(
    plan: DiscoveryRetrievalPlanV3,
    retriever: HybridMarkdownRetriever,
) -> DiscoveryRetrievalPlanV3:
    """Resolve candidate TSs from the selected corpus without rewriting probes."""

    retrievals: list[PlannedDiscoveryRetrieval] = []
    for item in plan.retrievals:
        explicit_specifications = tuple(dict.fromkeys(
            anchor.canonical.removeprefix("3GPP ").removeprefix("TS ").strip()
            for anchor in item.anchors
            if anchor.entity_type == "specification"
        ))
        routes: tuple[SpecificationRoute, ...] = ()
        if item.planner_action == "planned" and explicit_specifications:
            routes = tuple(
                SpecificationRoute(
                    specification=specification,
                    weight=1.0,
                    reason="specification explicitly named in the question",
                    source="explicit_query",
                )
                for specification in explicit_specifications
            )
        elif item.planner_action == "planned":
            route_scores: dict[str, float] = {}
            for probe in item.retrieval_probes:
                for route in _document_routes(
                    retriever,
                    probe,
                    limit=item.routing_policy.max_candidate_specifications,
                ):
                    route_scores[route.specification] = (
                        route_scores.get(route.specification, 0.0)
                        + route.weight * probe.weight
                    )
            ordered = sorted(
                route_scores.items(),
                key=lambda value: (-value[1], value[0]),
            )[:item.routing_policy.max_candidate_specifications]
            maximum = ordered[0][1] if ordered else 1.0
            routes = tuple(
                SpecificationRoute(
                    specification=specification,
                    weight=max(0.01, score / maximum),
                    reason=(
                        "aggregated hybrid document-level match across "
                        "same-role extractive probes"
                    ),
                    source="document_router",
                )
                for specification, score in ordered
            )
        probes = [
            probe.model_copy(update={"routes": probe.routes or routes})
            for probe in item.retrieval_probes
        ]
        retrievals.append(item.model_copy(update={"retrieval_probes": tuple(probes)}))
    return plan.model_copy(update={"retrievals": tuple(retrievals)})


def _is_heading_only(chunk: MarkdownChunk) -> bool:
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    return bool(lines) and all(re.fullmatch(r"#{1,6}\s+.+", line) for line in lines)


@dataclass
class _Candidate:
    chunk: MarkdownChunk
    rrf_score: float = 0.0
    route_score: float = 0.0
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    exact_anchor_rank: int | None = None
    exact_probe_ranks: dict[str, int] = field(default_factory=dict)
    lexical_probe_ranks: dict[str, int] = field(default_factory=dict)
    probe_ids: set[str] = field(default_factory=set)
    evidence_roles: set[str] = field(default_factory=set)
    anchors: set[str] = field(default_factory=set)
    heading_anchors: set[str] = field(default_factory=set)
    retrieval_hop: Literal[1, 2] = 2
    reference_source_chunk_ids: set[str] = field(default_factory=set)


def _ordered_unique(values: set[str]) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda value: (value.casefold(), value)))


def _fold_protocol_text(text: str) -> str:
    return re.sub(r"[\u2010-\u2015]", "-", text).casefold()


def _exact_match_priority(chunk: MarkdownChunk, probe: RetrievalProbe) -> float:
    """Rank exact entity occurrences for the probe's requested evidence shape."""

    searchable = _fold_protocol_text(searchable_text(chunk))
    heading = _fold_protocol_text(" ".join(chunk.heading_path))
    score = float(sum(
        max(
            (searchable.count(variant)
             for variant in protocol_identifier_variants(anchor)),
            default=0,
        )
        for anchor in probe.anchors
    ))
    if any(protocol_identifier_in_text(anchor, heading)
           for anchor in probe.anchors):
        score += 4.0
    facets = set(probe.facets)
    if "protocol_message" in facets:
        if "message content" in searchable:
            score += 8.0
        if "message type:" in searchable or "direction:" in searchable:
            score += 4.0
        if any(
            suffix in heading
            for suffix in (
                " request", " response", " reject", " accept", " command",
                " complete", " indication", " notification",
            )
        ):
            score += 2.0
    if "information_element" in facets:
        if "type/reference" in searchable or "| iei |" in searchable:
            score += 7.0
        if "purpose of the" in searchable and "information element" in searchable:
            score += 7.0
        if "coded as" in searchable:
            score += 2.0
    if "container" in facets and "container" in searchable:
        score += 5.0
    for facet in probe.facets:
        if facet.replace("_", " ") in searchable:
            score += 1.0
    return score


def _exact_anchor_results(
    retriever: HybridMarkdownRetriever,
    probe: RetrievalProbe,
    *,
    limit: int,
) -> list[tuple[str, float]]:
    """Corpus-wide exact recall so document routing cannot hide an entity."""

    anchors = tuple(
        anchor
        for anchor in probe.anchors
        if anchor.casefold() not in _GENERIC_FACET_TERMS
    )
    if not anchors:
        return []
    scored: list[tuple[str, float]] = []
    for chunk in retriever.chunks:
        if chunk.source_collection != "3gpp_markdown":
            continue
        searchable = _fold_protocol_text(searchable_text(chunk))
        if not any(protocol_identifier_in_text(anchor, searchable)
                   for anchor in anchors):
            continue
        scored.append((chunk.chunk_id, _exact_match_priority(chunk, probe)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]

def _add_exact_results(
    retriever: HybridMarkdownRetriever,
    probe: RetrievalProbe,
    candidates: dict[str, _Candidate],
    exact: list[tuple[str, float]],
    *,
    hop: Literal[1, 2],
) -> None:
    route_weights = {route.specification: route.weight for route in probe.routes}
    for rank, (chunk_id, _) in enumerate(exact, 1):
        chunk = retriever.chunk_by_id[chunk_id]
        candidate = candidates.setdefault(chunk_id, _Candidate(chunk=chunk))
        candidate.retrieval_hop = min(candidate.retrieval_hop, hop)
        candidate.probe_ids.add(probe.probe_id)
        candidate.evidence_roles.add(probe.evidence_role)
        candidate.exact_anchor_rank = min(candidate.exact_anchor_rank or rank, rank)
        candidate.exact_probe_ranks[probe.probe_id] = min(
            candidate.exact_probe_ranks.get(probe.probe_id, rank),
            rank,
        )
        # Treat exact occurrence as an independent recall channel in RRF.
        candidate.rrf_score += 1.5 * probe.weight / (retriever.cfg.rrf_k + rank)
        candidate.route_score = max(
            candidate.route_score,
            0.004 * route_weights.get(chunk.specification or "", 0.5),
        )
        searchable = _fold_protocol_text(searchable_text(chunk))
        heading = _fold_protocol_text(" ".join(chunk.heading_path))
        for anchor in probe.anchors:
            if protocol_identifier_in_text(anchor, searchable):
                candidate.anchors.add(anchor)
            if protocol_identifier_in_text(anchor, heading):
                candidate.heading_anchors.add(anchor)


def _add_search_results(
    retriever: HybridMarkdownRetriever,
    probe: RetrievalProbe,
    candidates: dict[str, _Candidate],
    *,
    dense: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    hop: Literal[1, 2],
    reference_source: str | None = None,
) -> None:
    route_weights = {route.specification: route.weight for route in probe.routes}
    dense_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense, 1)}
    lexical_ranks = {
        chunk_id: rank for rank, (chunk_id, _) in enumerate(lexical, 1)
    }
    for chunk_id in set(dense_ranks) | set(lexical_ranks):
        chunk = retriever.chunk_by_id[chunk_id]
        candidate = candidates.setdefault(chunk_id, _Candidate(chunk=chunk))
        candidate.retrieval_hop = min(candidate.retrieval_hop, hop)
        candidate.probe_ids.add(probe.probe_id)
        candidate.evidence_roles.add(probe.evidence_role)
        if reference_source is not None:
            candidate.reference_source_chunk_ids.add(reference_source)
        if chunk_id in dense_ranks:
            rank = dense_ranks[chunk_id]
            candidate.rrf_score += probe.weight / (retriever.cfg.rrf_k + rank)
            candidate.semantic_rank = min(candidate.semantic_rank or rank, rank)
        if chunk_id in lexical_ranks:
            rank = lexical_ranks[chunk_id]
            candidate.rrf_score += probe.weight / (retriever.cfg.rrf_k + rank)
            candidate.lexical_rank = min(candidate.lexical_rank or rank, rank)
            candidate.lexical_probe_ranks[probe.probe_id] = min(
                candidate.lexical_probe_ranks.get(probe.probe_id, rank),
                rank,
            )
        candidate.route_score = max(
            candidate.route_score,
            0.004 * route_weights.get(chunk.specification or "", 0.5),
        )
        searchable = searchable_text(chunk).casefold()
        heading = " ".join(chunk.heading_path).casefold()
        for anchor in probe.anchors:
            if protocol_identifier_in_text(anchor, searchable):
                candidate.anchors.add(anchor)
            if protocol_identifier_in_text(anchor, heading):
                candidate.heading_anchors.add(anchor)


def _references(chunk: MarkdownChunk) -> tuple[tuple[str, str | None, str], ...]:
    candidates: list[tuple[int, int, str, str | None, str]] = []
    references: list[tuple[str, str | None, str]] = []
    seen: set[tuple[str, str | None]] = set()
    searchable = searchable_text(chunk)
    for pattern in _SPEC_REFERENCE_PATTERNS:
        for match in pattern.finditer(searchable):
            candidates.append((
                match.start(),
                match.end(),
                match.group("spec"),
                match.groupdict().get("clause"),
                match.group("citation"),
            ))
    for start, end, specification, clause, citation in sorted(candidates):
        # A clause-qualified pattern and the plain TS pattern can match the
        # same citation. Keep the more specific form, without suppressing
        # distinct references to the same specification elsewhere.
        if clause is None and any(
            other_specification == specification
            and other_clause is not None
            and other_start <= start
            and end <= other_end
            for (
                other_start,
                other_end,
                other_specification,
                other_clause,
                _,
            ) in candidates
        ):
            continue
        key = (specification, clause)
        if key in seen:
            continue
        seen.add(key)
        references.append((specification, clause, citation))
    return tuple(references)


def _execute_task(
    retriever: HybridMarkdownRetriever,
    item: PlannedDiscoveryRetrieval,
    *,
    final_hits: int,
    retrieval_stage: Literal[1, 2] = 1,
    reserve_probe_candidates: bool = False,
    ranking_mode: Literal[
        "full",
        "rrf",
        "rrf_anchor",
        "rrf_heading",
        "rrf_noise",
        "rrf_anchor_heading",
        "rrf_anchor_single",
        "rrf_anchor_full_coverage",
        "rrf_anchor_single_full_coverage",
        "rrf_anchor_single_if_full",
    ] = "full",
    anchor_weight: float = 0.004,
    single_channel_max_rank: int = 3,
    single_channel_weight: float = 1.0,
    full_coverage_bonus: float = 0.004,
    reranker: CrossEncoderReranker | None = None,
) -> PlannedRetrievalTaskResult:
    available_specs = {
        document.specification
        for document in retriever.documents
        if document.specification is not None
    }
    candidates: dict[str, _Candidate] = {}
    errors: list[str] = []
    unavailable_specs: set[str] = set()
    routed_specifications: set[str] = set()
    seen_semantic_queries: set[tuple[str, tuple[str, ...]]] = set()

    for probe in item.retrieval_probes:
        configured_specs = tuple(route.specification for route in probe.routes)
        routed_specifications.update(configured_specs)
        allowed_specs = tuple(
            spec for spec in configured_specs if spec in available_specs
        )
        unavailable_specs.update(set(configured_specs) - set(allowed_specs))
        if configured_specs and not allowed_specs:
            errors.append(
                f"{probe.probe_id}: no routed specification is present "
                "in the selected corpus"
            )
            continue
        # Route B already supplies hypothesis-specific lexical and semantic
        # queries.  Its anchors are speculative search hints and can be broad
        # enough to swamp the intended target specification.  More importantly,
        # exact candidates receive per-probe priority below, which can displace
        # a much stronger BM25/semantic hit.  Keep exact-anchor recall on Route A
        # only; Route B is ranked solely by lexical/semantic retrieval.
        hypothesis_probe = any(
            route.source == "agent_hypothesis" for route in probe.routes
        )
        if not hypothesis_probe:
            exact = _exact_anchor_results(
                retriever,
                probe,
                limit=max(
                    40,
                    retriever.cfg.lexical_candidates_per_collection * 2,
                ),
            )
            _add_exact_results(
                retriever,
                probe,
                candidates,
                exact,
                hop=retrieval_stage,
            )
        try:
            semantic_key = (
                _fold_protocol_text(probe.semantic_query),
                tuple(sorted(allowed_specs)),
            )
            if semantic_key in seen_semantic_queries:
                # A hypothesis commonly has several lexical probes but one
                # shared semantic query. Counting the same dense result once
                # per lexical probe artificially multiplies its RRF score and
                # can displace the lexical top result for the other probes.
                dense = []
            else:
                seen_semantic_queries.add(semantic_key)
                dense = retriever._dense(
                    probe.semantic_query,
                    "3gpp_markdown",
                    allowed_specs,
                )
        except Exception as exc:
            dense = []
            errors.append(
                f"{probe.probe_id}: semantic: {type(exc).__name__}: {exc}"
            )
        try:
            lexical = retriever.bm25.search(
                probe.lexical_query,
                source_collection="3gpp_markdown",
                limit=retriever.cfg.lexical_candidates_per_collection,
                allowed_specifications=(
                    frozenset(allowed_specs) if allowed_specs else None
                ),
            )
        except Exception as exc:
            lexical = []
            errors.append(
                f"{probe.probe_id}: lexical: {type(exc).__name__}: {exc}"
            )
        _add_search_results(
            retriever,
            probe,
            candidates,
            dense=dense,
            lexical=lexical,
            hop=retrieval_stage,
        )
    followed: list[str] = []
    if (
        retrieval_stage == 1
        and item.reference_expansion.enabled
        and item.reference_expansion.max_references
    ):
        # References are evidence-driven expansion, not another free router.
        # Inspect only strong first-hop candidates that matched an extractive
        # anchor, and do not re-query specifications already in the route set.
        reference_sources = sorted(
            (
                candidate
                for candidate in candidates.values()
                if candidate.retrieval_hop == 1 and candidate.anchors
            ),
            key=lambda candidate: (
                -len(candidate.anchors),
                -(candidate.rrf_score + candidate.route_score),
                candidate.chunk.chunk_id,
            ),
        )[:max(5, item.reference_expansion.max_references * 2)]
        for source_candidate in reference_sources:
            if len(followed) >= item.reference_expansion.max_references:
                break
            source_chunk = source_candidate.chunk
            source_chunk_id = source_chunk.chunk_id
            for specification, clause, citation in _references(source_chunk):
                if len(followed) >= item.reference_expansion.max_references:
                    break
                if (
                    specification == source_chunk.specification
                    or specification in routed_specifications
                    or specification not in available_specs
                ):
                    continue
                reference_key = f"{specification}:{clause or '*'}"
                if reference_key in followed:
                    continue
                followed.append(reference_key)
                for probe in item.retrieval_probes:
                    reference_route = SpecificationRoute(
                        specification=specification,
                        weight=0.75,
                        reason=f"explicit first-hop citation: {citation}",
                        source="reference",
                    )
                    reference_probe = probe.model_copy(update={
                        "routes": (reference_route,),
                        "lexical_query": " ".join((probe.lexical_query, citation)),
                        "semantic_query": " ".join((probe.semantic_query, citation)),
                        "weight": probe.weight * 0.75,
                    })
                    try:
                        dense = retriever._dense(
                            reference_probe.semantic_query,
                            "3gpp_markdown",
                            (specification,),
                        )
                    except Exception as exc:
                        dense = []
                        errors.append(
                            f"{probe.probe_id}: reference semantic: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    try:
                        lexical = retriever.bm25.search(
                            reference_probe.lexical_query,
                            source_collection="3gpp_markdown",
                            limit=retriever.cfg.lexical_candidates_per_collection,
                            allowed_specifications=frozenset({specification}),
                        )
                    except Exception as exc:
                        lexical = []
                        errors.append(
                            f"{probe.probe_id}: reference lexical: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    _add_search_results(
                        retriever,
                        reference_probe,
                        candidates,
                        dense=dense,
                        lexical=lexical,
                        hop=2,
                        reference_source=source_chunk_id,
                    )

    def noise_penalty(candidate: _Candidate) -> float:
        headings = " ".join(candidate.chunk.heading_path).casefold()
        penalty = 0.0
        if _is_heading_only(candidate.chunk):
            penalty += 0.020
        if any(value in headings for value in _LOW_VALUE_HEADINGS):
            penalty += 0.020
        if candidate.retrieval_hop == 2:
            penalty += 0.002
        return penalty

    def score(candidate: _Candidate) -> tuple[float, float, str]:
        anchor_score = min(0.016, anchor_weight * len(candidate.anchors))
        heading_score = min(0.015, 0.005 * len(candidate.heading_anchors))
        agreement_score = (
            0.004
            if candidate.semantic_rank is not None
            and candidate.lexical_rank is not None
            else 0.0
        )
        channel_ranks = tuple(
            rank
            for rank in (candidate.semantic_rank, candidate.lexical_rank)
            if rank is not None
        )
        single_channel_bonus = (
            single_channel_weight
            / (retriever.cfg.rrf_k + channel_ranks[0])
            if len(channel_ranks) == 1
            and channel_ranks[0] <= single_channel_max_rank
            else 0.0
        )
        full_anchor_coverage = any(
            probe.probe_id in candidate.probe_ids
            and set(probe.anchors).issubset(candidate.anchors)
            for probe in item.retrieval_probes
        )
        coverage_score = full_coverage_bonus if full_anchor_coverage else 0.0
        if ranking_mode == "rrf":
            final = candidate.rrf_score
        elif ranking_mode == "rrf_anchor":
            final = candidate.rrf_score + anchor_score
        elif ranking_mode == "rrf_heading":
            final = candidate.rrf_score + heading_score
        elif ranking_mode == "rrf_noise":
            final = candidate.rrf_score - noise_penalty(candidate)
        elif ranking_mode == "rrf_anchor_heading":
            final = candidate.rrf_score + anchor_score + heading_score
        elif ranking_mode == "rrf_anchor_single":
            final = candidate.rrf_score + anchor_score + single_channel_bonus
        elif ranking_mode == "rrf_anchor_full_coverage":
            final = candidate.rrf_score + anchor_score + coverage_score
        elif ranking_mode == "rrf_anchor_single_full_coverage":
            final = (
                candidate.rrf_score
                + anchor_score
                + single_channel_bonus
                + coverage_score
            )
        elif ranking_mode == "rrf_anchor_single_if_full":
            final = (
                candidate.rrf_score
                + anchor_score
                + (single_channel_bonus if full_anchor_coverage else 0.0)
            )
        else:
            final = (
                candidate.rrf_score
                + candidate.route_score
                + anchor_score
                + agreement_score
                - noise_penalty(candidate)
            )
        return final, candidate.rrf_score, candidate.chunk.chunk_id

    deterministic_scores = {
        candidate.chunk.chunk_id: score(candidate)[0]
        for candidate in candidates.values()
    }
    reranker_scores: dict[str, float] = {}
    deterministic_normalized: dict[str, float] = {}
    reranker_normalized: dict[str, float] = {}
    reranker_status: Literal["disabled", "applied", "fallback"] = "disabled"
    reranker_error: str | None = None

    def normalized(values: list[float]) -> list[float]:
        if not values:
            return []
        low = min(values)
        high = max(values)
        if math.isclose(low, high):
            return [0.0 for _ in values]
        return [(value - low) / (high - low) for value in values]

    if reranker is not None and candidates:
        rerank_candidates = sorted(
            candidates.values(),
            key=lambda candidate: candidate.chunk.chunk_id,
        )
        query = "\n".join(dict.fromkeys(
            probe.semantic_query for probe in item.retrieval_probes
        ))
        passages = [
            "\n\n".join(filter(None, (
                " > ".join(candidate.chunk.heading_path),
                candidate.chunk.text,
            )))
            for candidate in rerank_candidates
        ]
        try:
            raw_scores = reranker.predict(query, passages)
            if len(raw_scores) != len(rerank_candidates):
                raise RuntimeError(
                    "reranker returned a different number of scores than passages"
                )
            deterministic_values = [
                deterministic_scores[candidate.chunk.chunk_id]
                for candidate in rerank_candidates
            ]
            deterministic_values_normalized = normalized(deterministic_values)
            reranker_values_normalized = normalized(list(raw_scores))
            for candidate, raw, deterministic_norm, reranker_norm in zip(
                rerank_candidates,
                raw_scores,
                deterministic_values_normalized,
                reranker_values_normalized,
            ):
                chunk_id = candidate.chunk.chunk_id
                reranker_scores[chunk_id] = raw
                deterministic_normalized[chunk_id] = deterministic_norm
                reranker_normalized[chunk_id] = reranker_norm
            reranker_status = "applied"
        except Exception as exc:  # Deterministic ranking remains authoritative.
            reranker_status = "fallback"
            reranker_error = f"{type(exc).__name__}: {exc}"

    def ranking_score(candidate: _Candidate) -> float:
        chunk_id = candidate.chunk.chunk_id
        if reranker_status != "applied":
            return deterministic_scores[chunk_id]
        return (
            reranker_normalized[chunk_id]
            + reranker.deterministic_weight * deterministic_normalized[chunk_id]
        )

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (
            -ranking_score(candidate),
            -score(candidate)[1],
            score(candidate)[2],
        ),
    )
    selected: list[_Candidate] = []
    selected_ids: set[str] = set()
    if reserve_probe_candidates:
        for probe in item.retrieval_probes:
            exact_for_probe = sorted(
                (
                    value
                    for value in candidates.values()
                    if probe.probe_id in value.exact_probe_ranks
                    and value.chunk.chunk_id not in selected_ids
                ),
                key=lambda value: (
                    value.exact_probe_ranks[probe.probe_id],
                    value.chunk.chunk_id,
                ),
            )
            lexical_for_probe = sorted(
                (
                    value
                    for value in candidates.values()
                    if probe.probe_id in value.lexical_probe_ranks
                    and value.chunk.chunk_id not in selected_ids
                ),
                key=lambda value: (
                    value.lexical_probe_ranks[probe.probe_id],
                    -score(value)[0],
                    value.chunk.chunk_id,
                ),
            )
            candidate = (
                exact_for_probe[0]
                if exact_for_probe
                else lexical_for_probe[0]
                if lexical_for_probe
                else next(
                    (
                        value
                        for value in ordered
                        if probe.probe_id in value.probe_ids
                        and value.chunk.chunk_id not in selected_ids
                    ),
                    None,
                )
            )
            if candidate is not None:
                selected.append(candidate)
                selected_ids.add(candidate.chunk.chunk_id)
    for candidate in ordered:
        if len(selected) >= final_hits:
            break
        if candidate.chunk.chunk_id in selected_ids:
            continue
        # Do not discard a candidate solely because two chunks from the same
        # section were already selected. Long normative clauses are commonly
        # split across several chunks, and the third chunk can contain the
        # decisive table row or procedure step. Section diversity belongs in
        # the final compact-evidence ranking as a soft preference, not here as
        # an irreversible recall filter.
        selected.append(candidate)
        selected_ids.add(candidate.chunk.chunk_id)
    selected.sort(
        key=lambda candidate: (
            -ranking_score(candidate),
            candidate.chunk.chunk_id,
        )
    )

    hits: list[PlannedRetrievalHit] = []
    for rank, candidate in enumerate(selected[:final_hits], 1):
        chunk = candidate.chunk
        anchor_score = min(0.016, anchor_weight * len(candidate.anchors))
        heading_score = min(0.015, 0.005 * len(candidate.heading_anchors))
        agreement_score = (
            0.004
            if candidate.semantic_rank is not None
            and candidate.lexical_rank is not None
            else 0.0
        )
        penalty = noise_penalty(candidate)
        deterministic_score = deterministic_scores[candidate.chunk.chunk_id]
        final_score = ranking_score(candidate)
        source_hit_ids = set(candidate.reference_source_chunk_ids)
        for probe in item.retrieval_probes:
            if probe.probe_id in candidate.probe_ids:
                source_hit_ids.update(probe.source_hit_ids)
        hits.append(PlannedRetrievalHit(
            hit_id=f"{item.retrieval_id}:3gpp_markdown:{rank}",
            chunk_id=chunk.chunk_id,
            collection_rank=rank,
            specification=chunk.specification or "",
            version=chunk.version or "",
            source_path=chunk.source_path,
            document_sha256=chunk.document_sha256,
            heading_path=chunk.heading_path,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            excerpt=chunk.text,
            evidence_roles=_ordered_unique(candidate.evidence_roles),
            probe_ids=_ordered_unique(candidate.probe_ids),
            matched_anchors=_ordered_unique(candidate.anchors),
            retrieval_hop=candidate.retrieval_hop,
            reference_source_chunk_ids=_ordered_unique(source_hit_ids),
            semantic_rank=candidate.semantic_rank,
            lexical_rank=candidate.lexical_rank,
            rrf_score=candidate.rrf_score,
            route_score=candidate.route_score,
            anchor_score=anchor_score,
            heading_score=heading_score,
            agreement_score=agreement_score,
            noise_penalty=penalty,
            deterministic_score=deterministic_score,
            reranker_score=reranker_scores.get(candidate.chunk.chunk_id),
            deterministic_score_normalized=deterministic_normalized.get(
                candidate.chunk.chunk_id
            ),
            reranker_score_normalized=reranker_normalized.get(
                candidate.chunk.chunk_id
            ),
            reranker_model=(
                reranker.model_name if reranker_status == "applied" else None
            ),
            final_score=final_score,
        ))

    if hits and errors:
        status: Literal["completed", "partial", "no_hits", "unavailable"] = (
            "partial"
        )
    elif hits:
        status = "completed"
    elif errors:
        status = "unavailable"
    else:
        status = "no_hits"
    return PlannedRetrievalTaskResult(
        retrieval_id=item.retrieval_id,
        query_id=item.query_id,
        planner_action=item.planner_action,
        status=status,
        unavailable_specifications=tuple(sorted(unavailable_specs)),
        followed_references=tuple(followed),
        reranker_status=reranker_status,
        reranker_error=reranker_error,
        hits=tuple(hits),
        errors=tuple(errors),
    )


_INTENT_QUERY_PHRASES: dict[str, str] = {
    "definition": "definition and normative meaning",
    "representation": "normative representation",
    "field_structure": "fields and information elements",
    "encoding": "encoding and value construction",
    "carrier_usage": "messages or containers that carry the term",
    "procedure_usage": "procedure usage and message occurrence",
    "transport_path": "transport path and encapsulation",
    "endpoint_interface": "endpoints and interfaces",
    "direction": "protocol direction",
    "authority": "authorized entity or role",
    "security_protection": "security protection conditions",
    "state_condition": "protocol state and conditions",
    "timing_order": "timing and ordering",
    "prerequisites": "required inputs and prerequisites",
    "relationship": "normative relationship and mapping",
    "resource_constraint": "limits and resource constraints",
    "explicit_reference": "referenced normative clause",
}


_RELATION_MESSAGE = re.compile(
    r"\b(?:[A-Z][A-Z0-9_/-]*\s+){1,6}"
    r"(?:REQUEST|RESPONSE|REJECT|ACCEPT|COMMAND|COMPLETE|INDICATION|"
    r"NOTIFICATION)\b"
)
_RELATION_IE = re.compile(
    r"\b(?:[A-Z0-9][A-Za-z0-9'()/+\-]*\s+){1,6}"
    r"(?:IE|information element)\b"
)
_TABLE_REFERENCE_NAME = re.compile(
    r"^(?P<name>.+?)(?=\d+(?:\.\d+)+[A-Za-z]?$)"
)
_RELATION_TERM_STOPWORDS = _GENERIC_FACET_TERMS | frozenset({
    "format",
    "information element",
    "information element type/reference",
    "message type",
    "presence",
    "spare half octet",
    "type/reference",
})


def _useful_relation_term(surface: str) -> bool:
    folded = surface.strip(" .,:;()[]").casefold()
    if folded in _RELATION_TERM_STOPWORDS:
        return False
    if len(folded) < 3 or len(folded) > 100:
        return False
    if re.match(r"^\d+\s", folded):
        return False
    if re.match(r"^\d+\)", folded) or re.match(r"^\d{2,}[a-z]", folded):
        return False
    if folded.startswith(("this ", "the ", "a ", "an ")):
        return False
    if folded.endswith("message identity") or "conditional ie" in folded:
        return False
    return bool(re.search(r"[a-z]", folded, re.I))


def _grounded_terms_from_chunk(
    chunk: MarkdownChunk,
    focus_terms: tuple[str, ...],
) -> tuple[str, ...]:
    """Extract message/IE/table bridge terms without adding protocol facts."""

    text = chunk.text
    folded_focus = tuple(_fold_protocol_text(term) for term in focus_terms)
    candidates: list[tuple[int, str]] = []

    for match in _RELATION_MESSAGE.finditer(text):
        candidates.append((match.start(), match.group(0)))
    for match in _RELATION_IE.finditer(text):
        surface = match.group(0)
        candidates.append((match.start(), surface))
        if surface.casefold().endswith(" ie"):
            candidates.append((match.start(), surface[:-3]))

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        folded_line = _fold_protocol_text(stripped)
        if stripped.startswith("|") and any(
            term in folded_line for term in folded_focus
        ):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            for cell_index, cell in enumerate(cells):
                if not cell or cell in {"---", "M", "O", "C", "V", "TV", "TLV", "TLV-E", "LV", "LV-E"}:
                    continue
                surface = cell
                reference_match = _TABLE_REFERENCE_NAME.match(cell)
                if reference_match is not None:
                    surface = reference_match.group("name").strip()
                # The Information Element and Type/Reference columns carry
                # useful reverse-lookup terms. Other columns are identifiers,
                # presence flags, formats, or lengths.
                if cell_index not in {1, 2}:
                    continue
                position = text.find(surface, offset, offset + len(line))
                if position >= 0:
                    candidates.append((position, surface))
        offset += len(line)

    ordered: list[str] = []
    seen: set[str] = set()
    for _, surface in sorted(candidates, key=lambda item: (item[0], -len(item[1]))):
        cleaned = surface.strip(" .,:;()[]")
        key = cleaned.casefold()
        if key in seen or not _useful_relation_term(cleaned):
            continue
        if cleaned not in text:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return tuple(ordered[:24])


def _grounded_term_catalog(
    source_evidence_id: str,
    chunk: MarkdownChunk,
    focus_terms: tuple[str, ...],
) -> tuple[GroundedTerm, ...]:
    return tuple(
        GroundedTerm(
            term_id=f"{source_evidence_id}:T{index}",
            source_evidence_id=source_evidence_id,
            surface=surface,
        )
        for index, surface in enumerate(
            _grounded_terms_from_chunk(chunk, focus_terms),
            1,
        )
    )


def _relation_term_priority(term: str) -> tuple[int, int, str]:
    folded = term.casefold()
    score = 0
    if any(
        folded.endswith(suffix)
        for suffix in (
            " request", " response", " reject", " accept", " command",
            " complete", " indication", " notification",
        )
    ):
        score += 8
    if folded.endswith(" ie"):
        score += 6
    return (-score, len(term), folded)


def _build_grounded_relation_evidence(
    retriever: HybridMarkdownRetriever,
    resolved_plan: DiscoveryRetrievalPlanV3,
    *,
    max_per_question: int = 64,
    existing_chunk_ids_by_query: dict[str, frozenset[str]] | None = None,
) -> tuple[GroundedRelationEvidence, ...]:
    """Build a small corpus-grounded entity-to-message/IE relation frontier."""

    all_evidence: list[GroundedRelationEvidence] = []
    corpus_chunks = tuple(
        chunk
        for chunk in retriever.chunks
        if chunk.source_collection == "3gpp_markdown"
    )
    chunks_by_source: dict[str, list[MarkdownChunk]] = {}
    for chunk in corpus_chunks:
        chunks_by_source.setdefault(chunk.source_path, []).append(chunk)
    for chunks in chunks_by_source.values():
        chunks.sort(key=lambda value: (value.line_start, value.line_end, value.chunk_id))
    source_chunk_indexes = {
        (source_path, chunk.chunk_id): index
        for source_path, chunks in chunks_by_source.items()
        for index, chunk in enumerate(chunks)
    }

    for item in resolved_plan.retrievals:
        existing_chunk_ids = (
            existing_chunk_ids_by_query or {}
        ).get(item.query_id, frozenset())
        original_anchors = tuple(
            anchor.canonical
            for anchor in item.anchors
            if anchor.entity_type != "specification"
            and anchor.canonical.casefold() not in _GENERIC_FACET_TERMS
        )
        if not original_anchors:
            continue
        folded_anchors = tuple(original_anchors)
        route_weights: dict[str, float] = {}
        for probe in item.retrieval_probes:
            for route in probe.routes:
                route_weights[route.specification] = max(
                    route_weights.get(route.specification, 0.0),
                    route.weight,
                )
        routed = {
            specification
            for specification in route_weights
        }
        seed_scores: list[tuple[float, MarkdownChunk]] = []
        for chunk in corpus_chunks:
            searchable = _fold_protocol_text(searchable_text(chunk))
            if not any(protocol_identifier_in_text(anchor, searchable)
                       for anchor in folded_anchors):
                continue
            priority = max(
                _exact_match_priority(chunk, probe)
                for probe in item.retrieval_probes
            )
            priority += 5.0 * route_weights.get(chunk.specification or "", 0.0)
            seed_scores.append((priority, chunk))
        seed_scores.sort(key=lambda value: (-value[0], value[1].chunk_id))
        # Process every exact occurrence when discovering bridge terms. The
        # final relation frontier remains bounded, but a low-ranked exact
        # occurrence can still contribute a message/IE relation.
        seeds = [chunk for _, chunk in seed_scores]
        seed_priority = {chunk.chunk_id: score for score, chunk in seed_scores}

        selected: dict[str, MarkdownChunk] = {}
        kinds: dict[str, set[str]] = {}
        focus_by_chunk: dict[str, set[str]] = {}
        priority_by_chunk: dict[str, float] = {}

        def add(
            chunk: MarkdownChunk,
            kind: str,
            focus: tuple[str, ...],
            priority: float,
        ) -> None:
            selected.setdefault(chunk.chunk_id, chunk)
            kinds.setdefault(chunk.chunk_id, set()).add(kind)
            focus_by_chunk.setdefault(chunk.chunk_id, set()).update(focus)
            priority_by_chunk[chunk.chunk_id] = max(
                priority_by_chunk.get(chunk.chunk_id, 0.0),
                priority,
            )

        for chunk in seeds:
            add(
                chunk,
                "exact_entity_occurrence",
                original_anchors,
                seed_priority[chunk.chunk_id],
            )

        bridge_terms: list[str] = []
        seen_terms: set[str] = set()
        for chunk in seeds:
            for term in _grounded_terms_from_chunk(chunk, original_anchors):
                key = term.casefold()
                if key in seen_terms or key in {value.casefold() for value in original_anchors}:
                    continue
                seen_terms.add(key)
                bridge_terms.append(term)
        bridge_terms.sort(key=_relation_term_priority)

        # Reverse lookup discovers message tables and IE definitions through
        # terms that were observed next to the original entity in the corpus.
        active_bridge_terms = bridge_terms[:24]
        per_term_limit = max(
            3,
            min(12, max_per_question // max(1, len(active_bridge_terms))),
        )
        for term in active_bridge_terms:
            folded_term = _fold_protocol_text(term)
            matches: list[tuple[float, MarkdownChunk]] = []
            for chunk in corpus_chunks:
                searchable = _fold_protocol_text(searchable_text(chunk))
                if not protocol_identifier_in_text(folded_term, searchable):
                    continue
                score = float(min(3, searchable.count(folded_term)))
                if chunk.specification in routed:
                    score += 3.0
                if protocol_identifier_in_text(
                    folded_term,
                    _fold_protocol_text(" ".join(chunk.heading_path)),
                ):
                    score += 8.0
                if "message content" in searchable:
                    score += 5.0
                if "type/reference" in searchable:
                    score += 3.0
                if "purpose of the" in searchable:
                    score += 7.0
                if any(protocol_identifier_in_text(anchor, searchable)
                       for anchor in folded_anchors):
                    score += 4.0
                matches.append((score, chunk))
            matches.sort(key=lambda value: (-value[0], value[1].chunk_id))
            # Allocate the reverse-lookup budget uniformly from the available
            # question budget; no protocol term receives a special quota.
            for score, chunk in matches[:per_term_limit]:
                searchable = _fold_protocol_text(searchable_text(chunk))
                kind = (
                    "reverse_message_table"
                    if "message content" in searchable
                    else "reverse_grounded_term"
                )
                add(chunk, kind, (term,), score)

        # Preserve immediate source neighbours for split tables and adjacent
        # message definitions. This follows document layout, not model memory.
        frontier = list(selected.values())
        for chunk in frontier:
            siblings = chunks_by_source[chunk.source_path]
            index = source_chunk_indexes[(chunk.source_path, chunk.chunk_id)]
            for neighbor_index in (index - 1, index + 1):
                if 0 <= neighbor_index < len(siblings):
                    neighbor = siblings[neighbor_index]
                    if abs(neighbor.line_start - chunk.line_start) <= 120:
                        add(neighbor, "adjacent_source_chunk", (), 0.0)

        base_ordered = sorted(
            selected.values(),
            key=lambda chunk: (
                0 if chunk.chunk_id not in existing_chunk_ids else 1,
                -priority_by_chunk[chunk.chunk_id],
                0 if chunk.specification in routed else 1,
                chunk.source_path,
                chunk.line_start,
                chunk.chunk_id,
            ),
        )
        ordered_chunks: list[MarkdownChunk] = []
        ordered_ids: set[str] = set()

        def take_bucket(
            predicate: Any,
            limit: int,
        ) -> None:
            if limit <= 0:
                return
            added = 0
            for chunk in base_ordered:
                if chunk.chunk_id in ordered_ids or not predicate(chunk):
                    continue
                ordered_chunks.append(chunk)
                ordered_ids.add(chunk.chunk_id)
                added += 1
                if added >= limit:
                    return

        # Keep independent recall channels. Exact-anchor occurrences are seed
        # evidence; they must not consume the entire relation frontier and
        # hide message tables that omit the original surface form.
        message_table_budget = max(1, (max_per_question * 2) // 5)
        reverse_relation_budget = max(1, max_per_question // 5)
        exact_budget = max(
            1,
            max_per_question - message_table_budget - reverse_relation_budget,
        )
        take_bucket(
            lambda chunk: "reverse_message_table" in kinds[chunk.chunk_id],
            message_table_budget,
        )
        take_bucket(
            lambda chunk: (
                "reverse_grounded_term" in kinds[chunk.chunk_id]
                and "reverse_message_table" not in kinds[chunk.chunk_id]
                and "exact_entity_occurrence" not in kinds[chunk.chunk_id]
            ),
            reverse_relation_budget,
        )
        take_bucket(
            lambda chunk: "exact_entity_occurrence" in kinds[chunk.chunk_id],
            exact_budget,
        )
        take_bucket(
            lambda chunk: kinds[chunk.chunk_id] != {"adjacent_source_chunk"},
            max_per_question - len(ordered_chunks),
        )
        take_bucket(
            lambda chunk: True,
            max_per_question - len(ordered_chunks),
        )
        for index, chunk in enumerate(ordered_chunks, 1):
            evidence_id = f"R-{item.query_id}:{index}"
            searchable = _fold_protocol_text(searchable_text(chunk))
            matched = tuple(
                anchor
                for anchor in original_anchors
                if protocol_identifier_in_text(anchor, searchable)
            )
            focus = tuple(sorted(focus_by_chunk[chunk.chunk_id], key=str.casefold))
            terms = _grounded_term_catalog(
                evidence_id,
                chunk,
                focus or original_anchors,
            )
            all_evidence.append(GroundedRelationEvidence(
                evidence_id=evidence_id,
                query_id=item.query_id,
                chunk_id=chunk.chunk_id,
                specification=chunk.specification or "",
                version=chunk.version or "",
                source_path=chunk.source_path,
                document_sha256=chunk.document_sha256,
                heading_path=chunk.heading_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                excerpt=chunk.text,
                matched_original_anchors=matched,
                bridge_terms=tuple(
                    term
                    for term in focus
                    if term.casefold()
                    not in {anchor.casefold() for anchor in original_anchors}
                ),
                grounded_terms=terms,
                relation_kinds=tuple(sorted(kinds[chunk.chunk_id])),
            ))
    return tuple(all_evidence)


@dataclass
class _CuratedCandidate:
    chunk: MarkdownChunk
    source_evidence_ids: set[str] = field(default_factory=set)
    candidate_sources: set[str] = field(default_factory=set)
    matched_anchors: set[str] = field(default_factory=set)
    grounded_terms: set[str] = field(default_factory=set)
    relation_kinds: set[str] = field(default_factory=set)
    first_stage_score: float = 0.0


_ANGLE_DIRECT_CUES: dict[str, re.Pattern[str]] = {
    "carrier": re.compile(
        r"\b(?:include(?:s|d)?|contain(?:s|ed)?|convey(?:s|ed)?|"
        r"carr(?:y|ies|ied)|indicat(?:e|es|ed)|provide(?:s|d)?|"
        r"encapsulat(?:e|es|ed))\b",
        re.I,
    ),
    "format": re.compile(
        r"\b(?:represent(?:s|ed)?|consist(?:s|ed)?|compris(?:e|es|ed)|"
        r"contain(?:s|ed)?|cod(?:e|ed|ing)|encod(?:e|ed|ing)|"
        r"format|field|octet|bits?)\b",
        re.I,
    ),
    "reachability": re.compile(
        r"\b(?:send|sent|transmit(?:s|ted)?|transport(?:s|ed)?|via|over|"
        r"between|uplink|downlink|interface|encapsulat(?:e|es|ed))\b",
        re.I,
    ),
    "authority": re.compile(
        r"\b(?:authori[sz](?:e|es|ed|ation)|allow(?:s|ed)?|permit(?:s|ted)?|"
        r"shall|may|role|entity)\b",
        re.I,
    ),
    "protection": re.compile(
        r"\b(?:protect(?:s|ed|ion)?|integrity|confidentiality|authenticat(?:e|ed|ion)|"
        r"cipher(?:ed|ing)?|security context|replay|freshness)\b",
        re.I,
    ),
    "state_timing": re.compile(
        r"\b(?:before|after|during|when|while|state|phase|timer|expire(?:s|d)?|"
        r"start(?:s|ed)?|stop(?:s|ped)?|condition)\b",
        re.I,
    ),
    "required_inputs": re.compile(
        r"\b(?:require(?:s|d)?|mandatory|conditional|input|prerequisite|"
        r"shall include|available)\b",
        re.I,
    ),
    "relation": re.compile(
        r"\b(?:map(?:s|ped|ping)?|relat(?:e|es|ed|ionship)|"
        r"associat(?:e|es|ed|ion)|link(?:s|ed)?)\b",
        re.I,
    ),
    "resource_constraints": re.compile(
        r"\b(?:maximum|minimum|at most|at least|times?|attempts?|repeat(?:s|ed)?|"
        r"rate|duration|limit(?:s|ed)?|successive)\b",
        re.I,
    ),
}


def _candidate_relation_class(
    chunk: MarkdownChunk,
    *,
    angle: str,
    original_anchors: tuple[str, ...],
    relation_kinds: frozenset[str],
    grounded_terms: tuple[str, ...] = (),
) -> tuple[EvidenceRelationClass, tuple[str, ...], float]:
    """Conservatively distinguish explicit relations from useful bridges."""

    searchable = searchable_text(chunk)
    folded_searchable = _fold_protocol_text(searchable)
    folded_anchors = tuple(_fold_protocol_text(value) for value in original_anchors)
    matched = tuple(value for value in folded_anchors if value in folded_searchable)
    anchor_rows = tuple(
        line
        for line in chunk.text.splitlines()
        if line.strip().startswith("|")
        and any(value in _fold_protocol_text(line) for value in folded_anchors)
    )
    structured_table_relation = bool(anchor_rows) and (
        "message content" in folded_searchable
        or "type/reference" in folded_searchable
        or "| iei |" in folded_searchable
    )
    cue = _ANGLE_DIRECT_CUES.get(angle)
    relation_blocks = tuple(
        block.strip()
        for block in re.split(r"(?<=[.!?])\s+|\n\s*\n", chunk.text)
        if block.strip()
    )
    normative_relation = bool(cue) and any(
        any(value in _fold_protocol_text(block) for value in folded_anchors)
        and cue.search(block) is not None
        for block in relation_blocks
    )

    reasons: list[str] = []
    score = 0.0
    if structured_table_relation and angle in {"carrier", "format"}:
        reasons.append("structured table row contains the original anchor")
        score += 60.0
    if normative_relation:
        reasons.append("same normative text block links the anchor to the requested angle")
        score += 50.0
    if reasons:
        if "exact_entity_occurrence" in relation_kinds:
            score += 15.0
        if "reverse_message_table" in relation_kinds:
            score += 10.0
        return "direct_relation", tuple(reasons), score

    heading = _fold_protocol_text(" ".join(chunk.heading_path))
    non_message_bridge_terms = tuple(
        term
        for term in grounded_terms
        if _RELATION_MESSAGE.fullmatch(term.strip()) is None
    )
    if "reverse_message_table" in relation_kinds and non_message_bridge_terms:
        reasons.append(
            "message table contains a non-message bridge term useful for an evidence chain"
        )
        score += 35.0
    if matched and (
        "information element" in folded_searchable
        or "coded as" in folded_searchable
        or "message content" in folded_searchable
    ):
        reasons.append("anchor occurs in a message/IE definition useful for an evidence chain")
        score += 30.0
    if "reverse_grounded_term" in relation_kinds:
        reasons.append("chunk was found by reversing a term extracted from anchor evidence")
        score += 15.0
    if matched and any(value in heading for value in folded_anchors):
        reasons.append("original anchor occurs in the section heading")
        score += 10.0
    if reasons:
        if relation_kinds == frozenset({"adjacent_source_chunk"}):
            return (
                "context_only",
                ("adjacent chunk has no independently established relation",),
                0.0,
            )
        return "supporting_relation", tuple(dict.fromkeys(reasons)), score

    return (
        "context_only",
        ("no explicit or chain-supporting relation to the original anchor was established",),
        0.0,
    )


def curate_discovery_evidence(
    resolved_plan: DiscoveryRetrievalPlanV3,
    first_stage_tasks: tuple[PlannedRetrievalTaskResult, ...],
    relation_evidence: tuple[GroundedRelationEvidence, ...],
    *,
    final_evidence_hits: int = 20,
) -> tuple[CuratedEvidenceTask, ...]:
    """Merge first-pass and relation candidates, then select compact evidence."""

    if final_evidence_hits < 1:
        raise ValueError("final_evidence_hits must be positive")
    first_by_query = {task.query_id: task for task in first_stage_tasks}
    curated_tasks: list[CuratedEvidenceTask] = []
    for item in resolved_plan.retrievals:
        original_anchors = tuple(
            anchor.canonical
            for anchor in item.anchors
            if anchor.entity_type != "specification"
            and anchor.canonical.casefold() not in _GENERIC_FACET_TERMS
        )
        candidates: dict[str, _CuratedCandidate] = {}

        def candidate_for(chunk: MarkdownChunk) -> _CuratedCandidate:
            return candidates.setdefault(chunk.chunk_id, _CuratedCandidate(chunk=chunk))

        for hit in first_by_query[item.query_id].hits:
            candidate = candidate_for(_chunk_from_planned_hit(hit))
            candidate.source_evidence_ids.add(hit.hit_id)
            candidate.candidate_sources.add("first_stage")
            candidate.matched_anchors.update(hit.matched_anchors)
            candidate.grounded_terms.update(
                _grounded_terms_from_chunk(candidate.chunk, original_anchors)
            )
            candidate.first_stage_score = max(candidate.first_stage_score, hit.final_score)
        for evidence in relation_evidence:
            if evidence.query_id != item.query_id:
                continue
            candidate = candidate_for(MarkdownChunk(
                chunk_id=evidence.chunk_id,
                source_collection="3gpp_markdown",
                specification=evidence.specification,
                version=evidence.version,
                source_path=evidence.source_path,
                document_sha256=evidence.document_sha256,
                heading_path=evidence.heading_path,
                line_start=evidence.line_start,
                line_end=evidence.line_end,
                text=evidence.excerpt,
            ))
            candidate.source_evidence_ids.add(evidence.evidence_id)
            candidate.candidate_sources.update(evidence.relation_kinds)
            candidate.relation_kinds.update(evidence.relation_kinds)
            candidate.matched_anchors.update(evidence.matched_original_anchors)
            candidate.grounded_terms.update(
                term.surface for term in evidence.grounded_terms
            )

        classified: list[CuratedEvidenceItem] = []
        for candidate in candidates.values():
            relation_class, reasons, score = _candidate_relation_class(
                candidate.chunk,
                angle=item.angle,
                original_anchors=original_anchors,
                relation_kinds=frozenset(candidate.relation_kinds),
                grounded_terms=_ordered_unique(candidate.grounded_terms),
            )
            if "first_stage" in candidate.candidate_sources:
                score += min(5.0, max(0.0, candidate.first_stage_score) * 20.0)
            classified.append(CuratedEvidenceItem(
                evidence_id="pending",
                query_id=item.query_id,
                chunk_id=candidate.chunk.chunk_id,
                specification=candidate.chunk.specification or "",
                version=candidate.chunk.version or "",
                source_path=candidate.chunk.source_path,
                document_sha256=candidate.chunk.document_sha256,
                heading_path=candidate.chunk.heading_path,
                line_start=candidate.chunk.line_start,
                line_end=candidate.chunk.line_end,
                excerpt=candidate.chunk.text,
                relation_class=relation_class,
                classification_reasons=reasons,
                candidate_sources=_ordered_unique(candidate.candidate_sources),
                source_evidence_ids=_ordered_unique(candidate.source_evidence_ids),
                matched_original_anchors=_ordered_unique(candidate.matched_anchors),
                grounded_terms=_ordered_unique(candidate.grounded_terms),
                curation_score=score,
            ))

        class_order = {
            "direct_relation": 0,
            "supporting_relation": 1,
            "context_only": 2,
        }
        classified.sort(key=lambda evidence: (
            class_order[evidence.relation_class],
            -evidence.curation_score,
            evidence.specification,
            evidence.line_start,
            evidence.chunk_id,
        ))
        selected = tuple(
            evidence.model_copy(update={
                "evidence_id": f"E-{item.query_id}:{rank}",
            })
            for rank, evidence in enumerate(
                (
                    value
                    for value in classified
                    if value.relation_class != "context_only"
                ),
                1,
            )
            if rank <= final_evidence_hits
        )
        curated_tasks.append(CuratedEvidenceTask(
            query_id=item.query_id,
            claim_id=item.claim_id,
            angle=item.angle,
            original_question=item.original_question,
            candidate_count=len(classified),
            direct_candidate_count=sum(
                value.relation_class == "direct_relation" for value in classified
            ),
            supporting_candidate_count=sum(
                value.relation_class == "supporting_relation" for value in classified
            ),
            context_candidate_count=sum(
                value.relation_class == "context_only" for value in classified
            ),
            selected_evidence=selected,
        ))
    return tuple(curated_tasks)


def _chunk_from_planned_hit(hit: PlannedRetrievalHit) -> MarkdownChunk:
    return MarkdownChunk(
        chunk_id=hit.chunk_id,
        source_collection="3gpp_markdown",
        specification=hit.specification,
        version=hit.version,
        source_path=hit.source_path,
        document_sha256=hit.document_sha256,
        heading_path=hit.heading_path,
        line_start=hit.line_start,
        line_end=hit.line_end,
        text=hit.excerpt,
    )


def _first_stage_agent_payload(
    resolved_plan: DiscoveryRetrievalPlanV3,
    tasks: tuple[PlannedRetrievalTaskResult, ...],
    relation_evidence: tuple[GroundedRelationEvidence, ...],
    *,
    query_ids: frozenset[str] | None = None,
) -> dict[str, Any]:
    task_by_query = {task.query_id: task for task in tasks}
    return {
        "questions": [
            {
                "query_id": item.query_id,
                "angle": item.angle,
                "original_question": item.original_question,
                "first_stage_routes": sorted({
                    route.specification
                    for probe in item.retrieval_probes
                    for route in probe.routes
                }),
                "first_stage_hits": [
                    {
                        "hit_id": hit.hit_id,
                        "specification": hit.specification,
                        "heading_path": list(hit.heading_path),
                        "grounded_terms": [
                            term.model_dump(mode="json")
                            for term in _grounded_term_catalog(
                                hit.hit_id,
                                _chunk_from_planned_hit(hit),
                                tuple(
                                    anchor.canonical
                                    for anchor in item.anchors
                                    if anchor.entity_type != "specification"
                                ),
                            )
                        ],
                        "excerpt": hit.excerpt,
                    }
                    for hit in task_by_query[item.query_id].hits
                ],
                "grounded_relation_evidence": [
                    {
                        "evidence_id": evidence.evidence_id,
                        "specification": evidence.specification,
                        "heading_path": list(evidence.heading_path),
                        "matched_original_anchors": list(
                            evidence.matched_original_anchors
                        ),
                        "grounded_terms": [
                            term.model_dump(mode="json")
                            for term in evidence.grounded_terms
                        ],
                        "relation_kinds": list(evidence.relation_kinds),
                        "excerpt": evidence.excerpt,
                    }
                    for evidence in relation_evidence
                    if evidence.query_id == item.query_id
                ],
            }
            for item in resolved_plan.retrievals
            if query_ids is None or item.query_id in query_ids
        ],
        "allowed_intents": list(_INTENT_QUERY_PHRASES),
    }


def _grounded_followup_plan(
    *,
    resolved_plan: DiscoveryRetrievalPlanV3,
    first_stage_tasks: tuple[PlannedRetrievalTaskResult, ...],
    expansion_agent: CodexStructuredAgent,
    available_specifications: frozenset[str],
    relation_evidence: tuple[GroundedRelationEvidence, ...],
) -> EvidenceGroundedExpansionPlan:
    instructions = (
        "You plan a second retrieval pass for protocol specifications. "
        "For every input question, return its query_id exactly once and "
        "preserve its frozen angle. Select useful bridge terms only by their "
        "provided term_id; never copy, rewrite, or invent a term surface. "
        "Every selected term ID must belong to one of the supplied parent "
        "evidence IDs for the same question. Prefer corpus-derived message, "
        "IE, table-reference, section, and explicit-reference bridges that "
        "help answer the requested facet. Choose an allowed retrieval intent, "
        "but do not write a free-form query and do not answer the question. "
        "Do not introduce any message, information element, interface, "
        "state, specification, relationship, or protocol fact from memory. "
        "If the evidence has no query-relevant bridge term, return no "
        "probes for that question. Use at most five non-duplicate probes."
    )
    # Keep each Codex context bounded when one hypothesis has many Discovery
    # questions. Batches are deterministic and preserve original query order.
    payloads: list[dict[str, Any]] = []
    current_ids: list[str] = []
    current_size = 0
    for item in resolved_plan.retrievals:
        single = _first_stage_agent_payload(
            resolved_plan,
            first_stage_tasks,
            relation_evidence,
            query_ids=frozenset({item.query_id}),
        )
        size = len(str(single))
        if current_ids and (current_size + size > 80_000 or len(current_ids) >= 4):
            payloads.append(_first_stage_agent_payload(
                resolved_plan,
                first_stage_tasks,
                relation_evidence,
                query_ids=frozenset(current_ids),
            ))
            current_ids = []
            current_size = 0
        current_ids.append(item.query_id)
        current_size += size
    if current_ids:
        payloads.append(_first_stage_agent_payload(
            resolved_plan,
            first_stage_tasks,
            relation_evidence,
            query_ids=frozenset(current_ids),
        ))

    returned_questions: list[QuestionExpansionProposal] = []
    receipts: list[CodexAgentReceipt] = []
    for payload in payloads:
        response = expansion_agent.parse(
            purpose="plan evidence-grounded second-stage Discovery retrieval",
            prompt_version="discovery-evidence-expansion/v1.1",
            instructions=instructions,
            input_payload=payload,
            output_type=EvidenceExpansionBatch,
        )
        expected_ids = [item["query_id"] for item in payload["questions"]]
        returned_ids = [
            item.query_id for item in response.output.questions
        ]
        if returned_ids != expected_ids:
            raise ValueError(
                "Codex expansion must preserve ordered Discovery query IDs"
            )
        returned_questions.extend(response.output.questions)
        receipts.append(response.receipt)

    first_task_by_query = {
        task.query_id: task for task in first_stage_tasks
    }
    proposal_by_query = {
        item.query_id: item for item in returned_questions
    }
    followups: list[EvidenceGroundedFollowup] = []
    for item in resolved_plan.retrievals:
        first_task = first_task_by_query[item.query_id]
        source_by_id: dict[str, Any] = {
            hit.hit_id: hit for hit in first_task.hits
        }
        source_by_id.update({
            evidence.evidence_id: evidence
            for evidence in relation_evidence
            if evidence.query_id == item.query_id
        })
        original_surfaces = tuple(
            anchor.canonical
            for anchor in item.anchors
            if anchor.entity_type != "specification"
            and anchor.canonical.casefold() not in _GENERIC_FACET_TERMS
        )
        term_by_id: dict[str, GroundedTerm] = {}
        for hit in first_task.hits:
            for term in _grounded_term_catalog(
                hit.hit_id,
                _chunk_from_planned_hit(hit),
                original_surfaces,
            ):
                term_by_id[term.term_id] = term
        for evidence in relation_evidence:
            if evidence.query_id != item.query_id:
                continue
            for term in evidence.grounded_terms:
                term_by_id[term.term_id] = term
        proposal = proposal_by_query[item.query_id]
        seen_followup_ids: set[str] = set()
        warnings: list[str] = []
        probes: list[RetrievalProbe] = []
        for index, proposed in enumerate(proposal.probes, 1):
            if proposed.followup_id in seen_followup_ids:
                warnings.append(
                    f"skipped duplicate follow-up ID: {proposed.followup_id}"
                )
                continue
            seen_followup_ids.add(proposed.followup_id)
            unknown_parents = tuple(
                parent
                for parent in proposed.parent_evidence_ids
                if parent not in source_by_id
            )
            if unknown_parents:
                warnings.append(
                    f"skipped {proposed.followup_id}: unknown parent evidence "
                    + ", ".join(unknown_parents)
                )
                continue
            surfaces: list[str] = []
            source_ids: list[str] = list(proposed.parent_evidence_ids)
            grounded_terms: list[EvidenceTermProposal] = []
            for term_id in proposed.selected_term_ids:
                term = term_by_id.get(term_id)
                if term is None:
                    warnings.append(
                        f"ignored unknown term ID in {proposed.followup_id}: "
                        f"{term_id}"
                    )
                    continue
                if term.source_evidence_id not in proposed.parent_evidence_ids:
                    warnings.append(
                        f"ignored term outside declared parents in "
                        f"{proposed.followup_id}: {term_id}"
                    )
                    continue
                if term.surface.casefold() in _GENERIC_FACET_TERMS:
                    continue
                grounded_terms.append(EvidenceTermProposal(
                    source_hit_id=term.source_evidence_id,
                    surface=term.surface,
                ))
                surfaces.append(term.surface)
                source_ids.append(term.source_evidence_id)

            if not grounded_terms:
                warnings.append(
                    f"skipped {proposed.followup_id}: no valid selected term IDs"
                )
                continue

            routes: dict[str, SpecificationRoute] = {}
            for parent_id in proposed.parent_evidence_ids:
                parent = source_by_id[parent_id]
                routes[parent.specification] = SpecificationRoute(
                    specification=parent.specification,
                    weight=1.0,
                    reason="specification containing grounded first-stage evidence",
                    source="reference",
                )
                for specification, _, citation in _references(MarkdownChunk(
                    chunk_id=parent.chunk_id,
                    source_collection="3gpp_markdown",
                    specification=parent.specification,
                    version=parent.version,
                    source_path=parent.source_path,
                    document_sha256=parent.document_sha256,
                    heading_path=parent.heading_path,
                    line_start=parent.line_start,
                    line_end=parent.line_end,
                    text=parent.excerpt,
                )):
                    if specification in available_specifications:
                        routes.setdefault(
                            specification,
                            SpecificationRoute(
                                specification=specification,
                                weight=0.8,
                                reason=f"explicit citation in parent evidence: {citation}",
                                source="reference",
                            ),
                        )
            unique_surfaces_list: list[str] = []
            seen_surfaces: set[str] = set()
            # Every second-stage query retains the original concrete entity.
            # A grounded bridge may widen the relation being searched, but it
            # cannot replace the subject of the Discovery question.
            for surface in (*original_surfaces, *surfaces):
                key = surface.casefold()
                if key not in seen_surfaces:
                    seen_surfaces.add(key)
                    unique_surfaces_list.append(surface)
            unique_surfaces = tuple(unique_surfaces_list)
            intent_phrase = _INTENT_QUERY_PHRASES[proposed.intent]
            lexical_query = " ".join((*unique_surfaces, intent_phrase))
            semantic_query = (
                f"{item.original_question}\n"
                f"Evidence-grounded focus: {'; '.join(unique_surfaces)}. "
                f"Retrieval objective: {intent_phrase}."
            )
            probes.append(RetrievalProbe(
                probe_id=f"S2P{index}",
                evidence_role=item.angle,
                lexical_query=lexical_query,
                semantic_query=semantic_query,
                routes=tuple(routes.values()),
                anchors=unique_surfaces,
                facets=(proposed.intent,),
                source_hit_ids=tuple(dict.fromkeys(source_ids)),
                evidence_terms=tuple(grounded_terms),
                stage=2,
            ))
        followups.append(EvidenceGroundedFollowup(
            retrieval_id=f"S2-{item.query_id}",
            query_id=item.query_id,
            claim_id=item.claim_id,
            angle=item.angle,
            original_question=item.original_question,
            status="expanded" if probes else "no_expansion",
            warnings=(
                tuple(warnings)
                if probes
                else tuple((*warnings, "agent found no usable grounded expansion probe"))
            ),
            retrieval_probes=tuple(probes),
        ))
    return EvidenceGroundedExpansionPlan(
        expansion_agent_receipts=tuple(receipts),
        retrievals=tuple(followups),
    )


def _second_stage_item(
    first: PlannedDiscoveryRetrieval,
    followup: EvidenceGroundedFollowup,
) -> PlannedDiscoveryRetrieval:
    return PlannedDiscoveryRetrieval(
        retrieval_id=followup.retrieval_id,
        query_id=followup.query_id,
        claim_id=followup.claim_id,
        angle=followup.angle,
        original_question=followup.original_question,
        planner_action="planned",
        anchors=first.anchors,
        explicit_facets=first.explicit_facets,
        context_dependent=first.context_dependent,
        warnings=followup.warnings,
        routing_policy=first.routing_policy,
        reference_expansion=ReferenceExpansionPolicy(enabled=False),
        retrieval_probes=followup.retrieval_probes,
    )


def _merge_stage_tasks(
    first_tasks: tuple[PlannedRetrievalTaskResult, ...],
    second_tasks: tuple[PlannedRetrievalTaskResult, ...],
) -> tuple[PlannedRetrievalTaskResult, ...]:
    second_by_query = {task.query_id: task for task in second_tasks}
    merged: list[PlannedRetrievalTaskResult] = []
    for first in first_tasks:
        second = second_by_query[first.query_id]
        hits_by_chunk: dict[str, PlannedRetrievalHit] = {}
        order: list[str] = []
        for hit in (*first.hits, *second.hits):
            existing = hits_by_chunk.get(hit.chunk_id)
            if existing is None:
                hits_by_chunk[hit.chunk_id] = hit
                order.append(hit.chunk_id)
                continue
            hits_by_chunk[hit.chunk_id] = existing.model_copy(update={
                "probe_ids": tuple(dict.fromkeys(
                    (*existing.probe_ids, *hit.probe_ids)
                )),
                "matched_anchors": tuple(dict.fromkeys(
                    (*existing.matched_anchors, *hit.matched_anchors)
                )),
                "reference_source_chunk_ids": tuple(dict.fromkeys(
                    (
                        *existing.reference_source_chunk_ids,
                        *hit.reference_source_chunk_ids,
                    )
                )),
                "final_score": max(existing.final_score, hit.final_score),
            })
        ranked_hits = tuple(
            hits_by_chunk[chunk_id].model_copy(update={
                "hit_id": f"M-{first.query_id}:3gpp_markdown:{rank}",
                "collection_rank": rank,
            })
            for rank, chunk_id in enumerate(order, 1)
        )
        errors = tuple((*first.errors, *second.errors))
        if ranked_hits and errors:
            status: Literal["completed", "partial", "no_hits", "unavailable"] = (
                "partial"
            )
        elif ranked_hits:
            status = "completed"
        elif errors:
            status = "unavailable"
        else:
            status = "no_hits"
        merged.append(PlannedRetrievalTaskResult(
            retrieval_id=f"M-{first.query_id}",
            query_id=first.query_id,
            planner_action=first.planner_action,
            status=status,
            unavailable_specifications=tuple(sorted(set(
                (*first.unavailable_specifications, *second.unavailable_specifications)
            ))),
            followed_references=tuple(dict.fromkeys(
                (*first.followed_references, *second.followed_references)
            )),
            hits=ranked_hits,
            errors=errors,
        ))
    return tuple(merged)


def execute_expanded_discovery_retrieval(
    plan: DiscoveryRetrievalPlanV3,
    *,
    app_config: AppConfig | None = None,
    first_stage_hits: int = 12,
    second_stage_candidates_per_question: int = 48,
    retriever: HybridMarkdownRetriever | None = None,
) -> ExpandedRetrievalExecution:
    """Retrieve once, then add novel corpus-grounded relation candidates."""

    if first_stage_hits < 1:
        raise ValueError("first_stage_hits must be positive")
    if second_stage_candidates_per_question < 0:
        raise ValueError(
            "second_stage_candidates_per_question cannot be negative"
        )
    cfg = (app_config or load_config()).retrieval
    try:
        active_retriever = retriever or HybridMarkdownRetriever(cfg)
        resolved = resolve_discovery_retrieval_plan(plan, active_retriever)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        unavailable = PlannedRetrievalExecution(
            status="unavailable",
            embedding_model=cfg.embedding_model,
            corpus_version_policy=cfg.version_policy,
            corpus_release=cfg.target_release,
            errors=(error,),
        )
        return ExpandedRetrievalExecution(
            status="unavailable",
            first_stage=unavailable,
            errors=(error,),
        )

    first_items = tuple(
        item.model_copy(update={
            "reference_expansion": ReferenceExpansionPolicy(enabled=False),
        })
        for item in resolved.retrievals
    )
    resolved_first = resolved.model_copy(update={"retrievals": first_items})
    first_tasks = tuple(
        _execute_task(
            active_retriever,
            item,
            final_hits=first_stage_hits,
            retrieval_stage=1,
        )
        for item in first_items
    )
    first_errors = tuple(error for task in first_tasks for error in task.errors)
    first_status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if first_tasks and all(task.status == "unavailable" for task in first_tasks)
        else "partial"
        if any(task.status in {"partial", "unavailable"} for task in first_tasks)
        else "completed"
    )
    first_execution = PlannedRetrievalExecution(
        status=first_status,
        resolved_plan=resolved_first,
        index_fingerprint=active_retriever.fingerprint,
        index_manifest_path=active_retriever.index_manifest_path,
        embedding_model=cfg.embedding_model,
        corpus_version_policy=cfg.version_policy,
        corpus_release=cfg.target_release,
        tasks=first_tasks,
        errors=first_errors,
    )
    first_chunks_by_query = {
        task.query_id: frozenset(hit.chunk_id for hit in task.hits)
        for task in first_tasks
    }
    # Relation expansion is experimental and disabled when its budget is zero.
    # The default batch path is intentionally first-stage retrieval only.
    expanded = (
        _build_grounded_relation_evidence(
            active_retriever,
            resolved_first,
            max_per_question=second_stage_candidates_per_question * 2,
            existing_chunk_ids_by_query=first_chunks_by_query,
        )
        if second_stage_candidates_per_question
        else ()
    )
    # The second stage is an answer-expansion channel. Do not spend its budget
    # returning chunks that the first stage already supplied.
    second_stage: list[GroundedRelationEvidence] = []
    counts: dict[str, int] = {}
    for evidence in expanded:
        if evidence.chunk_id in first_chunks_by_query.get(
            evidence.query_id,
            frozenset(),
        ):
            continue
        count = counts.get(evidence.query_id, 0)
        if count >= second_stage_candidates_per_question:
            continue
        counts[evidence.query_id] = count + 1
        second_stage.append(evidence)
    status: Literal["completed", "partial", "unavailable"] = (
        "partial" if first_errors else "completed"
    )
    return ExpandedRetrievalExecution(
        status=status,
        resolved_plan=resolved_first,
        first_stage=first_execution,
        second_stage_candidates=tuple(second_stage),
        errors=first_errors,
    )


def _hit_hypothesis_ids(hit: PlannedRetrievalHit) -> tuple[str, ...]:
    """Recover hypothesis provenance retained by Route B merging."""

    identifiers: set[str] = set()
    for probe_id in hit.probe_ids:
        matched = re.match(
            r"^(H[1-9][0-9]*)-(?:R|P)[1-9][0-9]*$",
            probe_id,
        )
        if matched is not None:
            identifiers.add(matched.group(1))
    for source in hit.retrieval_sources:
        matched = re.search(r"(?:^|:)route_b:(H[1-9][0-9]*)$", source)
        if matched is not None:
            identifiers.add(matched.group(1))
    return tuple(sorted(identifiers, key=lambda value: int(value[1:])))


def _hit_requirement_keys(
    hit: Mapping[str, Any],
) -> frozenset[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for probe_id in hit.get("probe_ids", ()):
        matched = re.match(
            r"^(H[1-9][0-9]*)-([RP][1-9][0-9]*)$",
            str(probe_id),
        )
        if matched is not None:
            keys.add((matched.group(1), matched.group(2)))
    return frozenset(keys)


def _build_hypothesis_evidence_packets(
    hypotheses: tuple[Mapping[str, Any], ...],
    evidence: list[dict[str, Any]],
    repair_context: tuple[Mapping[str, Any], ...],
) -> list[dict[str, Any]]:
    """Group evidence IDs without duplicating their specification excerpts."""

    packets: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        hypothesis_id = str(hypothesis.get("hypothesis_id", "")).strip()
        if not hypothesis_id:
            continue
        requirements = [
            requirement
            for requirement in hypothesis.get("evidence_requirements", ())
            if isinstance(requirement, Mapping)
            and requirement.get("requirement_id")
        ]
        evidence_by_requirement: list[dict[str, Any]] = []
        for requirement in requirements:
            requirement_id = str(requirement["requirement_id"]).strip()
            key = (hypothesis_id, requirement_id)
            evidence_ids = [
                str(candidate["evidence_id"])
                for candidate in evidence
                if key in _hit_requirement_keys(candidate)
            ]
            evidence_by_requirement.append({
                "requirement_id": requirement_id,
                "relation_to_verify": requirement.get("relation_to_verify"),
                "specification": requirement.get("specification"),
                "evidence_ids": list(dict.fromkeys(evidence_ids)),
            })
        linked_repairs = [
            {
                "repair_id": repair.get("repair_id"),
                "requirement_id": repair.get("requirement_id"),
                "evidence_ids": list(repair.get("evidence_ids", ())),
            }
            for repair in repair_context
            if hypothesis_id in repair.get("hypothesis_ids", ())
        ]
        packets.append({
            "hypothesis_id": hypothesis_id,
            "evidence_by_requirement": evidence_by_requirement,
            "repair_evidence": linked_repairs,
        })
    return packets


def summarize_expanded_discovery_evidence(
    execution: ExpandedRetrievalExecution,
    *,
    summary_agent: CodexStructuredAgent,
    max_answers_per_question: int = 10,
    answer_hypotheses_by_query: Mapping[
        str,
        tuple[Mapping[str, Any], ...],
    ] | None = None,
    upstream_discovery_results: tuple[Mapping[str, Any], ...] = (),
    upstream_discovery_results_by_query: Mapping[
        str,
        tuple[Mapping[str, Any], ...],
    ] | None = None,
    query_ids: frozenset[str] | None = None,
    allowed_repair_specifications: frozenset[str] = frozenset(),
    allow_repair: bool = True,
    repair_context_by_query: Mapping[
        str,
        tuple[Mapping[str, Any], ...],
    ] | None = None,
    agent_locator_context_by_query: Mapping[
        str,
        tuple[Mapping[str, Any], ...],
    ] | None = None,
) -> EvidenceAnswerSummary:
    """Select source evidence and produce final semantic answers in one call."""

    if not 1 <= max_answers_per_question <= 10:
        raise ValueError("max_answers_per_question must be between 1 and 10")
    if execution.resolved_plan is None:
        return EvidenceAnswerSummary(
            status="unavailable",
            max_answers_per_question=max_answers_per_question,
            errors=execution.errors or ("resolved retrieval plan is unavailable",),
        )

    task_by_query = {
        task.query_id: task for task in execution.first_stage.tasks
    }
    relation_by_query: dict[str, list[GroundedRelationEvidence]] = {}
    for evidence in execution.second_stage_candidates:
        relation_by_query.setdefault(evidence.query_id, []).append(evidence)

    instructions = (
        "Answer the original Discovery question using only the supplied evidence "
        "catalog. Return at most max_answers distinct answers and fewer when fewer "
        "are supported. Do not use protocol knowledge from memory. Answer "
        "hypotheses and upstream Discovery results are non-evidentiary coverage "
        "hints, not facts or an answer whitelist. Independently inspect every "
        "evidence candidate and include each distinct supported answer responsive "
        "to the question even when no hypothesis proposed it; never state a hint "
        "without current-question evidence. For a carrier question, one answer "
        "represents one immediate carrier relation, with its IE, container, "
        "direction, interface, endpoint, and conditions attached as supporting "
        "details. For a reachability or transport-path question, one answer "
        "represents one complete supported path alternative: combine connected "
        "carrier and forwarding edges instead of returning those edges separately. "
        "When only a partial path is supported, state only that partial path and "
        "mark the evidence insufficient. When a path joins messages or layers "
        "that use different names, IEs, or fields for the same object, combine "
        "the upstream carrier, downstream representation, field definition or "
        "type reference, and procedure evidence when available; do not require a "
        "literal statement that the object was converted, and do not infer "
        "continuity from endpoint mentions alone. If a material handoff is "
        "unsupported, mark the answer partial and the evidence insufficient, "
        "then request repair when allowed. Distinguish the wireless or attacker-"
        "observable segment requested by the question from path segments supplied "
        "only as context. Keep different concrete protocol messages separate. For "
        "other Discovery angles, follow the original question's requested answer "
        "unit. Silently discard irrelevant, opposite-direction, contextual, "
        "duplicate, or unsupported candidates. Merge only conditional variants or "
        "duplicate evidence describing the same answer. Every answer must cite one "
        "to eight supplied evidence IDs copied exactly character-for-character; "
        "never shorten, reconstruct, or rename an evidence ID. Every stated relation "
        "must be directly "
        "supported by those excerpts. Keep evidence insufficient when any "
        "material candidate relation remains unresolved or a repair task has no "
        "qualifying evidence; do not mark sufficient merely because other "
        "answers are supported. If repair_allowed is "
        "true, return one to three gap-driven repair tasks. Each task targets "
        "one supplied unmet evidence_requirement and returns its exact "
        "requirement_id. Use null only for a material relation discovered during "
        "review that no supplied requirement covers. Each task covers one missing "
        "relation and contains one to three short specification-local evidence "
        "queries only when separate clauses or specifications are needed. Each "
        "query selects one allowed specification, targets one concrete local fact, "
        "and has one to three distinctive required_terms. For a cross-"
        "specification gap, place separate specification-local queries under the "
        "same unmet requirement. Reformulate from the missing relation; never "
        "repeat an initial query. "
        "Each repair task must list the supplied hypothesis_ids whose missing "
        "claim it addresses; use an empty list only for an independent gap. "
        "Do not repeat supported path edges. Prioritize missing answers before "
        "interface details. If "
        "repair_allowed is false, return no repair queries."
        " For every supplied hypothesis_id, return exactly one coverage item; "
        "do not merge different hypothesis_ids, including different IE or "
        "container relations. Answers without a hypothesis_id are independent "
        "discoveries. Evidence requirements are non-evidentiary checklists for "
        "the material local relations within each hypothesis. Hypothesis evidence "
        "packets only organize supplied evidence IDs by requirement. Inspect every "
        "non-empty group and every linked repair group before assigning coverage; "
        "the packets are not evidence and do not permit unsupported path joins."
    )
    receipts: list[CodexAgentReceipt] = []
    summaries: list[QuestionAnswerSummary] = []
    errors: list[str] = []
    for item in execution.resolved_plan.retrievals:
        if query_ids is not None and item.query_id not in query_ids:
            continue
        first_task = task_by_query[item.query_id]
        hypothesis_items = tuple(
            (answer_hypotheses_by_query or {}).get(item.query_id, ())
        )
        repair_context = tuple(
            (repair_context_by_query or {}).get(item.query_id, ())
        )
        agent_locator_context = tuple(
            (agent_locator_context_by_query or {}).get(item.query_id, ())
        )
        evidence_payload = [
            {
                "evidence_id": hit.hit_id,
                "stage": (
                    "repair"
                    if any(
                        source.startswith("repair:")
                        for source in hit.retrieval_sources
                    )
                    else "agent_locator"
                    if "agent_locator" in hit.retrieval_sources
                    else "hypothesis_route"
                    if hit.hit_id.startswith("HB-")
                    else "first_stage"
                ),
                "specification": hit.specification,
                "version": hit.version,
                "heading_path": list(hit.heading_path),
                "line_start": hit.line_start,
                "line_end": hit.line_end,
                "retrieval_sources": list(hit.retrieval_sources),
                "probe_ids": list(hit.probe_ids),
                "hypothesis_ids": list(_hit_hypothesis_ids(hit)),
                "matched_anchors": list(hit.matched_anchors),
                "excerpt": hit.excerpt,
            }
            for hit in first_task.hits
        ]
        evidence_payload.extend(
            {
                "evidence_id": evidence.evidence_id,
                "stage": "second_stage_relation_expansion",
                "specification": evidence.specification,
                "version": evidence.version,
                "heading_path": list(evidence.heading_path),
                "line_start": evidence.line_start,
                "line_end": evidence.line_end,
                "hypothesis_ids": [],
                "relation_kinds": list(evidence.relation_kinds),
                "matched_original_anchors": list(
                    evidence.matched_original_anchors
                ),
                "grounded_terms": [
                    term.surface for term in evidence.grounded_terms
                ],
                "excerpt": evidence.excerpt,
            }
            for evidence in relation_by_query.get(item.query_id, ())
        )
        available_ids = {
            value["evidence_id"] for value in evidence_payload
        }
        hypothesis_evidence_packets = _build_hypothesis_evidence_packets(
            hypothesis_items,
            evidence_payload,
            repair_context,
        )
        payload = {
            "query_id": item.query_id,
            "claim_id": item.claim_id,
            "angle": item.angle,
            "original_question": item.original_question,
            "original_anchors": [
                anchor.canonical for anchor in item.anchors
            ],
            "explicit_facets": [
                facet.name for facet in item.explicit_facets
            ],
            "max_answers": max_answers_per_question,
            "answer_hypotheses": list(hypothesis_items),
            "hypothesis_evidence_packets": hypothesis_evidence_packets,
            "upstream_discovery_results": list(
                (
                    upstream_discovery_results_by_query or {}
                ).get(item.query_id, ())
                if upstream_discovery_results_by_query is not None
                else upstream_discovery_results
            ),
            "allowed_repair_specifications": sorted(
                allowed_repair_specifications
            ),
            "repair_allowed": allow_repair,
            "repair_context": list(repair_context),
            "agent_locator_context": list(agent_locator_context),
            "evidence": evidence_payload,
        }
        try:
            response = summary_agent.parse(
                purpose=(
                    "select Discovery evidence and produce final semantic answers"
                ),
                prompt_version="discovery-evidence-selection-and-answer/v1.14",
                instructions=instructions,
                input_payload=payload,
                output_type=AgentRepairableQuestionAnswerProposal,
            )
        except Exception as exc:
            error = f"{item.query_id}: {type(exc).__name__}: {exc}"
            errors.append(error)
            summaries.append(QuestionAnswerSummary(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                evidence_status="unavailable",
                warnings=(error,),
            ))
            continue
        receipts.append(response.receipt)
        if response.output.query_id != item.query_id:
            error = (
                f"{item.query_id}: agent returned query_id "
                f"{response.output.query_id}"
            )
            errors.append(error)
            summaries.append(QuestionAnswerSummary(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                evidence_status="unavailable",
                warnings=(error,),
            ))
            continue

        hypothesis_ids = tuple(
            str(value["hypothesis_id"])
            for value in hypothesis_items
            if value.get("hypothesis_id")
        )
        hypothesis_id_set = set(hypothesis_ids)
        requirement_ids_by_hypothesis = {
            str(hypothesis["hypothesis_id"]): {
                str(requirement["requirement_id"])
                for requirement in hypothesis.get("evidence_requirements", ())
                if isinstance(requirement, Mapping)
                and requirement.get("requirement_id")
            }
            for hypothesis in hypothesis_items
            if hypothesis.get("hypothesis_id")
        }
        answers_by_hypothesis: dict[str, EvidenceBackedAnswer] = {}
        independent_answers: list[EvidenceBackedAnswer] = []
        warnings: list[str] = []
        seen_independent_statements: set[str] = set()
        proposed_answers = list(response.output.answers)
        if len(hypothesis_ids) == 1 and len(proposed_answers) == 1:
            proposed_answers[0] = proposed_answers[0].model_copy(update={
                "hypothesis_id": hypothesis_ids[0],
            })
        for proposed in proposed_answers:
            resolved_evidence_ids, unknown_ids, aliases = (
                _resolve_evidence_id_aliases(
                    proposed.evidence_ids,
                    available_ids,
                )
            )
            if aliases:
                warnings.append(
                    "normalized abbreviated evidence IDs: "
                    + ", ".join(
                        f"{source} -> {target}"
                        for source, target in aliases
                    )
                )
            if unknown_ids:
                warnings.append(
                    "skipped answer with unknown evidence IDs: "
                    + ", ".join(unknown_ids)
                )
                continue
            hypothesis_id = proposed.hypothesis_id
            if hypothesis_id is not None and hypothesis_id not in hypothesis_id_set:
                warnings.append(
                    f"answer references unknown hypothesis_id: {hypothesis_id}"
                )
                hypothesis_id = None
            normalized = proposed.model_copy(update={
                "hypothesis_id": hypothesis_id,
                "evidence_ids": tuple(dict.fromkeys(resolved_evidence_ids)),
            })
            if hypothesis_id is not None:
                if hypothesis_id in answers_by_hypothesis:
                    warnings.append(
                        f"duplicate answer for hypothesis_id: {hypothesis_id}"
                    )
                    continue
                answers_by_hypothesis[hypothesis_id] = normalized
                continue
            key = re.sub(r"\s+", " ", proposed.statement).strip().casefold()
            if key in seen_independent_statements:
                warnings.append("skipped duplicate independent answer")
                continue
            seen_independent_statements.add(key)
            independent_answers.append(normalized)

        answers: list[EvidenceBackedAnswer] = []
        for hypothesis_id in hypothesis_ids:
            answer = answers_by_hypothesis.get(hypothesis_id)
            if answer is None:
                hypothesis = next(
                    value for value in hypothesis_items
                    if value.get("hypothesis_id") == hypothesis_id
                )
                answer = EvidenceBackedAnswer(
                    answer_id=f"A-{item.query_id}:{hypothesis_id}",
                    hypothesis_id=hypothesis_id,
                    coverage_status="unresolved",
                    answer_type="other",
                    statement=(
                        "Candidate answer unresolved by the supplied evidence: "
                        f"{hypothesis.get('possible_answer', hypothesis_id)}"
                    ),
                    support_kind="negative_boundary",
                    evidence_ids=(),
                )
            answers.append(answer.model_copy(update={
                "answer_id": f"A-{item.query_id}:{hypothesis_id}",
                "hypothesis_id": hypothesis_id,
            }))
        remaining = max(0, max_answers_per_question - len(answers))
        answers.extend(
            answer.model_copy(update={
                "answer_id": f"A-{item.query_id}:independent-{index}",
                "hypothesis_id": None,
            })
            for index, answer in enumerate(independent_answers[:remaining], 1)
        )
        if any(answer.coverage_status == "unresolved" for answer in answers):
            final_evidence_status: Literal["sufficient", "insufficient"] = (
                "insufficient"
            )
        else:
            final_evidence_status = response.output.evidence_status
        unmet_hypothesis_ids = {
            answer.hypothesis_id
            for answer in answers
            if answer.hypothesis_id is not None
            and answer.coverage_status != "supported"
        }
        repair_queries: list[EvidenceRepairQuery] = []
        seen_repairs: set[
            tuple[str, tuple[tuple[str, str, str], ...]]
        ] = set()
        for proposed in response.output.repair_queries[:3]:
            valid_evidence_queries = tuple(
                query
                for query in proposed.evidence_queries
                if query.specification in allowed_repair_specifications
            )
            invalid_specifications = tuple(dict.fromkeys(
                query.specification
                for query in proposed.evidence_queries
                if query.specification not in allowed_repair_specifications
            ))
            if invalid_specifications:
                warnings.append(
                    f"repair query {proposed.repair_id} ignored unavailable "
                    "specifications: " + ", ".join(invalid_specifications)
                )
            if not valid_evidence_queries:
                warnings.append(
                    f"skipped repair query {proposed.repair_id}: no local query "
                    "targets an allowed specification"
                )
                continue
            key = (
                proposed.missing_claim.casefold(),
                tuple(
                    (
                        query.specification.casefold(),
                        query.lexical_query.casefold(),
                        query.semantic_query.casefold(),
                    )
                    for query in valid_evidence_queries
                ),
            )
            if key in seen_repairs:
                warnings.append("skipped duplicate repair query")
                continue
            seen_repairs.add(key)
            valid_hypothesis_ids = tuple(dict.fromkeys(
                hypothesis_id
                for hypothesis_id in proposed.hypothesis_ids
                if hypothesis_id in hypothesis_id_set
            ))
            unknown_hypothesis_ids = tuple(
                hypothesis_id
                for hypothesis_id in proposed.hypothesis_ids
                if hypothesis_id not in hypothesis_id_set
            )
            if unknown_hypothesis_ids:
                warnings.append(
                    "repair query ignored unknown hypothesis IDs: "
                    + ", ".join(unknown_hypothesis_ids)
                )
            if proposed.requirement_id is not None:
                valid_hypothesis_ids = tuple(
                    hypothesis_id
                    for hypothesis_id in valid_hypothesis_ids
                    if hypothesis_id in unmet_hypothesis_ids
                    if proposed.requirement_id
                    in requirement_ids_by_hypothesis.get(
                        hypothesis_id,
                        set(),
                    )
                )
                if not valid_hypothesis_ids:
                    warnings.append(
                        f"skipped repair query {proposed.repair_id}: "
                        f"requirement {proposed.requirement_id} is not an unmet "
                        "requirement of a linked hypothesis"
                    )
                    continue
            repair_queries.append(proposed.model_copy(update={
                "repair_id": f"R-{item.query_id}:{len(repair_queries) + 1}",
                "hypothesis_ids": valid_hypothesis_ids,
                "evidence_queries": valid_evidence_queries,
            }))
        if not allow_repair and repair_queries:
            warnings.append(
                "repair limit reached; additional repair queries were not used"
            )
            repair_queries = []
        summaries.append(QuestionAnswerSummary(
            query_id=item.query_id,
            claim_id=item.claim_id,
            angle=item.angle,
            original_question=item.original_question,
            status="completed" if answers else "no_answers",
            evidence_status=final_evidence_status,
            answers=tuple(answers),
            repair_queries=tuple(repair_queries),
            warnings=tuple(warnings),
        ))

    unavailable_count = sum(item.status == "unavailable" for item in summaries)
    status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if summaries and unavailable_count == len(summaries)
        else "partial"
        if unavailable_count or errors
        else "completed"
    )
    return EvidenceAnswerSummary(
        status=status,
        max_answers_per_question=max_answers_per_question,
        agent_receipts=tuple(receipts),
        questions=tuple(summaries),
        errors=tuple(errors),
    )


def summarize_selected_discovery_evidence(
    selection: DiscoveryEvidenceSelection,
    *,
    summary_agent: CodexStructuredAgent,
    max_answers_per_question: int = 10,
) -> EvidenceAnswerSummary:
    """Turn selected source groups into final, evidence-backed answers."""

    if not 1 <= max_answers_per_question <= 10:
        raise ValueError("max_answers_per_question must be between 1 and 10")
    if selection.status == "unavailable" and not selection.questions:
        return EvidenceAnswerSummary(
            status="unavailable",
            max_answers_per_question=max_answers_per_question,
            errors=selection.errors or ("selected evidence is unavailable",),
        )

    instructions = (
        "Produce the final semantic answer to one 3GPP Discovery question from "
        "the supplied selected source-evidence groups. Evidence groups are "
        "candidate support structures, not a list of answers, and their count "
        "does not determine the answer count. Return at most the requested "
        "maximum and return fewer when fewer distinct answers are supported. "
        "One answer must be one proposition that directly answers the original "
        "question. For carrier or reachability questions, one answer represents "
        "one independently answer-bearing carrier or path relation. Merge groups "
        "and excerpts that describe the same relation. The queried carried object, "
        "payload name, information element, endpoint, direction, interface, "
        "security condition, and procedure context are supporting facts for the "
        "relevant relation; do not emit them as separate answers merely because "
        "they appear in a separate excerpt or group. An excerpt that only defines "
        "or sends the queried object is not a carrier answer unless it establishes "
        "the carrier relation asked by the question. For other Discovery angles, "
        "apply the same principle: distinguish direct answers from facts that only "
        "qualify or support those answers. Do not require a fixed number of answers "
        "or a fixed end-to-end path schema. Every answer must cite one to eight "
        "supplied evidence IDs copied exactly character-for-character; never "
        "shorten, reconstruct, or rename an evidence ID. Combine excerpts only when every link needed for "
        "the stated relation is present. Use no protocol knowledge outside the "
        "provided excerpts, do not infer missing path edges, and do not introduce "
        "any message, IE, container, condition, direction, or fact absent from the "
        "evidence."
    )
    receipts: list[CodexAgentReceipt] = []
    summaries: list[QuestionAnswerSummary] = []
    errors: list[str] = list(selection.errors)

    for item in selection.questions:
        if item.status == "unavailable":
            summaries.append(QuestionAnswerSummary(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                evidence_status="unavailable",
                warnings=item.warnings,
            ))
            continue
        if not item.evidence_groups:
            summaries.append(QuestionAnswerSummary(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="no_answers",
                warnings=item.warnings,
            ))
            continue

        available_ids: set[str] = set()
        evidence_groups: list[dict[str, Any]] = []
        for group in item.evidence_groups:
            evidence_payload: list[dict[str, Any]] = []
            for evidence in group.evidence:
                available_ids.add(evidence.evidence_id)
                evidence_payload.append({
                    "evidence_id": evidence.evidence_id,
                    "source_stage": evidence.source_stage,
                    "specification": evidence.specification,
                    "version": evidence.version,
                    "heading_path": list(evidence.heading_path),
                    "line_start": evidence.line_start,
                    "line_end": evidence.line_end,
                    "excerpt": evidence.excerpt,
                })
            evidence_groups.append({
                "group_id": group.group_id,
                "selection_basis": group.selection_basis,
                "hypothesis_ids": list(group.hypothesis_ids),
                "evidence": evidence_payload,
            })

        payload = {
            "query_id": item.query_id,
            "claim_id": item.claim_id,
            "angle": item.angle,
            "original_question": item.original_question,
            "max_answers": max_answers_per_question,
            "selected_evidence_groups": evidence_groups,
        }
        try:
            response = summary_agent.parse(
                purpose="summarize selected Discovery evidence",
                prompt_version="discovery-selected-evidence-answer-summary/v1.0",
                instructions=instructions,
                input_payload=payload,
                output_type=AgentQuestionAnswerProposal,
            )
        except Exception as exc:
            error = f"{item.query_id}: {type(exc).__name__}: {exc}"
            errors.append(error)
            summaries.append(QuestionAnswerSummary(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                evidence_status="unavailable",
                warnings=(*item.warnings, error),
            ))
            continue

        receipts.append(response.receipt)
        if response.output.query_id != item.query_id:
            error = (
                f"{item.query_id}: agent returned query_id "
                f"{response.output.query_id}"
            )
            errors.append(error)
            summaries.append(QuestionAnswerSummary(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                evidence_status="unavailable",
                warnings=(*item.warnings, error),
            ))
            continue

        answers: list[EvidenceBackedAnswer] = []
        warnings: list[str] = list(item.warnings)
        seen_statements: set[str] = set()
        for proposed in response.output.answers[:max_answers_per_question]:
            resolved_evidence_ids, unknown_ids, aliases = (
                _resolve_evidence_id_aliases(
                    proposed.evidence_ids,
                    available_ids,
                )
            )
            if aliases:
                warnings.append(
                    "normalized abbreviated evidence IDs: "
                    + ", ".join(
                        f"{source} -> {target}"
                        for source, target in aliases
                    )
                )
            if unknown_ids:
                warnings.append(
                    "skipped answer with unknown evidence IDs: "
                    + ", ".join(unknown_ids)
                )
                continue
            key = re.sub(r"\s+", " ", proposed.statement).strip().casefold()
            if key in seen_statements:
                warnings.append("skipped duplicate answer statement")
                continue
            seen_statements.add(key)
            answers.append(proposed.model_copy(update={
                "answer_id": f"A-{item.query_id}:{len(answers) + 1}",
                "evidence_ids": tuple(dict.fromkeys(resolved_evidence_ids)),
            }))
        summaries.append(QuestionAnswerSummary(
            query_id=item.query_id,
            claim_id=item.claim_id,
            angle=item.angle,
            original_question=item.original_question,
            status="completed" if answers else "no_answers",
            answers=tuple(answers),
            warnings=tuple(warnings),
        ))

    unavailable_count = sum(item.status == "unavailable" for item in summaries)
    status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if summaries and unavailable_count == len(summaries)
        else "partial"
        if unavailable_count or errors or selection.status == "partial"
        else "completed"
    )
    return EvidenceAnswerSummary(
        status=status,
        max_answers_per_question=max_answers_per_question,
        agent_receipts=tuple(receipts),
        questions=tuple(summaries),
        errors=tuple(dict.fromkeys(errors)),
    )


def _compact_agent_evidence_excerpt(
    excerpt: str,
    *,
    focus_terms: tuple[str, ...],
    max_chars: int = 1800,
) -> str:
    """Create a small review window while leaving stored evidence untouched."""

    if len(excerpt) <= max_chars:
        return excerpt
    lines = excerpt.splitlines()
    folded_terms = tuple(
        _fold_protocol_text(term)
        for term in focus_terms
        if _useful_relation_term(term)
    )
    scored: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        folded = _fold_protocol_text(line)
        score = sum(8 for term in folded_terms if term and term in folded)
        if line.strip().startswith("|") and score:
            score += 6
        if any(
            cue in folded
            for cue in (
                "shall include", "may include", "contains", "conveys",
                "message content", "type/reference", "purpose of the",
            )
        ):
            score += 2
        if score:
            scored.append((score, index))

    if not scored:
        return excerpt[:max_chars].rstrip() + "\n[review window truncated]"

    selected_indexes: set[int] = set()
    for _, center in sorted(scored, key=lambda item: (-item[0], item[1])):
        selected_indexes.update(
            range(max(0, center - 2), min(len(lines), center + 3))
        )
        rendered = "\n".join(lines[index] for index in sorted(selected_indexes))
        if len(rendered) >= max_chars:
            break

    parts: list[str] = []
    previous: int | None = None
    for index in sorted(selected_indexes):
        if previous is not None and index != previous + 1:
            parts.append("[...]")
        parts.append(lines[index])
        previous = index
    rendered = "\n".join(parts)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + "\n[review window truncated]"
    return rendered


def _relation_candidate_name(evidence: GroundedRelationEvidence) -> str:
    message_type = re.search(
        r"Message\s+type\s*:\s*(?P<name>[A-Z][A-Z0-9 /_-]{2,100})",
        evidence.excerpt,
    )
    if message_type is not None:
        return message_type.group("name").strip()
    message = _RELATION_MESSAGE.search(evidence.excerpt)
    if message is not None:
        return message.group(0).strip()
    if evidence.heading_path:
        return evidence.heading_path[-1]
    return evidence.evidence_id


def select_expanded_discovery_evidence(
    execution: ExpandedRetrievalExecution,
    *,
    selection_agent: CodexStructuredAgent,
    max_evidence_groups_per_question: int = 10,
    max_agent_excerpt_chars: int = 1800,
    answer_hypotheses_by_query: Mapping[
        str,
        tuple[Mapping[str, Any], ...],
    ] | None = None,
) -> DiscoveryEvidenceSelection:
    """Select source-faithful evidence groups without authoring answer text."""

    if not 1 <= max_evidence_groups_per_question <= 10:
        raise ValueError("max_evidence_groups_per_question must be between 1 and 10")
    if max_agent_excerpt_chars < 400:
        raise ValueError("max_agent_excerpt_chars must be at least 400")
    if execution.resolved_plan is None:
        return DiscoveryEvidenceSelection(
            status="unavailable",
            max_evidence_groups_per_question=max_evidence_groups_per_question,
            errors=execution.errors or ("resolved retrieval plan is unavailable",),
        )

    first_by_query = {
        task.query_id: task for task in execution.first_stage.tasks
    }
    second_by_query: dict[str, list[GroundedRelationEvidence]] = {}
    for evidence in execution.second_stage_candidates:
        second_by_query.setdefault(evidence.query_id, []).append(evidence)

    hypotheses_by_query = dict(answer_hypotheses_by_query or {})
    instructions = (
        "Select original 3GPP specification evidence for one Discovery question. "
        "Do not write, paraphrase, or summarize an answer. Return only groups of "
        "supplied evidence IDs. Each group must support one semantically distinct, "
        "answer-bearing proposition to the original question, rather than merely "
        "one named protocol entity. For carrier or reachability questions, one "
        "group represents one distinct carrier or path relation. A group may combine "
        "up to four excerpts when they support that same proposition or form a "
        "grounded relation chain, such as carried-object-to-IE and IE-to-message. "
        "Prefer direct normative sentences and exact message-content "
        "table rows. Evidence marked hypothesis_route is original specification "
        "text retrieved from an unverified answer hypothesis; judge only the text, "
        "not the hypothesis that led to it. Treat second-stage material as candidate discovery: select it "
        "only when it establishes a distinct candidate absent from the first-stage "
        "evidence, or when it is required to complete such a relation chain. Do not "
        "select generic term overlap, unrelated tables, headings without supporting "
        "content, or duplicate formulations. Use no protocol knowledge outside the "
        "provided excerpts. Perform two reviews in this single response. First, "
        "assess every supplied answer hypothesis as supported, partial, or "
        "unsupported using only the evidence catalog, cite evidence IDs for "
        "supported or partial assessments, and place hypothesis-backed groups in "
        "evidence_groups. Second, treat the hypotheses as non-exhaustive and inspect "
        "the entire evidence catalog again against the original question. Place "
        "direct, distinct answer-bearing propositions not adequately represented by the hypotheses in "
        "independent_evidence_groups. A hypothesis is never evidence by itself and "
        "must never act as an answer whitelist. Do not create an independent group "
        "for a payload definition, carried message, endpoint, direction, interface, "
        "security condition, or procedure context when it only supports an existing "
        "answer proposition. Attach that supporting evidence to the relevant group. "
        "Different protocol-layer messages may share a group when their excerpts "
        "jointly establish the same carrier or forwarding relation. Do not combine "
        "unrelated alternative carriers or unrelated answer propositions in one group. "
        "Do not require a fixed source/interface/carrier/relay template and do not "
        "infer missing path edges. Keep the combined number of hypothesis-backed and "
        "independent groups within max_evidence_groups. Return fewer than the maximum "
        "when fewer distinct source groups are supported. The system will copy the "
        "corresponding full original excerpts after selection."
    )
    receipts: list[CodexAgentReceipt] = []
    questions: list[QuestionEvidenceSelection] = []
    errors: list[str] = []

    for item in execution.resolved_plan.retrievals:
        first_task = first_by_query[item.query_id]
        anchors = tuple(anchor.canonical for anchor in item.anchors)
        source_catalog: dict[str, SelectedSpecificationEvidence] = {}
        review_catalog: list[dict[str, Any]] = []

        for hit in first_task.hits:
            source_stage: EvidenceSourceStage = (
                "hypothesis_route"
                if hit.hit_id.startswith("HB-")
                else "first_stage"
            )
            source_catalog[hit.hit_id] = SelectedSpecificationEvidence(
                evidence_id=hit.hit_id,
                source_stage=source_stage,
                chunk_id=hit.chunk_id,
                specification=hit.specification,
                version=hit.version,
                source_path=hit.source_path,
                document_sha256=hit.document_sha256,
                heading_path=hit.heading_path,
                line_start=hit.line_start,
                line_end=hit.line_end,
                excerpt=hit.excerpt,
                retrieval_sources=hit.retrieval_sources,
            )
            review_catalog.append({
                "evidence_id": hit.hit_id,
                "source_stage": source_stage,
                "specification": hit.specification,
                "version": hit.version,
                "heading_path": list(hit.heading_path),
                "line_start": hit.line_start,
                "line_end": hit.line_end,
                "retrieval_sources": list(hit.retrieval_sources),
                "review_excerpt": _compact_agent_evidence_excerpt(
                    hit.excerpt,
                    focus_terms=tuple((*anchors, *hit.matched_anchors)),
                    max_chars=max_agent_excerpt_chars,
                ),
            })

        relation_candidates: dict[tuple[str, tuple[str, ...]], list[str]] = {}
        for evidence in second_by_query.get(item.query_id, ()):
            source_catalog[evidence.evidence_id] = SelectedSpecificationEvidence(
                evidence_id=evidence.evidence_id,
                source_stage="second_stage",
                chunk_id=evidence.chunk_id,
                specification=evidence.specification,
                version=evidence.version,
                source_path=evidence.source_path,
                document_sha256=evidence.document_sha256,
                heading_path=evidence.heading_path,
                line_start=evidence.line_start,
                line_end=evidence.line_end,
                excerpt=evidence.excerpt,
                relation_kinds=evidence.relation_kinds,
                bridge_terms=evidence.bridge_terms,
            )
            candidate_name = _relation_candidate_name(evidence)
            candidate_key = (
                candidate_name.casefold(),
                tuple(sorted(value.casefold() for value in evidence.bridge_terms)),
            )
            relation_candidates.setdefault(candidate_key, []).append(
                evidence.evidence_id
            )
            focus_terms = tuple((
                *anchors,
                *evidence.bridge_terms,
                *(term.surface for term in evidence.grounded_terms),
            ))
            review_catalog.append({
                "evidence_id": evidence.evidence_id,
                "source_stage": "second_stage",
                "candidate_name": candidate_name,
                "bridge_terms": list(evidence.bridge_terms),
                "relation_kinds": list(evidence.relation_kinds),
                "specification": evidence.specification,
                "version": evidence.version,
                "heading_path": list(evidence.heading_path),
                "line_start": evidence.line_start,
                "line_end": evidence.line_end,
                "review_excerpt": _compact_agent_evidence_excerpt(
                    evidence.excerpt,
                    focus_terms=focus_terms,
                    max_chars=max_agent_excerpt_chars,
                ),
            })

        candidate_payload = [
            {
                "candidate_id": f"C-{item.query_id}:{index}",
                "candidate_name": key[0],
                "bridge_terms": list(key[1]),
                "evidence_ids": evidence_ids,
            }
            for index, (key, evidence_ids) in enumerate(
                relation_candidates.items(),
                1,
            )
        ]
        answer_hypotheses = hypotheses_by_query.get(item.query_id, ())
        payload = {
            "query_id": item.query_id,
            "claim_id": item.claim_id,
            "angle": item.angle,
            "original_question": item.original_question,
            "original_anchors": list(anchors),
            "explicit_facets": [facet.name for facet in item.explicit_facets],
            "max_evidence_groups": max_evidence_groups_per_question,
            "answer_hypotheses": [dict(value) for value in answer_hypotheses],
            "relation_candidates": candidate_payload,
            "evidence_catalog": review_catalog,
        }
        try:
            response = selection_agent.parse(
                purpose="select source-faithful Discovery evidence",
                prompt_version="discovery-evidence-selection/v1.3",
                instructions=instructions,
                input_payload=payload,
                output_type=AgentQuestionEvidenceSelectionProposal,
            )
        except Exception as exc:
            error = f"{item.query_id}: {type(exc).__name__}: {exc}"
            errors.append(error)
            questions.append(QuestionEvidenceSelection(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                warnings=(error,),
            ))
            continue
        receipts.append(response.receipt)
        if response.output.query_id != item.query_id:
            error = (
                f"{item.query_id}: agent returned query_id "
                f"{response.output.query_id}"
            )
            errors.append(error)
            questions.append(QuestionEvidenceSelection(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                warnings=(error,),
            ))
            continue

        warnings: list[str] = []
        known_hypotheses = {
            str(value.get("hypothesis_id")): value
            for value in answer_hypotheses
            if value.get("hypothesis_id")
        }
        proposed_assessments = {
            value.hypothesis_id: value
            for value in response.output.hypothesis_assessments
            if value.hypothesis_id in known_hypotheses
        }
        for value in response.output.hypothesis_assessments:
            if value.hypothesis_id not in known_hypotheses:
                warnings.append(
                    "ignored assessment for unknown hypothesis: "
                    + value.hypothesis_id
                )

        assessments: list[HypothesisEvidenceAssessment] = []
        for hypothesis_id in known_hypotheses:
            proposed = proposed_assessments.get(hypothesis_id)
            if proposed is None:
                warnings.append(
                    f"agent omitted hypothesis assessment: {hypothesis_id}"
                )
                assessments.append(HypothesisEvidenceAssessment(
                    hypothesis_id=hypothesis_id,
                    status="unassessed",
                ))
                continue
            evidence_ids = tuple(dict.fromkeys(
                value
                for value in proposed.evidence_ids
                if value in source_catalog
            ))
            unknown_evidence = tuple(
                value
                for value in proposed.evidence_ids
                if value not in source_catalog
            )
            if unknown_evidence:
                warnings.append(
                    f"{hypothesis_id} assessment ignored unknown evidence IDs: "
                    + ", ".join(unknown_evidence)
                )
            status: HypothesisSupportStatus = proposed.status
            if status in {"supported", "partial"} and not evidence_ids:
                warnings.append(
                    f"{hypothesis_id} was marked {status} without valid evidence"
                )
                status = "unassessed"
            if status == "unsupported":
                evidence_ids = ()
            assessments.append(HypothesisEvidenceAssessment(
                hypothesis_id=hypothesis_id,
                status=status,
                evidence_ids=evidence_ids,
            ))

        groups: list[SelectedEvidenceGroup] = []
        group_indexes: dict[tuple[str, ...], int] = {}
        for proposed in response.output.evidence_groups[
            :max_evidence_groups_per_question
        ]:
            evidence_ids = tuple(dict.fromkeys(proposed.evidence_ids))
            unknown = tuple(
                evidence_id
                for evidence_id in evidence_ids
                if evidence_id not in source_catalog
            )
            if unknown:
                warnings.append(
                    "skipped group with unknown evidence IDs: " + ", ".join(unknown)
                )
                continue
            hypothesis_ids = tuple(dict.fromkeys(
                hypothesis_id
                for hypothesis_id in proposed.hypothesis_ids
                if hypothesis_id in known_hypotheses
            ))
            unknown_hypotheses = tuple(
                hypothesis_id
                for hypothesis_id in proposed.hypothesis_ids
                if hypothesis_id not in known_hypotheses
            )
            if unknown_hypotheses:
                warnings.append(
                    "ignored unknown hypothesis IDs in evidence group: "
                    + ", ".join(unknown_hypotheses)
                )
            group_key = tuple(sorted(evidence_ids))
            existing_index = group_indexes.get(group_key)
            if existing_index is not None:
                existing = groups[existing_index]
                groups[existing_index] = existing.model_copy(update={
                    "hypothesis_ids": tuple(dict.fromkeys((
                        *existing.hypothesis_ids,
                        *hypothesis_ids,
                    ))),
                })
                continue
            group_indexes[group_key] = len(groups)
            selected = tuple(source_catalog[value] for value in evidence_ids)
            groups.append(SelectedEvidenceGroup(
                group_id=f"E-{item.query_id}:{len(groups) + 1}",
                source_stages=tuple(dict.fromkeys(
                    value.source_stage for value in selected
                )),
                hypothesis_ids=hypothesis_ids,
                selection_basis="hypothesis_assessment",
                evidence=selected,
            ))

        # A supported hypothesis must not disappear merely because the agent
        # forgot to duplicate its cited evidence in evidence_groups. Reuse or
        # append the agent's own cited IDs; this is a soft completeness guard,
        # not a protocol-specific path validator.
        for assessment in assessments:
            if assessment.status not in {"supported", "partial"}:
                continue
            if any(
                assessment.hypothesis_id in group.hypothesis_ids
                for group in groups
            ):
                continue
            assessment_ids = tuple(sorted(assessment.evidence_ids))
            matching_index = next(
                (
                    index
                    for index, group in enumerate(groups)
                    if set(assessment.evidence_ids).issubset(
                        evidence.evidence_id for evidence in group.evidence
                    )
                ),
                None,
            )
            if matching_index is not None:
                existing = groups[matching_index]
                groups[matching_index] = existing.model_copy(update={
                    "hypothesis_ids": tuple(dict.fromkeys((
                        *existing.hypothesis_ids,
                        assessment.hypothesis_id,
                    ))),
                })
                continue
            if len(groups) >= max_evidence_groups_per_question:
                warnings.append(
                    "supported hypothesis could not be added because the evidence "
                    f"group limit was reached: {assessment.hypothesis_id}"
                )
                continue
            selected = tuple(
                source_catalog[value] for value in assessment.evidence_ids
            )
            groups.append(SelectedEvidenceGroup(
                group_id=f"E-{item.query_id}:{len(groups) + 1}",
                source_stages=tuple(dict.fromkeys(
                    value.source_stage for value in selected
                )),
                hypothesis_ids=(assessment.hypothesis_id,),
                selection_basis="hypothesis_assessment",
                evidence=selected,
            ))
            group_indexes[assessment_ids] = len(groups) - 1

        # Hypotheses are deliberately non-exhaustive. Keep direct answers found
        # by the agent's independent second review even when no hypothesis named
        # them. This runs in the same model call and stores only source IDs.
        for proposed in response.output.independent_evidence_groups:
            if len(groups) >= max_evidence_groups_per_question:
                warnings.append(
                    "independent evidence group limit reached; remaining groups "
                    "were not added"
                )
                break
            evidence_ids = tuple(dict.fromkeys(proposed.evidence_ids))
            unknown = tuple(
                evidence_id
                for evidence_id in evidence_ids
                if evidence_id not in source_catalog
            )
            if unknown:
                warnings.append(
                    "skipped independent group with unknown evidence IDs: "
                    + ", ".join(unknown)
                )
                continue
            group_key = tuple(sorted(evidence_ids))
            if group_key in group_indexes:
                continue
            selected = tuple(source_catalog[value] for value in evidence_ids)
            group_indexes[group_key] = len(groups)
            groups.append(SelectedEvidenceGroup(
                group_id=f"E-{item.query_id}:{len(groups) + 1}",
                source_stages=tuple(dict.fromkeys(
                    value.source_stage for value in selected
                )),
                hypothesis_ids=(),
                selection_basis="original_question_independent",
                evidence=selected,
            ))
        questions.append(QuestionEvidenceSelection(
            query_id=item.query_id,
            claim_id=item.claim_id,
            angle=item.angle,
            original_question=item.original_question,
            status="completed" if groups else "no_evidence",
            hypothesis_assessments=tuple(assessments),
            evidence_groups=tuple(groups),
            warnings=tuple(warnings),
        ))

    unavailable_count = sum(value.status == "unavailable" for value in questions)
    status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if questions and unavailable_count == len(questions)
        else "partial"
        if unavailable_count or errors
        else "completed"
    )
    return DiscoveryEvidenceSelection(
        status=status,
        max_evidence_groups_per_question=max_evidence_groups_per_question,
        agent_receipts=tuple(receipts),
        questions=tuple(questions),
        errors=tuple(errors),
    )


def execute_curated_discovery_retrieval(
    plan: DiscoveryRetrievalPlanV3,
    *,
    app_config: AppConfig | None = None,
    first_stage_hits: int = 12,
    relation_candidates_per_question: int = 64,
    final_evidence_hits: int = 20,
    retriever: HybridMarkdownRetriever | None = None,
) -> CuratedRetrievalExecution:
    """Retrieve once, expand corpus-grounded relations, and curate evidence."""

    if first_stage_hits < 1:
        raise ValueError("first_stage_hits must be positive")
    if relation_candidates_per_question < 1:
        raise ValueError("relation_candidates_per_question must be positive")
    if final_evidence_hits < 1:
        raise ValueError("final_evidence_hits must be positive")
    cfg = (app_config or load_config()).retrieval
    try:
        active_retriever = retriever or HybridMarkdownRetriever(cfg)
        resolved = resolve_discovery_retrieval_plan(plan, active_retriever)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        unavailable = PlannedRetrievalExecution(
            status="unavailable",
            embedding_model=cfg.embedding_model,
            corpus_version_policy=cfg.version_policy,
            corpus_release=cfg.target_release,
            errors=(error,),
        )
        return CuratedRetrievalExecution(
            status="unavailable",
            first_stage=unavailable,
            errors=(error,),
        )

    # Explicit-reference and relation discovery remain evidence channels, but
    # there is no Agent-authored query and no second hybrid retrieval pass.
    first_items = tuple(
        item.model_copy(update={
            "reference_expansion": ReferenceExpansionPolicy(enabled=False),
        })
        for item in resolved.retrievals
    )
    resolved_first = resolved.model_copy(update={"retrievals": first_items})
    first_tasks = tuple(
        _execute_task(
            active_retriever,
            item,
            final_hits=first_stage_hits,
            retrieval_stage=1,
        )
        for item in first_items
    )
    first_errors = tuple(error for task in first_tasks for error in task.errors)
    first_status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if first_tasks and all(task.status == "unavailable" for task in first_tasks)
        else "partial"
        if any(task.status in {"partial", "unavailable"} for task in first_tasks)
        else "completed"
    )
    first_execution = PlannedRetrievalExecution(
        status=first_status,
        resolved_plan=resolved_first,
        index_fingerprint=active_retriever.fingerprint,
        index_manifest_path=active_retriever.index_manifest_path,
        embedding_model=cfg.embedding_model,
        corpus_version_policy=cfg.version_policy,
        corpus_release=cfg.target_release,
        tasks=first_tasks,
        errors=first_errors,
    )
    relation_candidates = _build_grounded_relation_evidence(
        active_retriever,
        resolved_first,
        max_per_question=relation_candidates_per_question,
    )
    curated_tasks = curate_discovery_evidence(
        resolved_first,
        first_tasks,
        relation_candidates,
        final_evidence_hits=final_evidence_hits,
    )
    status: Literal["completed", "partial", "unavailable"] = (
        "partial" if first_errors else "completed"
    )
    return CuratedRetrievalExecution(
        status=status,
        resolved_plan=resolved_first,
        first_stage=first_execution,
        relation_candidates=relation_candidates,
        curated_tasks=curated_tasks,
        errors=first_errors,
    )


def execute_dual_stage_discovery_retrieval(
    plan: DiscoveryRetrievalPlanV3,
    *,
    expansion_agent: CodexStructuredAgent,
    app_config: AppConfig | None = None,
    first_stage_hits: int = 12,
    second_stage_hits: int = 12,
    retriever: HybridMarkdownRetriever | None = None,
) -> DualStageRetrievalExecution:
    """Retrieve, derive grounded follow-up probes with Codex, then retrieve again."""

    if first_stage_hits < 1 or second_stage_hits < 1:
        raise ValueError("stage hit limits must be positive")
    cfg = (app_config or load_config()).retrieval
    try:
        active_retriever = retriever or HybridMarkdownRetriever(cfg)
        resolved = resolve_discovery_retrieval_plan(plan, active_retriever)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        unavailable = PlannedRetrievalExecution(
            status="unavailable",
            embedding_model=cfg.embedding_model,
            corpus_version_policy=cfg.version_policy,
            corpus_release=cfg.target_release,
            errors=(error,),
        )
        return DualStageRetrievalExecution(
            status="unavailable",
            first_stage=unavailable,
            errors=(error,),
        )

    first_items = tuple(
        item.model_copy(update={
            "reference_expansion": ReferenceExpansionPolicy(enabled=False),
        })
        for item in resolved.retrievals
    )
    resolved_first = resolved.model_copy(update={"retrievals": first_items})
    first_tasks = tuple(
        _execute_task(
            active_retriever,
            item,
            final_hits=first_stage_hits,
            retrieval_stage=1,
        )
        for item in first_items
    )
    first_execution = PlannedRetrievalExecution(
        status=(
            "unavailable"
            if first_tasks and all(task.status == "unavailable" for task in first_tasks)
            else "partial"
            if any(task.status in {"partial", "unavailable"} for task in first_tasks)
            else "completed"
        ),
        index_fingerprint=active_retriever.fingerprint,
        index_manifest_path=active_retriever.index_manifest_path,
        embedding_model=cfg.embedding_model,
        corpus_version_policy=cfg.version_policy,
        corpus_release=cfg.target_release,
        tasks=first_tasks,
        errors=tuple(error for task in first_tasks for error in task.errors),
    )
    relation_evidence = _build_grounded_relation_evidence(
        active_retriever,
        resolved_first,
    )
    try:
        expansion = _grounded_followup_plan(
            resolved_plan=resolved_first,
            first_stage_tasks=first_tasks,
            expansion_agent=expansion_agent,
            available_specifications=frozenset(
                document.specification
                for document in active_retriever.documents
                if document.specification is not None
            ),
            relation_evidence=relation_evidence,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return DualStageRetrievalExecution(
            status="partial" if first_tasks else "unavailable",
            resolved_first_stage_plan=resolved_first,
            relation_evidence=relation_evidence,
            first_stage=first_execution,
            merged_tasks=first_tasks,
            errors=(*first_execution.errors, error),
        )

    first_by_query = {item.query_id: item for item in first_items}
    second_tasks: list[PlannedRetrievalTaskResult] = []
    for followup in expansion.retrievals:
        if not followup.retrieval_probes:
            second_tasks.append(PlannedRetrievalTaskResult(
                retrieval_id=followup.retrieval_id,
                query_id=followup.query_id,
                planner_action="planned",
                status="no_hits",
            ))
            continue
        second_tasks.append(_execute_task(
            active_retriever,
            _second_stage_item(first_by_query[followup.query_id], followup),
            final_hits=second_stage_hits,
            retrieval_stage=2,
        ))
    second_task_tuple = tuple(second_tasks)
    second_execution = PlannedRetrievalExecution(
        status=(
            "unavailable"
            if second_task_tuple and all(
                task.status == "unavailable" for task in second_task_tuple
            )
            else "partial"
            if any(
                task.status in {"partial", "unavailable"}
                for task in second_task_tuple
            )
            else "completed"
        ),
        index_fingerprint=active_retriever.fingerprint,
        index_manifest_path=active_retriever.index_manifest_path,
        embedding_model=cfg.embedding_model,
        corpus_version_policy=cfg.version_policy,
        corpus_release=cfg.target_release,
        tasks=second_task_tuple,
        errors=tuple(
            error for task in second_task_tuple for error in task.errors
        ),
    )
    merged = _merge_stage_tasks(first_tasks, second_task_tuple)
    all_errors = (*first_execution.errors, *second_execution.errors)
    status: Literal["completed", "partial", "unavailable"] = (
        "partial" if all_errors else "completed"
    )
    return DualStageRetrievalExecution(
        status=status,
        resolved_first_stage_plan=resolved_first,
        relation_evidence=relation_evidence,
        expansion_plan=expansion,
        first_stage=first_execution,
        second_stage=second_execution,
        merged_tasks=merged,
        errors=all_errors,
    )


def execute_planned_discovery_retrieval(
    plan: DiscoveryRetrievalPlanV3,
    *,
    app_config: AppConfig | None = None,
    final_hits: int = 8,
    retriever: HybridMarkdownRetriever | None = None,
    reserve_probe_candidates: bool = False,
    ranking_mode: Literal[
        "full",
        "rrf",
        "rrf_anchor",
        "rrf_heading",
        "rrf_noise",
        "rrf_anchor_heading",
        "rrf_anchor_single",
        "rrf_anchor_full_coverage",
        "rrf_anchor_single_full_coverage",
        "rrf_anchor_single_if_full",
    ] = "full",
    anchor_weight: float = 0.004,
    single_channel_max_rank: int = 3,
    single_channel_weight: float = 1.0,
    full_coverage_bonus: float = 0.004,
    reranker: CrossEncoderReranker | None = None,
) -> PlannedRetrievalExecution:
    """Resolve routes, retrieve passages, and follow explicit TS references.

    ``reserve_probe_candidates`` preserves the historical channel/probe
    reservation behavior by default. Evaluation experiments may disable it
    to rank every deduplicated candidate only by the deterministic score.
    """

    if final_hits < 1:
        raise ValueError("final_hits must be positive")
    cfg = (app_config or load_config()).retrieval
    try:
        active_retriever = retriever or HybridMarkdownRetriever(cfg)
        resolved_plan = resolve_discovery_retrieval_plan(plan, active_retriever)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return PlannedRetrievalExecution(
            status="unavailable",
            embedding_model=cfg.embedding_model,
            reranker=(reranker.metadata() if reranker is not None else None),
            corpus_version_policy=cfg.version_policy,
            corpus_release=cfg.target_release,
            tasks=tuple(
                PlannedRetrievalTaskResult(
                    retrieval_id=item.retrieval_id,
                    query_id=item.query_id,
                    planner_action=item.planner_action,
                    status="unavailable",
                    errors=(error,),
                )
                for item in plan.retrievals
            ),
            errors=(error,),
        )

    tasks = tuple(
        _execute_task(
            active_retriever,
            item,
            final_hits=final_hits,
            reserve_probe_candidates=reserve_probe_candidates,
            ranking_mode=ranking_mode,
            anchor_weight=anchor_weight,
            single_channel_max_rank=single_channel_max_rank,
            single_channel_weight=single_channel_weight,
            full_coverage_bonus=full_coverage_bonus,
            reranker=reranker,
        )
        for item in resolved_plan.retrievals
    )
    if tasks and all(item.status == "unavailable" for item in tasks):
        status: Literal["completed", "partial", "unavailable"] = "unavailable"
    elif any(item.status in {"partial", "unavailable"} for item in tasks):
        status = "partial"
    else:
        status = "completed"
    return PlannedRetrievalExecution(
        status=status,
        resolved_plan=resolved_plan,
        index_fingerprint=active_retriever.fingerprint,
        index_manifest_path=active_retriever.index_manifest_path,
        embedding_model=cfg.embedding_model,
        reranker=(reranker.metadata() if reranker is not None else None),
        corpus_version_policy=cfg.version_policy,
        corpus_release=cfg.target_release,
        tasks=tasks,
        errors=tuple(error for task in tasks for error in task.errors),
    )


__all__ = [
    "DualStageRetrievalExecution",
    "DiscoveryRetrievalPlanV2",
    "DiscoveryRetrievalPlanV3",
    "EvidenceExpansionBatch",
    "EvidenceGroundedExpansionPlan",
    "EvidenceGroundedFollowup",
    "EvidenceTermProposal",
    "ExplicitFacet",
    "FollowupProbeProposal",
    "LLMAnchorBatch",
    "LLMAnchorProposal",
    "LLMQuestionAnchors",
    "PLANNER_PROFILE",
    "PlannedDiscoveryRetrieval",
    "PlannedRetrievalExecution",
    "PlannedRetrievalHit",
    "PlannedRetrievalTaskResult",
    "QuestionExpansionProposal",
    "QueryAnchor",
    "ReferenceExpansionPolicy",
    "RetrievalProbe",
    "RetrievalSubquery",
    "RoutingPolicy",
    "SourceSpan",
    "SpecificationRoute",
    "compile_discovery_retrieval_plan",
    "execute_dual_stage_discovery_retrieval",
    "execute_planned_discovery_retrieval",
    "resolve_discovery_retrieval_plan",
]

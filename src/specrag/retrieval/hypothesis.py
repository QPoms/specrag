"""Agent-inspired retrieval hypotheses executed as a parallel RAG channel.

The agent is allowed to propose likely terminology, protocol-layer bridges, and
specification families. Those proposals are search hints only. Every retained
result remains an original corpus chunk and downstream selection is forbidden
from treating an unsupported proposal as evidence.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..infrastructure.structured_agent import CodexAgentReceipt, CodexStructuredAgent
from .planner import (
    DiscoveryRetrievalPlanV3,
    PlannedDiscoveryRetrieval,
    PlannedRetrievalExecution,
    PlannedRetrievalHit,
    PlannedRetrievalTaskResult,
    ReferenceExpansionPolicy,
    RetrievalProbe,
    SpecificationRoute,
    execute_planned_discovery_retrieval,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


HypothesisScope = Literal[
    "same_layer",
    "adjacent_layer",
    "procedure_context",
    "security_context",
    "state_context",
    "other",
]


class AgentEvidenceQuery(_StrictModel):
    """One specification-local lookup for an evidence requirement."""

    specification: str = Field(min_length=1)
    lexical_query: str = Field(min_length=1)
    semantic_query: str = Field(min_length=1)
    required_terms: tuple[str, ...] = Field(min_length=1, max_length=3)

    @field_validator(
        "specification",
        "lexical_query",
        "semantic_query",
    )
    @classmethod
    def _non_blank_query_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("evidence query text must be non-blank")
        return cleaned

    @field_validator("required_terms")
    @classmethod
    def _unique_query_terms(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("evidence query terms must be non-blank")
        if len({value.casefold() for value in cleaned}) != len(cleaned):
            raise ValueError("evidence query terms must be unique")
        return cleaned


class AgentEvidenceRequirement(_StrictModel):
    """One relation required by an answer hypothesis."""

    requirement_id: str = Field(min_length=1)
    relation_to_verify: str = Field(min_length=1)
    evidence_queries: tuple[AgentEvidenceQuery, ...] = Field(
        min_length=1,
        max_length=3,
    )

    @field_validator(
        "requirement_id",
        "relation_to_verify",
    )
    @classmethod
    def _non_blank_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("specification search text must be non-blank")
        return cleaned

    @model_validator(mode="after")
    def _unique_evidence_queries(self) -> "AgentEvidenceRequirement":
        keys = {
            (
                query.specification.casefold(),
                query.lexical_query.casefold(),
                query.semantic_query.casefold(),
            )
            for query in self.evidence_queries
        }
        if len(keys) != len(self.evidence_queries):
            raise ValueError("evidence queries must be unique")
        return self


class AgentAnswerHypothesis(_StrictModel):
    hypothesis_id: str = Field(min_length=1)
    possible_answer: str = Field(min_length=1)
    evidence_goal: str = Field(min_length=1)
    scope: HypothesisScope
    search_terms: tuple[str, ...] = Field(min_length=1, max_length=8)
    evidence_requirements: tuple[AgentEvidenceRequirement, ...] = Field(
        min_length=1,
        max_length=4,
    )

    @field_validator("search_terms")
    @classmethod
    def _unique_non_blank(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("hypothesis list values must be non-blank")
        if len({value.casefold() for value in cleaned}) != len(cleaned):
            raise ValueError("hypothesis list values must be unique")
        return cleaned

    @field_validator("evidence_requirements")
    @classmethod
    def _unique_evidence_requirements(
        cls,
        values: tuple[AgentEvidenceRequirement, ...],
    ) -> tuple[AgentEvidenceRequirement, ...]:
        identifiers = tuple(value.requirement_id.casefold() for value in values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("evidence requirement IDs must be unique")
        return values

    @property
    def target_specifications(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            query.specification
            for requirement in self.evidence_requirements
            for query in requirement.evidence_queries
        ))


class AgentQuestionHypothesisProposal(_StrictModel):
    query_id: str = Field(min_length=1)
    hypotheses: tuple[AgentAnswerHypothesis, ...] = Field(max_length=6)


class QuestionHypothesisPlan(_StrictModel):
    query_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    angle: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    status: Literal["planned", "no_expansion", "unavailable"]
    hypotheses: tuple[AgentAnswerHypothesis, ...] = ()
    warnings: tuple[str, ...] = ()


class AnswerHypothesisPlan(_StrictModel):
    schema_version: Literal["answer-hypothesis-plan/v1.3"] = (
        "answer-hypothesis-plan/v1.3"
    )
    release: int = Field(ge=1)
    agent_receipts: tuple[CodexAgentReceipt, ...] = ()
    questions: tuple[QuestionHypothesisPlan, ...]
    errors: tuple[str, ...] = ()


_PLANNING_INSTRUCTIONS = """
Plan a small parallel retrieval channel for one 3GPP Discovery question.

The supplied original-question retrieval already searches literal entities and
facets. Propose only additional, useful answer hypotheses that may bridge an
unstated protocol term, adjacent protocol layer, procedure, security context,
or state context. A hypothesis is a search hint, never established evidence.

Rules:
1. Preserve the supplied Discovery angle and claim meaning.
2. Return zero to six independent hypotheses. Return zero when literal
   retrieval should be sufficient.
3. Use general 3GPP knowledge only to suggest what might be true and where it
   might be specified. Do not assert attack feasibility or vulnerability.
   For every state or action relation, explicitly identify the actor or
   component. Keep relations belonging to different actors or components
   separate; do not combine one actor's event with another actor's resulting
   state. For state or timing questions, do not infer order from message names.
   Tie each proposed transition to its trigger; if uncertain, make the trigger
   or order an evidence requirement rather than asserting it.
4. Match each hypothesis to the answer unit requested by the question. Do not
   combine alternative answers with "or". Different concrete messages or
   answer objects remain separate. For a direct-carrier question, one
   hypothesis is one immediate carrier relation. For a reachability or
   transport-path question, one hypothesis is one complete end-to-end path
   alternative; do not return its individual edges as separate hypotheses.
5. For carrier or reachability questions, consider whether an object changes
   names across adjacent layers (for example, an inner message becoming a
   generic payload, container, or PDU). Create separate hypotheses only for
   materially different answer-bearing carrier or path alternatives. Include
   the carried object, endpoints, direction, interface, container, and any
   needed adjacent-layer relation as facets of the relevant hypothesis. Do
   not create a separate hypothesis merely for the carried message, endpoint,
   direction, interface, security state, procedure context, or one incomplete
   edge of a requested path. Respect the procedure context when choosing
   carriers and forwarding
   messages; do not substitute a carrier from another procedure merely because
   it carries the same NAS type. If materially different procedure paths are
   possible, keep them as separate complete hypotheses.
6. Determine the evidence shape from the question. For path, reachability,
   carrier, interface, or direction questions, search both direct evidence
   about the object and relation evidence supporting its transmission; for
   other questions, search only the facts explicitly requested. For format,
   protection, authority, state/timing, relation, required-input, and resource
   questions, propose only the corresponding evidence shape.
7. Give each hypothesis one to four evidence_requirements. Each requirement is
   one necessary relation, not a separate answer. Give each requirement one to
   three evidence_queries; each query targets one available specification and
   one local fact. Use multiple queries whenever the relation spans independently
   searchable local facts, even within one specification or section; otherwise
   use one. For state or timing relations, each local query covers one trigger
   and its direct state or action. Do not combine source state, destination state,
   and a later action unless one passage is expected to state them together.
   Do not duplicate a fact covered by another requirement.
   Within the limits, cover every material relation before ancillary detail.
   When the same object or value crosses messages, interfaces, or layers, query
   its upstream carrier, downstream representation, and defining correspondence
   separately when needed. Do not require the literal word "converted"; field
   definitions, type references, and procedure text may establish continuity.
8. search_terms and evidence requirements should use concise terms likely to
   occur verbatim. required_terms must contain one to three distinctive terms
   that a passage must actually contain to earn that specification's reserved
   candidate. Prefer the concrete message, field, procedure, state, or relation
   name; do not use only a broad object shared by many unrelated passages.
9. Do not duplicate the direct literal lookup unless it is needed to connect
   an adjacent-layer or cross-specification relation.
10. If upstream_discovery_results are supplied, use their evidence-backed
    answers only as terminology and relationship hints for this question.
    They are not evidence for the current question. Generate searches for the
    missing current-question relation instead of merely repeating the upstream
    answer. Before returning, check every upstream answer relevant to the
    current question and do not omit or combine its distinct answer object.
""".strip()


def plan_answer_hypothesis_channel(
    base_plan: DiscoveryRetrievalPlanV3,
    *,
    planning_agent: CodexStructuredAgent,
    release: int,
    available_specifications: tuple[str, ...],
    shared_context: Mapping[str, Any] | None = None,
) -> tuple[AnswerHypothesisPlan, DiscoveryRetrievalPlanV3 | None]:
    """Ask the agent for speculative queries and compile them into RAG probes."""

    if release < 1:
        raise ValueError("release must be positive")
    available = tuple(dict.fromkeys(available_specifications))
    available_set = set(available)
    context = dict(shared_context or {})
    upstream_by_query = context.get("upstream_discovery_results_by_query")
    claims_by_id = {
        str(value.get("claim_id")): value
        for value in context.get("claims", ())
        if isinstance(value, dict) and value.get("claim_id")
    }
    receipts: list[CodexAgentReceipt] = []
    question_plans: list[QuestionHypothesisPlan] = []
    retrievals: list[PlannedDiscoveryRetrieval] = []
    errors: list[str] = []

    for item in base_plan.retrievals:
        payload = {
            "release": release,
            "query_id": item.query_id,
            "claim_id": item.claim_id,
            "angle": item.angle,
            "original_question": item.original_question,
            "literal_anchors": [anchor.canonical for anchor in item.anchors],
            "explicit_facets": [facet.name for facet in item.explicit_facets],
            "claim": claims_by_id.get(item.claim_id),
            "attack_hypothesis": context.get("attack_hypothesis"),
            "threat_model": context.get("threat_model"),
            "upstream_discovery_results": (
                upstream_by_query.get(item.query_id, [])
                if isinstance(upstream_by_query, dict)
                else context.get("upstream_discovery_results", [])
            ),
            "available_specifications": list(available),
        }
        try:
            response = planning_agent.parse(
                purpose="plan answer-hypothesis retrieval channel",
                prompt_version="answer-hypothesis-retrieval/v1.17",
                instructions=_PLANNING_INSTRUCTIONS,
                input_payload=payload,
                output_type=AgentQuestionHypothesisProposal,
            )
        except Exception as exc:
            error = f"{item.query_id}: {type(exc).__name__}: {exc}"
            errors.append(error)
            question_plans.append(QuestionHypothesisPlan(
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
            question_plans.append(QuestionHypothesisPlan(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                warnings=(error,),
            ))
            continue

        accepted: list[AgentAnswerHypothesis] = []
        warnings: list[str] = []
        seen: set[tuple[str, str]] = set()
        for proposed in response.output.hypotheses[:6]:
            valid_requirements: list[AgentEvidenceRequirement] = []
            unknown_specs: list[str] = []
            for requirement in proposed.evidence_requirements:
                valid_queries = tuple(
                    query
                    for query in requirement.evidence_queries
                    if query.specification in available_set
                )
                unknown_specs.extend(
                    query.specification
                    for query in requirement.evidence_queries
                    if query.specification not in available_set
                )
                if valid_queries:
                    valid_requirements.append(requirement.model_copy(update={
                        "evidence_queries": valid_queries,
                    }))
            if unknown_specs:
                warnings.append(
                    "ignored unavailable evidence-query specifications for "
                    f"{proposed.hypothesis_id}: "
                    f"{', '.join(dict.fromkeys(unknown_specs))}"
                )
            if not valid_requirements:
                warnings.append(
                    f"skipped {proposed.hypothesis_id}: no evidence query "
                    "targets a specification in the selected Release corpus"
                )
                continue
            key = (
                proposed.possible_answer.casefold(),
                proposed.evidence_goal.casefold(),
            )
            if key in seen:
                warnings.append(
                    f"skipped duplicate hypothesis {proposed.hypothesis_id}"
                )
                continue
            seen.add(key)
            normalized_requirements = tuple(
                requirement.model_copy(update={
                    "requirement_id": f"R{index}",
                })
                for index, requirement in enumerate(valid_requirements, 1)
            )
            accepted.append(proposed.model_copy(update={
                "hypothesis_id": f"H{len(accepted) + 1}",
                "evidence_requirements": normalized_requirements,
            }))

        question_plans.append(QuestionHypothesisPlan(
            query_id=item.query_id,
            claim_id=item.claim_id,
            angle=item.angle,
            original_question=item.original_question,
            status="planned" if accepted else "no_expansion",
            hypotheses=tuple(accepted),
            warnings=tuple(warnings),
        ))
        if not accepted:
            continue

        probes: list[RetrievalProbe] = []
        for hypothesis in accepted:
            for requirement in hypothesis.evidence_requirements:
                for query_index, query in enumerate(
                    requirement.evidence_queries,
                    1,
                ):
                    route = SpecificationRoute(
                        specification=query.specification,
                        weight=1.0,
                        reason=(
                            "specification-local evidence query proposed by "
                            "the answer-hypothesis agent for "
                            f"{hypothesis.hypothesis_id}"
                        ),
                        source="agent_hypothesis",
                    )
                    probes.append(RetrievalProbe(
                        probe_id=(
                            f"{hypothesis.hypothesis_id}-"
                            f"{requirement.requirement_id}-Q{query_index}"
                        ),
                        evidence_role=item.angle,
                        lexical_query=query.lexical_query,
                        semantic_query=query.semantic_query,
                        routes=(route,),
                        # Preserve the agent's complete evidence requirement.
                        # A term may be broad in the original question yet
                        # still be essential to distinguish the relation.
                        anchors=query.required_terms,
                        facets=(hypothesis.scope,),
                        stage=1,
                    ))
        retrievals.append(PlannedDiscoveryRetrieval(
            retrieval_id=f"HB-{item.query_id}",
            query_id=item.query_id,
            claim_id=item.claim_id,
            angle=item.angle,
            original_question=item.original_question,
            planner_action="planned",
            anchors=item.anchors,
            explicit_facets=item.explicit_facets,
            context_dependent=item.context_dependent,
            warnings=tuple((
                *item.warnings,
                "agent hypotheses are retrieval hints and require corpus evidence",
            )),
            reference_expansion=ReferenceExpansionPolicy(enabled=False),
            retrieval_probes=tuple(probes),
        ))

    hypothesis_plan = AnswerHypothesisPlan(
        release=release,
        agent_receipts=tuple(receipts),
        questions=tuple(question_plans),
        errors=tuple(errors),
    )
    retrieval_plan = (
        DiscoveryRetrievalPlanV3(
            anchor_extraction_method=base_plan.anchor_extraction_method,
            retrievals=tuple(retrievals),
        )
        if retrievals
        else None
    )
    return hypothesis_plan, retrieval_plan


def _hypothesis_id_from_probe(probe_id: str) -> str | None:
    matched = re.match(
        r"^(H[1-9][0-9]*)-(?:R|P)[1-9][0-9]*(?:-Q[1-9][0-9]*)?$",
        probe_id,
    )
    return matched.group(1) if matched is not None else None


def _requirement_key_from_probe(probe_id: str) -> tuple[str, str] | None:
    """Return a stable hypothesis/requirement key, including legacy probes."""

    matched = re.match(
        r"^(H[1-9][0-9]*)-([RP][1-9][0-9]*)(?:-Q[1-9][0-9]*)?$",
        probe_id,
    )
    if matched is None:
        return None
    return matched.group(1), matched.group(2)


def _subquery_key_from_probe(
    probe_id: str,
) -> tuple[str, str, str] | None:
    """Return a stable hypothesis/requirement/subquery key."""

    matched = re.match(
        r"^(H[1-9][0-9]*)-(R[1-9][0-9]*)-(Q[1-9][0-9]*)$",
        probe_id,
    )
    if matched is None:
        return None
    return matched.group(1), matched.group(2), matched.group(3)


def _requirement_keys_from_hit(
    hit: PlannedRetrievalHit,
) -> tuple[tuple[str, str], ...]:
    keys = {
        key
        for probe_id in hit.probe_ids
        if (key := _requirement_key_from_probe(probe_id)) is not None
    }
    return tuple(sorted(
        keys,
        key=lambda value: (int(value[0][1:]), value[1][0], int(value[1][1:])),
    ))


def _subquery_keys_from_hit(
    hit: PlannedRetrievalHit,
) -> tuple[tuple[str, str, str], ...]:
    keys = {
        key
        for probe_id in hit.probe_ids
        if (key := _subquery_key_from_probe(probe_id)) is not None
    }
    return tuple(sorted(
        keys,
        key=lambda value: (
            int(value[0][1:]),
            int(value[1][1:]),
            int(value[2][1:]),
        ),
    ))


def _merge_hypothesis_hits(
    existing: PlannedRetrievalHit,
    incoming: PlannedRetrievalHit,
) -> PlannedRetrievalHit:
    """Combine provenance when independent hypotheses retrieve one chunk."""

    best = incoming if incoming.final_score > existing.final_score else existing
    return best.model_copy(update={
        "evidence_roles": tuple(dict.fromkeys((
            *existing.evidence_roles,
            *incoming.evidence_roles,
        ))),
        "probe_ids": tuple(dict.fromkeys((
            *existing.probe_ids,
            *incoming.probe_ids,
        ))),
        "matched_anchors": tuple(dict.fromkeys((
            *existing.matched_anchors,
            *incoming.matched_anchors,
        ))),
        "reference_source_chunk_ids": tuple(dict.fromkeys((
            *existing.reference_source_chunk_ids,
            *incoming.reference_source_chunk_ids,
        ))),
        "final_score": max(existing.final_score, incoming.final_score),
    })


def _is_reservable_hypothesis_hit(hit: PlannedRetrievalHit) -> bool:
    """Return whether a hit has enough query evidence to earn a reservation.

    A specification is never represented merely because the agent named it.
    Reservations require a matched specification-scoped required term. A BM25
    rank alone is deliberately insufficient because it can be earned by one
    generic token from a longer query. Other hits remain eligible for ordinary
    score-based filling and final semantic review.
    """

    return bool(hit.matched_anchors)


def _best_reservable_hypothesis_hit(
    hits: tuple[PlannedRetrievalHit, ...] | list[PlannedRetrievalHit],
) -> PlannedRetrievalHit | None:
    qualified = tuple(hit for hit in hits if _is_reservable_hypothesis_hit(hit))
    if not qualified:
        return None
    return min(
        qualified,
        key=lambda hit: (
            -len(hit.matched_anchors),
            hit.lexical_rank if hit.lexical_rank is not None else 1_000_000,
            hit.semantic_rank if hit.semantic_rank is not None else 1_000_000,
            -hit.final_score,
            hit.chunk_id,
        ),
    )


def _select_balanced_retrieval_hits(
    hits: tuple[PlannedRetrievalHit, ...],
    *,
    required_terms: tuple[str, ...],
    limit: int,
) -> tuple[PlannedRetrievalHit, ...]:
    """Reduce a larger recall pool without losing lexical/semantic diversity."""

    if limit < 1:
        raise ValueError("limit must be positive")
    if len(hits) <= limit:
        return hits

    selected: list[PlannedRetrievalHit] = []
    selected_ids: set[str] = set()

    def add(hit: PlannedRetrievalHit | None) -> None:
        if (
            hit is None
            or len(selected) >= limit
            or hit.chunk_id in selected_ids
        ):
            return
        selected.append(hit)
        selected_ids.add(hit.chunk_id)

    indexed = tuple(enumerate(hits))
    for term in dict.fromkeys(
        re.sub(r"\s+", " ", value).strip().casefold()
        for value in required_terms
    ):
        add(next((
            hit
            for _, hit in indexed
            if term in {
                re.sub(r"\s+", " ", anchor).strip().casefold()
                for anchor in hit.matched_anchors
            }
        ), None))

    add(min(
        (
            (hit.semantic_rank, index, hit)
            for index, hit in indexed
            if hit.chunk_id not in selected_ids
            and hit.semantic_rank is not None
        ),
        default=(0, 0, None),
        key=lambda value: (value[0], value[1]),
    )[2])
    add(min(
        (
            (hit.lexical_rank, index, hit)
            for index, hit in indexed
            if hit.chunk_id not in selected_ids
            and hit.lexical_rank is not None
        ),
        default=(0, 0, None),
        key=lambda value: (value[0], value[1]),
    )[2])

    represented_sections = {
        (hit.source_path, hit.heading_path) for hit in selected
    }
    for _, hit in indexed:
        if len(selected) >= limit:
            break
        section = (hit.source_path, hit.heading_path)
        if section in represented_sections:
            continue
        add(hit)
        represented_sections.add(section)
    for _, hit in indexed:
        add(hit)
    return tuple(selected)


def _evidence_shape(hit: PlannedRetrievalHit) -> str:
    """Classify broad document structure for deterministic diversity."""

    text = " ".join((*hit.heading_path, hit.excerpt[:1200])).casefold()
    if any(marker in text for marker in (
        "information element definition",
        "ie/group name",
        "semantics description",
        "field description",
        "asn.1",
        "asn1start",
        "message definition",
    )):
        return "definition"
    if any(marker in text for marker in (
        "procedure",
        "successful operation",
        "unsuccessful operation",
        " shall ",
    )):
        return "procedure"
    return "context"


def _normalized_excerpt_tokens(hit: PlannedRetrievalHit) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9_-]+", hit.excerpt.casefold()))


def _near_duplicate(
    candidate: PlannedRetrievalHit,
    selected: PlannedRetrievalHit,
) -> bool:
    left = _normalized_excerpt_tokens(candidate)
    right = _normalized_excerpt_tokens(selected)
    if not left or not right:
        return False
    return len(left & right) / len(left | right) >= 0.9


def _adds_complementary_information(
    candidate: PlannedRetrievalHit,
    selected: tuple[PlannedRetrievalHit, ...],
) -> bool:
    """Keep a third hit only when it adds auditable structural information."""

    if not selected or any(
        _near_duplicate(candidate, existing) for existing in selected
    ):
        return not selected
    covered_terms = {
        anchor.casefold()
        for existing in selected
        for anchor in existing.matched_anchors
    }
    if any(
        anchor.casefold() not in covered_terms
        for anchor in candidate.matched_anchors
    ):
        return True
    if _evidence_shape(candidate) not in {
        _evidence_shape(existing) for existing in selected
    }:
        return True
    candidate_section = (candidate.source_path, candidate.heading_path)
    if candidate_section not in {
        (existing.source_path, existing.heading_path) for existing in selected
    }:
        return True
    return any(
        candidate.source_path == existing.source_path
        and candidate.heading_path == existing.heading_path
        and (
            abs(candidate.line_start - existing.line_end) <= 2
            or abs(existing.line_start - candidate.line_end) <= 2
        )
        for existing in selected
    )


def execute_answer_hypothesis_retrieval(
    plan: DiscoveryRetrievalPlanV3,
    *,
    app_config: Any,
    retriever: Any,
    hits_per_hypothesis: int = 3,
    max_hits_per_question: int = 16,
    reranker: Any | None = None,
) -> PlannedRetrievalExecution:
    """Retrieve every hypothesis evidence requirement independently.

    ``hits_per_hypothesis`` is retained as a public compatibility name, but it
    now limits each evidence requirement. Every requirement receives a recall
    pool of at least twelve hits before local balancing.
    """

    if hits_per_hypothesis < 1:
        raise ValueError("hits_per_hypothesis must be positive")
    if max_hits_per_question < 1:
        raise ValueError("max_hits_per_question must be positive")

    tasks: list[PlannedRetrievalTaskResult] = []
    executions: list[PlannedRetrievalExecution] = []
    all_errors: list[str] = []

    for item in plan.retrievals:
        requirement_tasks: list[
            tuple[str, str, str, PlannedRetrievalTaskResult]
        ] = []
        for probe in item.retrieval_probes:
            requirement_key = _requirement_key_from_probe(probe.probe_id)
            if requirement_key is None:
                continue
            hypothesis_id, requirement_id = requirement_key
            for route in probe.routes:
                specification = route.specification
                scoped_probe = probe.model_copy(update={"routes": (route,)})
                isolated_item = item.model_copy(update={
                    "retrieval_id": (
                        f"{item.retrieval_id}-{hypothesis_id}-"
                        f"{requirement_id}-{specification}"
                    ),
                    "retrieval_probes": (scoped_probe,),
                })
                isolated_plan = plan.model_copy(update={
                    "retrievals": (isolated_item,),
                })
                execution = execute_planned_discovery_retrieval(
                    isolated_plan,
                    app_config=app_config,
                    final_hits=max(12, hits_per_hypothesis * 3),
                    retriever=retriever,
                    ranking_mode=(
                        "rrf_anchor_single_if_full" if reranker is not None
                        else "full"
                    ),
                    single_channel_weight=0.6,
                    reranker=reranker,
                )
                executions.append(execution)
                all_errors.extend(execution.errors)
                if execution.tasks:
                    task = execution.tasks[0]
                    task = task.model_copy(update={
                        "hits": tuple(
                            hit.model_copy(update={
                                "probe_ids": tuple(dict.fromkeys((
                                    *hit.probe_ids,
                                    scoped_probe.probe_id,
                                ))),
                            })
                            for hit in _select_balanced_retrieval_hits(
                                task.hits,
                                required_terms=scoped_probe.anchors,
                                limit=hits_per_hypothesis,
                            )
                        ),
                    })
                    requirement_tasks.append((
                        hypothesis_id,
                        requirement_id,
                        specification,
                        task,
                    ))

        selected_by_chunk: dict[str, PlannedRetrievalHit] = {}
        selected_order: list[str] = []

        def add_hit(hit: PlannedRetrievalHit) -> None:
            existing = selected_by_chunk.get(hit.chunk_id)
            if existing is not None:
                selected_by_chunk[hit.chunk_id] = _merge_hypothesis_hits(
                    existing,
                    hit,
                )
                return
            selected_by_chunk[hit.chunk_id] = hit
            selected_order.append(hit.chunk_id)

        # This is an intermediate local pool, not the final LLM payload. Keep
        # all locally selected requirement candidates so the 24-item merge can
        # make the actual quota decision without silently losing a requirement.
        effective_limit = max(
            max_hits_per_question,
            len(requirement_tasks) * hits_per_hypothesis,
        )
        max_depth = max(
            (len(task.hits) for _, _, _, task in requirement_tasks),
            default=0,
        )
        for depth in range(max_depth):
            for _, _, _, task in requirement_tasks:
                if depth >= len(task.hits):
                    continue
                hit = task.hits[depth]
                if hit.chunk_id in selected_by_chunk:
                    add_hit(hit)
                    continue
                if len(selected_order) >= effective_limit:
                    continue
                add_hit(hit)

        hits = tuple(
            selected_by_chunk[chunk_id].model_copy(update={
                "hit_id": f"{item.retrieval_id}:3gpp_markdown:{rank}",
                "collection_rank": rank,
            })
            for rank, chunk_id in enumerate(selected_order, 1)
        )
        task_errors = tuple(
            error
            for _, _, _, task in requirement_tasks
            for error in task.errors
        )
        unavailable_specs = tuple(dict.fromkeys(
            specification
            for _, _, _, task in requirement_tasks
            for specification in task.unavailable_specifications
        ))
        followed_references = tuple(dict.fromkeys(
            reference
            for _, _, _, task in requirement_tasks
            for reference in task.followed_references
        ))
        reranker_statuses = tuple(
            task.reranker_status for _, _, _, task in requirement_tasks
        )
        reranker_errors = tuple(dict.fromkeys(
            task.reranker_error
            for _, _, _, task in requirement_tasks
            if task.reranker_error
        ))
        status: Literal["completed", "partial", "no_hits", "unavailable"] = (
            "partial"
            if hits and task_errors
            else "completed"
            if hits
            else "unavailable"
            if requirement_tasks and all(
                task.status == "unavailable"
                for _, _, _, task in requirement_tasks
            )
            else "no_hits"
        )
        tasks.append(PlannedRetrievalTaskResult(
            retrieval_id=item.retrieval_id,
            query_id=item.query_id,
            planner_action=item.planner_action,
            status=status,
            unavailable_specifications=unavailable_specs,
            followed_references=followed_references,
            reranker_status=(
                "fallback"
                if "fallback" in reranker_statuses
                else "applied"
                if "applied" in reranker_statuses
                else "disabled"
            ),
            reranker_error="; ".join(reranker_errors) or None,
            hits=hits,
            errors=task_errors,
        ))

    exemplar = executions[0] if executions else None
    status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if tasks and all(task.status == "unavailable" for task in tasks)
        else "partial"
        if all_errors or any(
            task.status in {"partial", "unavailable"} for task in tasks
        )
        else "completed"
    )
    return PlannedRetrievalExecution(
        status=status,
        resolved_plan=plan,
        index_fingerprint=(exemplar.index_fingerprint if exemplar else None),
        index_manifest_path=(exemplar.index_manifest_path if exemplar else None),
        embedding_model=(
            exemplar.embedding_model
            if exemplar is not None
            else app_config.retrieval.embedding_model
        ),
        reranker=(exemplar.reranker if exemplar is not None else None),
        corpus_version_policy=(
            exemplar.corpus_version_policy
            if exemplar is not None
            else app_config.retrieval.version_policy
        ),
        corpus_release=(
            exemplar.corpus_release
            if exemplar is not None
            else app_config.retrieval.target_release
        ),
        tasks=tuple(tasks),
        errors=tuple(all_errors),
    )


def merge_parallel_retrieval_channels(
    literal: PlannedRetrievalExecution,
    hypothesis: PlannedRetrievalExecution | None,
    *,
    retriever: Any | None = None,
    route_a_quota: int = 8,
    route_b_quota: int = 16,
    minimum_per_requirement: int = 2,
    max_hits_per_question: int = 24,
    adjacent_chunk_budget: int = 8,
) -> PlannedRetrievalExecution:
    """Deduplicate Route A/B and add bounded same-section source neighbours.

    Adjacent expansion is deliberately deterministic and corpus-grounded. It
    does not ask the model to invent a query or decide whether a neighbouring
    passage is relevant. A hit contributes the first chunk of its exact
    specification subsection plus immediate same-subsection neighbours. The
    expanded passages are considered after two requirement-coverage rounds and
    before optional third passages. They are marked through
    ``retrieval_sources`` so downstream selection can distinguish them from
    ordinary ranked hits.
    """

    if route_a_quota < 0 or route_b_quota < 0:
        raise ValueError("route quotas cannot be negative")
    if minimum_per_requirement < 0:
        raise ValueError("minimum_per_requirement cannot be negative")
    if max_hits_per_question < 1 or route_a_quota + route_b_quota < 1:
        raise ValueError("at least one merged candidate must be allowed")
    if adjacent_chunk_budget < 0:
        raise ValueError("adjacent_chunk_budget cannot be negative")

    def hypothesis_ids(hit: Any) -> tuple[str, ...]:
        values = {
            key[0]
            for probe_id in hit.probe_ids
            if (key := _requirement_key_from_probe(probe_id)) is not None
        }
        return tuple(sorted(values, key=lambda value: int(value[1:])))

    def requirement_keys(hit: Any) -> tuple[tuple[str, str], ...]:
        return _requirement_keys_from_hit(hit)

    def route_b_sources(hit: Any) -> tuple[str, ...]:
        identifiers = hypothesis_ids(hit)
        return tuple(
            f"route_b:{identifier}" for identifier in identifiers
        ) or ("route_b",)

    def add_or_annotate(
        selected: list[Any],
        selected_indexes: dict[str, int],
        hit: Any,
        sources: tuple[str, ...],
    ) -> bool:
        existing_index = selected_indexes.get(hit.chunk_id)
        if existing_index is not None:
            existing = selected[existing_index]
            selected[existing_index] = existing.model_copy(update={
                "retrieval_sources": tuple(dict.fromkeys((
                    *existing.retrieval_sources,
                    *sources,
                ))),
            })
            return False
        if len(selected) >= max_hits_per_question:
            return False
        selected_indexes[hit.chunk_id] = len(selected)
        selected.append(hit.model_copy(update={
            "retrieval_sources": sources,
        }))
        return True

    def adjacent_candidates(
        seeds: list[Any],
        source_label_for_seed: Any,
    ) -> list[Any]:
        if retriever is None or adjacent_chunk_budget <= 0:
            return []
        chunks_by_source: dict[str, list[Any]] = {}
        for chunk in retriever.chunks:
            if chunk.source_collection != "3gpp_markdown":
                continue
            chunks_by_source.setdefault(chunk.source_path, []).append(chunk)
        for chunks in chunks_by_source.values():
            chunks.sort(key=lambda value: (
                value.line_start,
                value.line_end,
                value.chunk_id,
            ))

        seed_ids = {hit.chunk_id for hit in seeds}
        by_chunk: dict[str, Any] = {}
        priority: dict[str, tuple[int, int, int, str]] = {}
        for seed_order, seed in enumerate(seeds):
            siblings = chunks_by_source.get(seed.source_path, [])
            indexes = {
                chunk.chunk_id: index
                for index, chunk in enumerate(siblings)
            }
            seed_index = indexes.get(seed.chunk_id)
            if seed_index is None:
                continue
            same_section = [
                index
                for index, chunk in enumerate(siblings)
                if chunk.heading_path == seed.heading_path
                and chunk.specification == seed.specification
                and chunk.version == seed.version
            ]
            if not same_section:
                continue
            same_section_set = set(same_section)
            candidate_indexes: list[tuple[int, int, int]] = []
            # The first chunk is important for list-style normative sections:
            # a search hit may land on a later item while the opening items
            # contain other independent conditions. Immediate context is more
            # likely to complete a split sentence or relation, so consider it
            # before the section start.
            section_start = same_section[0]
            for distance in range(1, 2):
                for relation_priority, index in enumerate((
                    seed_index - distance,
                    seed_index + distance,
                )):
                    if index in same_section_set:
                        candidate_indexes.append((
                            index,
                            relation_priority,
                            distance,
                        ))
            if section_start != seed_index:
                candidate_indexes.append((section_start, 2, 0))
            for index, relation_priority, distance in candidate_indexes:
                neighbor = siblings[index]
                if neighbor.chunk_id in seed_ids:
                    continue
                neighbor_text = re.sub(
                    r"\s+",
                    " ",
                    neighbor.text,
                ).casefold()
                neighbor_anchors = tuple(
                    anchor
                    for anchor in seed.matched_anchors
                    if re.sub(r"\s+", " ", anchor).casefold()
                    in neighbor_text
                )
                expanded = seed.model_copy(update={
                    "hit_id": f"{seed.hit_id}:adjacent:{neighbor.chunk_id}",
                    "chunk_id": neighbor.chunk_id,
                    "collection_rank": 1,
                    "source_path": neighbor.source_path,
                    "document_sha256": neighbor.document_sha256,
                    "heading_path": neighbor.heading_path,
                    "line_start": neighbor.line_start,
                    "line_end": neighbor.line_end,
                    "excerpt": neighbor.text,
                    "matched_anchors": neighbor_anchors,
                    "retrieval_sources": tuple(dict.fromkeys((
                        *seed.retrieval_sources,
                        source_label_for_seed(seed),
                        "adjacent_chunk",
                    ))),
                    "retrieval_hop": 2,
                    "semantic_rank": None,
                    "lexical_rank": None,
                    "rrf_score": 0.0,
                    "route_score": 0.0,
                    "anchor_score": 0.0,
                    "heading_score": 0.0,
                    "agreement_score": 0.0,
                    "noise_penalty": 0.0,
                    "final_score": 0.0,
                })
                existing = by_chunk.get(neighbor.chunk_id)
                if existing is None:
                    by_chunk[neighbor.chunk_id] = expanded
                else:
                    by_chunk[neighbor.chunk_id] = _merge_hypothesis_hits(
                        existing,
                        expanded,
                    ).model_copy(update={
                        "retrieval_sources": tuple(dict.fromkeys((
                            *existing.retrieval_sources,
                            *expanded.retrieval_sources,
                        ))),
                    })
                # Prefer immediate predecessors, then immediate successors,
                # then section starts; retain seed order as the tie-breaker.
                priority[neighbor.chunk_id] = min(
                    priority.get(
                        neighbor.chunk_id,
                        (
                            relation_priority,
                            seed_order,
                            distance,
                            neighbor.chunk_id,
                        ),
                    ),
                    (
                        relation_priority,
                        seed_order,
                        distance,
                        neighbor.chunk_id,
                    ),
                )

        return [
            by_chunk[chunk_id]
            for chunk_id in sorted(
                by_chunk,
                key=lambda chunk_id: priority[chunk_id],
            )[:adjacent_chunk_budget]
        ]

    hypothesis_by_query = {
        task.query_id: task for task in hypothesis.tasks
    } if hypothesis is not None else {}
    tasks: list[PlannedRetrievalTaskResult] = []
    for literal_task in literal.tasks:
        hypothesis_task = hypothesis_by_query.get(literal_task.query_id)
        b_hits = list(hypothesis_task.hits) if hypothesis_task is not None else []
        ordered_subqueries = tuple(dict.fromkeys(
            subquery
            for hit in b_hits
            if _is_reservable_hypothesis_hit(hit)
            for subquery in _subquery_keys_from_hit(hit)
        ))
        # A subquery reservation is more useful than a fixed Route-A/Route-B
        # split when the agent deliberately separated field structure from
        # field semantics. Keep the total bounded, but allow the Route-B side
        # to use the space needed for one qualifying passage per subquery.
        desired_route_b_quota = min(
            max_hits_per_question,
            max(route_b_quota, len(ordered_subqueries)),
        )
        # Keep a small part of the fixed total budget available for source
        # context. Any unused reservation is returned to ordinary Route A hits
        # after focused selection has finished.
        adjacent_reserve = (
            min(2, adjacent_chunk_budget, max_hits_per_question)
            if retriever is not None
            else 0
        )
        effective_route_a_quota = min(
            route_a_quota,
            max(
                0,
                max_hits_per_question
                - desired_route_b_quota
                - adjacent_reserve,
            ),
        )
        selected: list[Any] = []
        selected_indexes: dict[str, int] = {}
        for hit in literal_task.hits[:effective_route_a_quota]:
            add_or_annotate(
                selected,
                selected_indexes,
                hit,
                ("route_a",),
            )

        b_selected_chunks: set[str] = set()
        b_unique_count = 0
        ordered_requirements = tuple(dict.fromkeys(
            requirement
            for hit in b_hits
            if _is_reservable_hypothesis_hit(hit)
            for requirement in requirement_keys(hit)
        ))
        available_b_positions = max(0, max_hits_per_question - len(selected))
        effective_route_b_quota = min(
            desired_route_b_quota,
            available_b_positions,
        )
        selected_for_requirement: dict[
            tuple[str, str], list[PlannedRetrievalHit]
        ] = {key: [] for key in ordered_requirements}
        selected_for_subquery: dict[
            tuple[str, str, str], list[PlannedRetrievalHit]
        ] = {key: [] for key in ordered_subqueries}

        def add_for_requirement(
            key: tuple[str, str],
            hit: PlannedRetrievalHit,
            reason: str,
        ) -> None:
            nonlocal b_unique_count
            if hit.chunk_id not in {
                existing.chunk_id for existing in selected_for_requirement[key]
            }:
                selected_for_requirement[key].append(hit)
            b_selected_chunks.add(hit.chunk_id)
            sources = tuple(dict.fromkeys((
                *route_b_sources(hit),
                f"route_b_requirement:{key[0]}:{key[1]}",
                f"selection:{reason}",
            )))
            if add_or_annotate(selected, selected_indexes, hit, sources):
                b_unique_count += 1

        def add_for_subquery(
            key: tuple[str, str, str],
            hit: PlannedRetrievalHit,
        ) -> None:
            nonlocal b_unique_count
            if hit.chunk_id not in {
                existing.chunk_id for existing in selected_for_subquery[key]
            }:
                selected_for_subquery[key].append(hit)
            b_selected_chunks.add(hit.chunk_id)
            sources = tuple(dict.fromkeys((
                *route_b_sources(hit),
                f"route_b_requirement:{key[0]}:{key[1]}",
                f"route_b_subquery:{key[0]}:{key[1]}:{key[2]}",
                "selection:subquery_base",
            )))
            if add_or_annotate(
                selected,
                selected_indexes,
                hit,
                sources,
            ):
                b_unique_count += 1

        # Preserve one qualifying passage for every distinct specification-
        # local subquery before ordinary requirement and ranking fill.
        for key in ordered_subqueries:
            if b_unique_count >= effective_route_b_quota:
                break
            hit = next((
                candidate
                for candidate in b_hits
                if key in _subquery_keys_from_hit(candidate)
                and _is_reservable_hypothesis_hit(candidate)
                and candidate.chunk_id not in {
                    existing.chunk_id
                    for existing in selected_for_subquery[key]
                }
            ), None)
            if hit is not None:
                add_for_subquery(key, hit)

        # Give each requirement up to two qualifying, distinct base passages
        # in round-robin order before spending capacity on third passages.
        for _depth in range(minimum_per_requirement):
            for key in ordered_requirements:
                if b_unique_count >= effective_route_b_quota:
                    break
                hit = next((
                    candidate
                    for candidate in b_hits
                    if key in requirement_keys(candidate)
                    and _is_reservable_hypothesis_hit(candidate)
                    and candidate.chunk_id not in {
                        existing.chunk_id
                        for existing in selected_for_requirement[key]
                    }
                ), None)
                if hit is None:
                    continue
                add_for_requirement(key, hit, "requirement_base")
            if b_unique_count >= effective_route_b_quota:
                break

        # Expand after required base coverage but before optional third hits.
        # Limit this pass to two new chunks so neighbours cannot consume the
        # entire focused-retrieval budget.
        if retriever is not None and adjacent_chunk_budget:
            # Use the merged seed order so immediate context around strong
            # literal hits competes fairly with focused Route B context.
            expanded = adjacent_candidates(
                list(selected),
                lambda hit: next(
                    (
                        f"adjacent:{source}"
                        for source in hit.retrieval_sources
                        if source.startswith("route_b:")
                    ),
                    "adjacent:route_a",
                ),
            )
            expanded_by_chunk: dict[str, Any] = {}
            for hit in expanded:
                existing = expanded_by_chunk.get(hit.chunk_id)
                if existing is None:
                    expanded_by_chunk[hit.chunk_id] = hit
                else:
                    expanded_by_chunk[hit.chunk_id] = _merge_hypothesis_hits(
                        existing,
                        hit,
                    ).model_copy(update={
                        "retrieval_sources": tuple(dict.fromkeys((
                            *existing.retrieval_sources,
                            *hit.retrieval_sources,
                        ))),
                    })
            adjacent_added = 0
            for hit in expanded_by_chunk.values():
                if (
                    len(selected) >= max_hits_per_question
                    and hit.chunk_id not in selected_indexes
                ):
                    break
                was_added = add_or_annotate(
                    selected,
                    selected_indexes,
                    hit,
                    hit.retrieval_sources or ("adjacent_chunk",),
                )
                if was_added:
                    adjacent_added += 1
                for key in requirement_keys(hit):
                    if key in selected_for_requirement and hit.chunk_id not in {
                        existing.chunk_id
                        for existing in selected_for_requirement[key]
                    }:
                        selected_for_requirement[key].append(hit)
                if adjacent_added >= 2:
                    break

        # A third passage must add terms, a different section/evidence shape,
        # or immediate source context. It is optional and never displaces the
        # two base rounds above.
        for key in ordered_requirements:
            if b_unique_count >= effective_route_b_quota:
                break
            prior = tuple(selected_for_requirement[key])
            if len(prior) >= 3:
                continue
            hit = next((
                candidate
                for candidate in b_hits
                if key in requirement_keys(candidate)
                and _is_reservable_hypothesis_hit(candidate)
                and candidate.chunk_id not in {
                    existing.chunk_id for existing in prior
                }
                and _adds_complementary_information(candidate, prior)
            ), None)
            if hit is not None:
                add_for_requirement(key, hit, "requirement_complement")

        # Fill any remaining Route B capacity in balanced upstream order.
        for hit in b_hits:
            if b_unique_count >= effective_route_b_quota:
                break
            if hit.chunk_id in b_selected_chunks:
                continue
            b_selected_chunks.add(hit.chunk_id)
            if add_or_annotate(
                selected,
                selected_indexes,
                hit,
                route_b_sources(hit),
            ):
                b_unique_count += 1

        # Return unused adjacent capacity to the ordinary literal channel.
        for hit in literal_task.hits[effective_route_a_quota:route_a_quota]:
            if len(selected) >= max_hits_per_question:
                break
            add_or_annotate(
                selected,
                selected_indexes,
                hit,
                ("route_a",),
            )

        selected = [
            hit.model_copy(update={"collection_rank": index})
            for index, hit in enumerate(selected, 1)
        ]
        task_errors = tuple((
            *literal_task.errors,
            *(hypothesis_task.errors if hypothesis_task is not None else ()),
        ))
        status: Literal["completed", "partial", "no_hits", "unavailable"] = (
            "partial"
            if selected and task_errors
            else "completed"
            if selected
            else "unavailable"
            if literal_task.status == "unavailable"
            and (hypothesis_task is None or hypothesis_task.status == "unavailable")
            else "no_hits"
        )
        tasks.append(PlannedRetrievalTaskResult(
            retrieval_id=literal_task.retrieval_id,
            query_id=literal_task.query_id,
            planner_action=literal_task.planner_action,
            status=status,
            unavailable_specifications=tuple(dict.fromkeys((
                *literal_task.unavailable_specifications,
                *(
                    hypothesis_task.unavailable_specifications
                    if hypothesis_task is not None
                    else ()
                ),
            ))),
            followed_references=tuple(dict.fromkeys((
                *literal_task.followed_references,
                *(
                    hypothesis_task.followed_references
                    if hypothesis_task is not None
                    else ()
                ),
            ))),
            hits=tuple(selected),
            errors=task_errors,
        ))

    all_errors = tuple((
        *literal.errors,
        *(hypothesis.errors if hypothesis is not None else ()),
    ))
    status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if tasks and all(task.status == "unavailable" for task in tasks)
        else "partial"
        if all_errors or any(
            task.status in {"partial", "unavailable"} for task in tasks
        )
        else "completed"
    )
    return PlannedRetrievalExecution(
        status=status,
        resolved_plan=literal.resolved_plan,
        index_fingerprint=literal.index_fingerprint,
        index_manifest_path=literal.index_manifest_path,
        embedding_model=literal.embedding_model,
        corpus_version_policy=literal.corpus_version_policy,
        corpus_release=literal.corpus_release,
        tasks=tuple(tasks),
        errors=all_errors,
    )


__all__ = [
    "AgentAnswerHypothesis",
    "AgentEvidenceQuery",
    "AgentEvidenceRequirement",
    "AgentQuestionHypothesisProposal",
    "AnswerHypothesisPlan",
    "QuestionHypothesisPlan",
    "execute_answer_hypothesis_retrieval",
    "merge_parallel_retrieval_channels",
    "plan_answer_hypothesis_channel",
]

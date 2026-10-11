"""Agent-inspired retrieval hypotheses executed as a parallel RAG channel.

The agent is allowed to propose likely terminology, protocol-layer bridges, and
specification families. Those proposals are search hints only. Every retained
result remains an original corpus chunk and downstream selection is forbidden
from treating an unsupported proposal as evidence.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping

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


from .hypothesis_execution import execute_answer_hypothesis_retrieval
from .hypothesis_merge import merge_parallel_retrieval_channels
from .hypothesis_models import (
    AgentAnswerHypothesis,
    AgentEvidenceQuery,
    AgentEvidenceRequirement,
    AgentQuestionHypothesisProposal,
    AnswerHypothesisPlan,
    QuestionHypothesisPlan,
)
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
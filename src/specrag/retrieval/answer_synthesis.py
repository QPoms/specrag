"""Grounded answer synthesis over expanded retrieval evidence."""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping

from ..infrastructure.structured_agent import CodexAgentReceipt, CodexStructuredAgent
from .answer_models import (
    AgentRepairableQuestionAnswerProposal,
    EvidenceAnswerSummary,
    EvidenceBackedAnswer,
    EvidenceRepairQuery,
    QuestionAnswerSummary,
)
from .evidence_selection import resolve_evidence_id_aliases
from .execution_models import (
    ExpandedRetrievalExecution,
    GroundedRelationEvidence,
    PlannedRetrievalHit,
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
                resolve_evidence_id_aliases(
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



__all__ = ['summarize_expanded_discovery_evidence']

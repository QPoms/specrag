"""Grounded second-stage retrieval planning and execution."""

from __future__ import annotations

from typing import Any, Literal

from ..config import AppConfig, load_config
from ..infrastructure.structured_agent import CodexAgentReceipt, CodexStructuredAgent
from .answer_models import EvidenceGroundedExpansionPlan
from .execution import (
    chunk_from_planned_hit as _chunk_from_planned_hit,
    execute_retrieval_task as _execute_task,
    specification_references as _references,
)
from .execution_models import (
    EvidenceExpansionBatch,
    EvidenceGroundedFollowup,
    ExpandedRetrievalExecution,
    GroundedRelationEvidence,
    PlannedRetrievalExecution,
    PlannedRetrievalHit,
    PlannedRetrievalTaskResult,
    QuestionExpansionProposal,
)
from .hybrid import HybridMarkdownRetriever, effective_corpus_scope
from .models import MarkdownChunk
from .plan_models import (
    DiscoveryRetrievalPlanV3,
    EvidenceTermProposal,
    GroundedTerm,
    PlannedDiscoveryRetrieval,
    ReferenceExpansionPolicy,
    RetrievalProbe,
    SpecificationRoute,
)
from .planning import GENERIC_FACET_TERMS
from .relation_evidence import (
    _INTENT_QUERY_PHRASES,
    _build_grounded_relation_evidence,
    _grounded_term_catalog,
)
from .routing import resolve_discovery_retrieval_plan
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
            and anchor.canonical.casefold() not in GENERIC_FACET_TERMS
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
                if term.surface.casefold() in GENERIC_FACET_TERMS:
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
    active_scope = effective_corpus_scope(cfg)
    try:
        active_retriever = retriever or HybridMarkdownRetriever(cfg)
        active_scope = effective_corpus_scope(cfg, active_retriever)
        resolved = resolve_discovery_retrieval_plan(plan, active_retriever)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        unavailable = PlannedRetrievalExecution(
            status="unavailable",
            embedding_model=cfg.embedding_model,
            corpus_version_policy=active_scope.version_policy,
            corpus_release=active_scope.release,
            corpus_versions=active_scope.versions,
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
        corpus_version_policy=active_scope.version_policy,
        corpus_release=active_scope.release,
        corpus_versions=active_scope.versions,
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



__all__ = [
    'execute_expanded_discovery_retrieval',
    '_grounded_followup_plan',
    '_merge_stage_tasks',
    '_second_stage_item',
]

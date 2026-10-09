"""Researcher action constraints and compilation into trusted retrieval plans."""

from __future__ import annotations

from typing import Any

from specrag.retrieval.planner import (
    DiscoveryRetrievalPlanV3,
    PlannedDiscoveryRetrieval,
    ReferenceExpansionPolicy,
    RetrievalProbe,
    SpecificationRoute,
)

from .models import AgentRetrievalAction
from .state import EvidenceLedger


CONTROLLER_INSTRUCTIONS = """\
You control a bounded source-retrieval loop. Your only goal is to collect enough
normative evidence to answer the supplied Discovery questions. You may call:

1. hybrid_search: lexical plus semantic search. Supply a query_id, both query
   strings, optional target specifications, no seed IDs, and a result limit.
2. expand_same_section: retrieve the opening/adjacent chunks of the exact
   subsection containing one to four seed evidence IDs. Supply a query_id and
   seed IDs; leave query strings and target specifications empty.
3. finish: stop retrieval. Use an empty query_id, empty query strings, no target
   specifications, no seed IDs, and max_results=1.

Inspect the evidence before choosing the next action. Search for missing facts
or missing relation edges, not paraphrases of evidence already present. Ordered
questions may form a semantic chain: terms found for an earlier question may be
used to formulate a later search, but they are search hints only. A later answer
still requires its own source evidence. Do not assume a candidate answer is true.
Do not repeat a prior tool call. Use only listed specification numbers. Finish
when the requested answer units are directly supportable or the remaining
budget cannot usefully improve them. Return one to three actions per round; a
finish action must be the only action.
"""


def build_search_plan(
    base_item: PlannedDiscoveryRetrieval,
    action: AgentRetrievalAction,
    *,
    round_number: int,
) -> DiscoveryRetrievalPlanV3:
    """Build a one-probe search with exact-anchor recall explicitly disabled."""

    routes = tuple(
        SpecificationRoute(
            specification=specification,
            weight=1.0,
            reason=f"agent-selected target for {action.action_id}",
            source="agent_hypothesis",
        )
        for specification in action.target_specifications
    )
    probe = RetrievalProbe(
        probe_id=f"AG-R{round_number}-{action.action_id}",
        evidence_role=base_item.angle,
        lexical_query=action.lexical_query.strip(),
        semantic_query=action.semantic_query.strip(),
        routes=routes,
        anchors=(),
        facets=(),
        stage=1,
    )
    item = base_item.model_copy(update={
        "retrieval_id": (
            f"{base_item.query_id}:agent:r{round_number}:{action.action_id}"
        ),
        "planner_action": "planned",
        "anchors": (),
        "warnings": (*base_item.warnings, "agent-controlled hybrid search"),
        "reference_expansion": ReferenceExpansionPolicy(
            enabled=False,
            max_references=0,
        ),
        "retrieval_probes": (probe,),
    })
    return DiscoveryRetrievalPlanV3(
        anchor_extraction_method="deterministic_extractive",
        retrievals=(item,),
    )


def validate_action(
    action: AgentRetrievalAction,
    *,
    query_ids: frozenset[str],
    available_specifications: frozenset[str],
    ledger: EvidenceLedger,
) -> tuple[str, ...]:
    errors: list[str] = []
    if action.tool == "finish":
        if (
            action.query_id
            or action.lexical_query
            or action.semantic_query
            or action.target_specifications
            or action.seed_evidence_ids
            or action.max_results != 1
        ):
            errors.append("finish contains non-empty tool parameters")
        return tuple(errors)
    if action.query_id not in query_ids:
        errors.append(f"unknown query_id: {action.query_id}")
        return tuple(errors)
    unknown_specs = sorted(
        set(action.target_specifications) - available_specifications
    )
    if unknown_specs:
        errors.append("unknown specifications: " + ", ".join(unknown_specs))
    if action.tool == "hybrid_search":
        if not action.lexical_query.strip() or not action.semantic_query.strip():
            errors.append("hybrid_search requires both query strings")
        if action.seed_evidence_ids:
            errors.append("hybrid_search cannot contain seed evidence IDs")
    elif action.tool == "expand_same_section":
        if action.lexical_query or action.semantic_query:
            errors.append("expand_same_section cannot contain query strings")
        if action.target_specifications:
            errors.append("expand_same_section cannot contain target specifications")
        if not action.seed_evidence_ids:
            errors.append("expand_same_section requires seed evidence IDs")
        unknown_ids = [
            evidence_id
            for evidence_id in action.seed_evidence_ids
            if ledger.find(action.query_id, evidence_id) is None
        ]
        if unknown_ids:
            errors.append("unknown seed evidence IDs: " + ", ".join(unknown_ids))
    return tuple(errors)


def action_signature(action: AgentRetrievalAction) -> tuple[Any, ...]:
    return (
        action.tool,
        action.query_id,
        action.lexical_query.casefold().strip(),
        action.semantic_query.casefold().strip(),
        tuple(sorted(action.target_specifications)),
        tuple(sorted(action.seed_evidence_ids)),
    )


__all__ = [
    "CONTROLLER_INSTRUCTIONS",
    "action_signature",
    "build_search_plan",
    "validate_action",
]

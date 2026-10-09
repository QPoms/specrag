"""Resolve candidate 3GPP specifications for an immutable retrieval plan."""

from __future__ import annotations

from .hybrid import HybridMarkdownRetriever
from .plan_models import (
    DiscoveryRetrievalPlanV3,
    PlannedDiscoveryRetrieval,
    RetrievalProbe,
    SpecificationRoute,
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
__all__ = ["resolve_discovery_retrieval_plan"]
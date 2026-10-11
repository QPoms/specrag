"""Planned retrieval execution and evidence-driven reference expansion.

The planner describes what to retrieve. This module performs those searches,
merges their candidate sets, and delegates scoring to the ranking policy.
"""

from __future__ import annotations

import re
from typing import Literal

from .execution_models import PlannedRetrievalHit, PlannedRetrievalTaskResult
from .hybrid import HybridMarkdownRetriever
from .index import searchable_text
from .models import MarkdownChunk
from .normalization import (
    protocol_identifier_in_text,
    protocol_identifier_variants,
)
from .plan_models import (
    PlannedDiscoveryRetrieval,
    RetrievalProbe,
    SpecificationRoute,
)
from .planning import GENERIC_FACET_TERMS
from .ranking import (
    RankingMode,
    RetrievalCandidate,
    rank_retrieval_candidates,
)
from .reranker import CrossEncoderReranker


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


def fold_protocol_text(text: str) -> str:
    return re.sub(r"[\u2010-\u2015]", "-", text).casefold()


def exact_match_priority(chunk: MarkdownChunk, probe: RetrievalProbe) -> float:
    """Rank exact entity occurrences for the probe's evidence shape."""

    searchable = fold_protocol_text(searchable_text(chunk))
    heading = fold_protocol_text(" ".join(chunk.heading_path))
    score = float(sum(
        max(
            (searchable.count(variant)
             for variant in protocol_identifier_variants(anchor)),
            default=0,
        )
        for anchor in probe.anchors
    ))
    if any(protocol_identifier_in_text(anchor, heading) for anchor in probe.anchors):
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
    """Recall exact occurrences across the corpus before route ranking."""

    anchors = tuple(
        anchor
        for anchor in probe.anchors
        if anchor.casefold() not in GENERIC_FACET_TERMS
    )
    if not anchors:
        return []
    scored: list[tuple[str, float]] = []
    for chunk in retriever.chunks:
        if chunk.source_collection != "3gpp_markdown":
            continue
        searchable = fold_protocol_text(searchable_text(chunk))
        if not any(
            protocol_identifier_in_text(anchor, searchable) for anchor in anchors
        ):
            continue
        scored.append((chunk.chunk_id, exact_match_priority(chunk, probe)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]


def _add_exact_results(
    retriever: HybridMarkdownRetriever,
    probe: RetrievalProbe,
    candidates: dict[str, RetrievalCandidate],
    exact: list[tuple[str, float]],
    *,
    hop: Literal[1, 2],
) -> None:
    route_weights = {route.specification: route.weight for route in probe.routes}
    for rank, (chunk_id, _) in enumerate(exact, 1):
        chunk = retriever.chunk_by_id[chunk_id]
        candidate = candidates.setdefault(
            chunk_id,
            RetrievalCandidate(chunk=chunk),
        )
        candidate.retrieval_hop = min(candidate.retrieval_hop, hop)
        candidate.probe_ids.add(probe.probe_id)
        candidate.evidence_roles.add(probe.evidence_role)
        candidate.exact_anchor_rank = min(candidate.exact_anchor_rank or rank, rank)
        candidate.exact_probe_ranks[probe.probe_id] = min(
            candidate.exact_probe_ranks.get(probe.probe_id, rank),
            rank,
        )
        candidate.rrf_score += 1.5 * probe.weight / (retriever.cfg.rrf_k + rank)
        candidate.route_score = max(
            candidate.route_score,
            0.004 * route_weights.get(chunk.specification or "", 0.5),
        )
        searchable = fold_protocol_text(searchable_text(chunk))
        heading = fold_protocol_text(" ".join(chunk.heading_path))
        for anchor in probe.anchors:
            if protocol_identifier_in_text(anchor, searchable):
                candidate.anchors.add(anchor)
            if protocol_identifier_in_text(anchor, heading):
                candidate.heading_anchors.add(anchor)


def _add_search_results(
    retriever: HybridMarkdownRetriever,
    probe: RetrievalProbe,
    candidates: dict[str, RetrievalCandidate],
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
        candidate = candidates.setdefault(
            chunk_id,
            RetrievalCandidate(chunk=chunk),
        )
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


def specification_references(
    chunk: MarkdownChunk,
) -> tuple[tuple[str, str | None, str], ...]:
    """Return unique specification/clause citations in source order."""

    matches: list[tuple[int, int, str, str | None, str]] = []
    references: list[tuple[str, str | None, str]] = []
    seen: set[tuple[str, str | None]] = set()
    searchable = searchable_text(chunk)
    for pattern in _SPEC_REFERENCE_PATTERNS:
        for match in pattern.finditer(searchable):
            matches.append((
                match.start(),
                match.end(),
                match.group("spec"),
                match.groupdict().get("clause"),
                match.group("citation"),
            ))
    for start, end, specification, clause, citation in sorted(matches):
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
            ) in matches
        ):
            continue
        key = (specification, clause)
        if key in seen:
            continue
        seen.add(key)
        references.append((specification, clause, citation))
    return tuple(references)


def chunk_from_planned_hit(hit: PlannedRetrievalHit) -> MarkdownChunk:
    """Restore the immutable source chunk represented by a planned hit."""

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


def _search_probe(
    retriever: HybridMarkdownRetriever,
    probe: RetrievalProbe,
    allowed_specs: tuple[str, ...],
    errors: list[str],
    *,
    seen_semantic_queries: set[tuple[str, tuple[str, ...]]],
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    semantic_key = (
        fold_protocol_text(probe.semantic_query),
        tuple(sorted(allowed_specs)),
    )
    try:
        if semantic_key in seen_semantic_queries:
            dense: list[tuple[str, float]] = []
        else:
            seen_semantic_queries.add(semantic_key)
            dense = retriever._dense(
                probe.semantic_query,
                "3gpp_markdown",
                allowed_specs,
            )
    except Exception as exc:
        dense = []
        errors.append(f"{probe.probe_id}: semantic: {type(exc).__name__}: {exc}")
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
        errors.append(f"{probe.probe_id}: lexical: {type(exc).__name__}: {exc}")
    return dense, lexical


def execute_retrieval_task(
    retriever: HybridMarkdownRetriever,
    item: PlannedDiscoveryRetrieval,
    *,
    final_hits: int,
    retrieval_stage: Literal[1, 2] = 1,
    reserve_probe_candidates: bool = False,
    ranking_mode: RankingMode = "full",
    anchor_weight: float = 0.004,
    single_channel_max_rank: int = 3,
    single_channel_weight: float = 1.0,
    full_coverage_bonus: float = 0.004,
    reranker: CrossEncoderReranker | None = None,
) -> PlannedRetrievalTaskResult:
    """Execute one planned retrieval task without changing its plan."""

    available_specs = {
        document.specification
        for document in retriever.documents
        if document.specification is not None
    }
    candidates: dict[str, RetrievalCandidate] = {}
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
        hypothesis_probe = any(
            route.source == "agent_hypothesis" for route in probe.routes
        )
        if not hypothesis_probe:
            exact = _exact_anchor_results(
                retriever,
                probe,
                limit=max(40, retriever.cfg.lexical_candidates_per_collection * 2),
            )
            _add_exact_results(
                retriever,
                probe,
                candidates,
                exact,
                hop=retrieval_stage,
            )
        dense, lexical = _search_probe(
            retriever,
            probe,
            allowed_specs,
            errors,
            seen_semantic_queries=seen_semantic_queries,
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
            for specification, clause, citation in specification_references(
                source_chunk
            ):
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
                    reference_probe = probe.model_copy(update={
                        "routes": (SpecificationRoute(
                            specification=specification,
                            weight=0.75,
                            reason=f"explicit first-hop citation: {citation}",
                            source="reference",
                        ),),
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
                        reference_source=source_chunk.chunk_id,
                    )

    ranked = rank_retrieval_candidates(
        candidates,
        item,
        rrf_k=retriever.cfg.rrf_k,
        final_hits=final_hits,
        reserve_probe_candidates=reserve_probe_candidates,
        ranking_mode=ranking_mode,
        anchor_weight=anchor_weight,
        single_channel_max_rank=single_channel_max_rank,
        single_channel_weight=single_channel_weight,
        full_coverage_bonus=full_coverage_bonus,
        reranker=reranker,
    )
    if ranked.hits and errors:
        status: Literal["completed", "partial", "no_hits", "unavailable"] = (
            "partial"
        )
    elif ranked.hits:
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
        reranker_status=ranked.reranker_status,
        reranker_error=ranked.reranker_error,
        hits=ranked.hits,
        errors=tuple(errors),
    )


__all__ = [
    "chunk_from_planned_hit",
    "exact_match_priority",
    "execute_retrieval_task",
    "fold_protocol_text",
    "specification_references",
]

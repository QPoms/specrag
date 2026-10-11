"""Execution policy for the speculative answer-hypothesis channel."""

from __future__ import annotations

import re
from typing import Any, Literal

from .planner import (
    DiscoveryRetrievalPlanV3,
    PlannedRetrievalExecution,
    PlannedRetrievalHit,
    PlannedRetrievalTaskResult,
    execute_planned_discovery_retrieval,
)
from .hybrid import effective_corpus_scope
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

    active_scope = effective_corpus_scope(app_config.retrieval, retriever)
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
            else active_scope.version_policy
        ),
        corpus_release=(
            exemplar.corpus_release
            if exemplar is not None
            else active_scope.release
        ),
        corpus_versions=(
            exemplar.corpus_versions
            if exemplar is not None
            else active_scope.versions
        ),
        tasks=tuple(tasks),
        errors=tuple(all_errors),
    )



__all__ = ['execute_answer_hypothesis_retrieval']

"""Deterministic merge policy for literal and hypothesis retrieval channels."""

from __future__ import annotations

import re
from typing import Any, Literal

from .hypothesis_execution import (
    _adds_complementary_information,
    _is_reservable_hypothesis_hit,
    _merge_hypothesis_hits,
    _requirement_key_from_probe,
    _requirement_keys_from_hit,
    _subquery_keys_from_hit,
)
from .planner import (
    PlannedRetrievalExecution,
    PlannedRetrievalHit,
    PlannedRetrievalTaskResult,
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
        corpus_versions=literal.corpus_versions,
        tasks=tuple(tasks),
        errors=all_errors,
    )



__all__ = ['merge_parallel_retrieval_channels']

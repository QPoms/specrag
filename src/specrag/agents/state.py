"""Evidence state accumulated by a bounded Researcher loop."""

from __future__ import annotations

from typing import Any

from specrag.retrieval.planner import PlannedRetrievalHit


def _minimum_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


class EvidenceLedger:
    """Deduplicated, per-question evidence with stable public identifiers."""

    def __init__(
        self,
        query_ids: tuple[str, ...],
        *,
        max_per_question: int,
    ) -> None:
        self._hits = {query_id: [] for query_id in query_ids}
        self._indexes = {query_id: {} for query_id in query_ids}
        self._next = {query_id: 1 for query_id in query_ids}
        self._max_per_question = max_per_question

    def hits(self, query_id: str) -> tuple[PlannedRetrievalHit, ...]:
        return tuple(self._hits[query_id])

    def find(self, query_id: str, evidence_id: str) -> PlannedRetrievalHit | None:
        for hit in self._hits[query_id]:
            if hit.hit_id == evidence_id:
                return hit
        return None

    def add(
        self,
        query_id: str,
        hits: tuple[PlannedRetrievalHit, ...],
        *,
        action_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        observed: list[str] = []
        added: list[str] = []
        for hit in hits:
            existing_index = self._indexes[query_id].get(hit.chunk_id)
            source = f"agent_tool:{action_id}"
            if existing_index is not None:
                existing = self._hits[query_id][existing_index]
                merged = existing.model_copy(update={
                    "retrieval_sources": tuple(dict.fromkeys((
                        *existing.retrieval_sources,
                        *hit.retrieval_sources,
                        source,
                    ))),
                    "probe_ids": tuple(dict.fromkeys((
                        *existing.probe_ids,
                        *hit.probe_ids,
                    ))),
                    "matched_anchors": tuple(dict.fromkeys((
                        *existing.matched_anchors,
                        *hit.matched_anchors,
                    ))),
                    "reference_source_chunk_ids": tuple(dict.fromkeys((
                        *existing.reference_source_chunk_ids,
                        *hit.reference_source_chunk_ids,
                    ))),
                    "semantic_rank": _minimum_optional(
                        existing.semantic_rank,
                        hit.semantic_rank,
                    ),
                    "lexical_rank": _minimum_optional(
                        existing.lexical_rank,
                        hit.lexical_rank,
                    ),
                    "final_score": max(existing.final_score, hit.final_score),
                })
                self._hits[query_id][existing_index] = merged
                observed.append(existing.hit_id)
                continue
            if len(self._hits[query_id]) >= self._max_per_question:
                continue
            evidence_id = f"E-{query_id}:{self._next[query_id]}"
            self._next[query_id] += 1
            stored = hit.model_copy(update={
                "hit_id": evidence_id,
                "collection_rank": len(self._hits[query_id]) + 1,
                "retrieval_sources": tuple(dict.fromkeys((
                    *hit.retrieval_sources,
                    source,
                ))),
            })
            self._indexes[query_id][stored.chunk_id] = len(self._hits[query_id])
            self._hits[query_id].append(stored)
            observed.append(evidence_id)
            added.append(evidence_id)
        return tuple(observed), tuple(added)

    def payload(self, *, excerpt_characters: int = 1600) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for query_id, hits in self._hits.items():
            for hit in hits:
                excerpt = hit.excerpt
                if len(excerpt) > excerpt_characters:
                    excerpt = excerpt[:excerpt_characters].rstrip() + " ..."
                values.append({
                    "evidence_id": hit.hit_id,
                    "query_id": query_id,
                    "specification": hit.specification,
                    "version": hit.version,
                    "heading_path": list(hit.heading_path),
                    "line_start": hit.line_start,
                    "line_end": hit.line_end,
                    "retrieval_sources": list(hit.retrieval_sources),
                    "excerpt": excerpt,
                })
        return values


__all__ = ["EvidenceLedger"]

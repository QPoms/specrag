"""Evidence state accumulated by a bounded Researcher loop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from specrag.retrieval.planner import PlannedRetrievalHit


def _minimum_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


class EvidenceLedger:
    """Candidate pool plus a bounded, model-selected active evidence set."""

    def __init__(
        self,
        query_ids: tuple[str, ...],
        *,
        max_per_question: int,
    ) -> None:
        self._candidates = {query_id: [] for query_id in query_ids}
        self._indexes = {query_id: {} for query_id in query_ids}
        self._next = {query_id: 1 for query_id in query_ids}
        self._active_ids = {query_id: [] for query_id in query_ids}
        self._pending_ids = {query_id: [] for query_id in query_ids}
        self._max_per_question = max_per_question

    @property
    def max_per_question(self) -> int:
        """Maximum evidence items that may be active for one question."""

        return self._max_per_question

    def query_ids_with_candidates(self) -> tuple[str, ...]:
        return tuple(
            query_id
            for query_id, hits in self._candidates.items()
            if hits
        )

    def candidate_hits(
        self,
        query_id: str,
    ) -> tuple[PlannedRetrievalHit, ...]:
        return tuple(self._candidates[query_id])

    def active_hits(self, query_id: str) -> tuple[PlannedRetrievalHit, ...]:
        by_id = {hit.hit_id: hit for hit in self._candidates[query_id]}
        return tuple(by_id[value] for value in self._active_ids[query_id])

    def hits(self, query_id: str) -> tuple[PlannedRetrievalHit, ...]:
        """Return all observed candidates for trusted grounding and navigation."""

        return self.candidate_hits(query_id)

    def active_evidence_ids(self, query_id: str) -> tuple[str, ...]:
        return tuple(self._active_ids[query_id])

    def pending_evidence_ids(self, query_id: str) -> tuple[str, ...]:
        return tuple(self._pending_ids[query_id])

    def find(self, query_id: str, evidence_id: str) -> PlannedRetrievalHit | None:
        for hit in self._candidates[query_id]:
            if hit.hit_id == evidence_id:
                return hit
        return None

    def select(self, query_id: str, evidence_ids: Sequence[str]) -> None:
        """Replace the active set after trusted validation of model-selected IDs."""

        selected = tuple(evidence_ids)
        if len(selected) > self._max_per_question:
            raise ValueError("active evidence selection exceeds its limit")
        if len(set(selected)) != len(selected):
            raise ValueError("active evidence selection contains duplicate IDs")
        unknown = tuple(
            evidence_id
            for evidence_id in selected
            if self.find(query_id, evidence_id) is None
        )
        if unknown:
            raise ValueError(
                "active evidence selection contains unknown IDs: "
                + ", ".join(unknown)
            )
        self._active_ids[query_id] = list(selected)
        self._pending_ids[query_id] = []

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
                existing = self._candidates[query_id][existing_index]
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
                self._candidates[query_id][existing_index] = merged
                observed.append(existing.hit_id)
                continue
            evidence_id = f"E-{query_id}:{self._next[query_id]}"
            self._next[query_id] += 1
            stored = hit.model_copy(update={
                "hit_id": evidence_id,
                "collection_rank": len(self._candidates[query_id]) + 1,
                "retrieval_sources": tuple(dict.fromkeys((
                    *hit.retrieval_sources,
                    source,
                ))),
            })
            self._indexes[query_id][stored.chunk_id] = len(
                self._candidates[query_id]
            )
            self._candidates[query_id].append(stored)
            self._pending_ids[query_id].append(evidence_id)
            observed.append(evidence_id)
            added.append(evidence_id)
        return tuple(observed), tuple(added)

    def selection_payload(self) -> dict[str, list[dict[str, Any]]]:
        """Return bounded review views without discarding inactive candidates."""

        return {
            "active_evidence": self._payload_for_status(
                "active",
                excerpt_characters=1600,
            ),
            "pending_candidates": self._payload_for_status(
                "pending",
                excerpt_characters=1600,
            ),
            "inactive_candidates": self._payload_for_status(
                "inactive",
                excerpt_characters=300,
            ),
        }

    def payload(self, *, excerpt_characters: int = 1600) -> list[dict[str, Any]]:
        """Return the complete candidate pool for audit-compatible callers."""

        values: list[dict[str, Any]] = []
        for query_id, hits in self._candidates.items():
            for hit in hits:
                values.append(self._hit_payload(
                    query_id,
                    hit,
                    status=self._status(query_id, hit.hit_id),
                    excerpt_characters=excerpt_characters,
                ))
        return values

    def _status(
        self,
        query_id: str,
        evidence_id: str,
    ) -> Literal["active", "pending", "inactive"]:
        if evidence_id in self._pending_ids[query_id]:
            return "pending"
        if evidence_id in self._active_ids[query_id]:
            return "active"
        return "inactive"

    def _payload_for_status(
        self,
        status: Literal["active", "pending", "inactive"],
        *,
        excerpt_characters: int,
    ) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for query_id, hits in self._candidates.items():
            for hit in hits:
                if self._status(query_id, hit.hit_id) != status:
                    continue
                values.append(self._hit_payload(
                    query_id,
                    hit,
                    status=status,
                    excerpt_characters=excerpt_characters,
                ))
        return values

    @staticmethod
    def _hit_payload(
        query_id: str,
        hit: PlannedRetrievalHit,
        *,
        status: Literal["active", "pending", "inactive"],
        excerpt_characters: int,
    ) -> dict[str, Any]:
        excerpt = hit.excerpt
        if len(excerpt) > excerpt_characters:
            excerpt = excerpt[:excerpt_characters].rstrip() + " ..."
        return {
            "evidence_id": hit.hit_id,
            "query_id": query_id,
            "status": status,
            "specification": hit.specification,
            "version": hit.version,
            "heading_path": list(hit.heading_path),
            "line_start": hit.line_start,
            "line_end": hit.line_end,
            "retrieval_sources": list(hit.retrieval_sources),
            "excerpt": excerpt,
        }


__all__ = ["EvidenceLedger"]

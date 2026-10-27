"""Evidence state accumulated by a bounded Researcher loop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from specrag.models import CandidateEvidence
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

    def seed(
        self,
        query_id: str,
        hits: tuple[PlannedRetrievalHit, ...],
        *,
        active_evidence_ids: Sequence[str] = (),
    ) -> None:
        """Restore source-valid candidates for a corrective Researcher run."""

        if self._candidates[query_id]:
            raise ValueError("evidence ledger can only be seeded once")
        identifiers = tuple(hit.hit_id for hit in hits)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("seed evidence IDs must be unique")
        chunk_ids = tuple(hit.chunk_id for hit in hits)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("seed evidence chunk IDs must be unique")
        active = tuple(active_evidence_ids)
        if len(active) > self._max_per_question:
            raise ValueError("seed active evidence exceeds its limit")
        if len(active) != len(set(active)):
            raise ValueError("seed active evidence contains duplicate IDs")
        unknown = sorted(set(active) - set(identifiers))
        if unknown:
            raise ValueError(
                "seed active evidence contains unknown IDs: "
                + ", ".join(unknown)
            )
        self._candidates[query_id] = list(hits)
        self._indexes[query_id] = {
            hit.chunk_id: index for index, hit in enumerate(hits)
        }
        self._active_ids[query_id] = list(active)
        self._pending_ids[query_id] = []
        while f"E-{query_id}:{self._next[query_id]}" in identifiers:
            self._next[query_id] += 1

    def seed_candidates(
        self,
        candidates: Sequence[CandidateEvidence],
        *,
        active_evidence_ids: Sequence[str] = (),
    ) -> None:
        """Seed all task ledgers from persisted source-valid candidates."""

        unknown_tasks = sorted(
            {candidate.task_id for candidate in candidates}
            - set(self._candidates)
        )
        if unknown_tasks:
            raise ValueError(
                "seed evidence references unknown tasks: "
                + ", ".join(unknown_tasks)
            )
        active = set(active_evidence_ids)
        known = {candidate.evidence_id for candidate in candidates}
        unknown = sorted(active - known)
        if unknown:
            raise ValueError(
                "seed active evidence contains unknown IDs: "
                + ", ".join(unknown)
            )
        for query_id in self._candidates:
            task_candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.task_id == query_id
            )
            if task_candidates:
                self.seed(
                    query_id,
                    tuple(
                        planned_hit_from_candidate(candidate, index=index)
                        for index, candidate in enumerate(
                            task_candidates,
                            start=1,
                        )
                    ),
                    active_evidence_ids=tuple(
                        candidate.evidence_id
                        for candidate in task_candidates
                        if candidate.evidence_id in active
                    ),
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
        """Return complete review views without discarding candidates."""

        return {
            "active_evidence": self._payload_for_status(
                "active",
                excerpt_characters=None,
            ),
            "pending_candidates": self._payload_for_status(
                "pending",
                excerpt_characters=None,
            ),
            "inactive_candidates": self._payload_for_status(
                "inactive",
                excerpt_characters=240,
            ),
        }

    def payload(
        self,
        *,
        excerpt_characters: int | None = None,
    ) -> list[dict[str, Any]]:
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
        excerpt_characters: int | None,
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
        excerpt_characters: int | None,
    ) -> dict[str, Any]:
        excerpt = hit.excerpt
        if (
            excerpt_characters is not None
            and len(excerpt) > excerpt_characters
        ):
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


def planned_hit_from_candidate(
    candidate: CandidateEvidence,
    *,
    index: int,
) -> PlannedRetrievalHit:
    """Restore a persisted candidate as a navigable Researcher hit."""

    return PlannedRetrievalHit(
        hit_id=candidate.evidence_id,
        chunk_id=candidate.chunk_id,
        collection_rank=index,
        specification=candidate.specification,
        version=candidate.version,
        release=candidate.release,
        source_path=candidate.source_path,
        document_sha256=candidate.document_sha256,
        heading_path=candidate.heading_path,
        line_start=candidate.line_start,
        line_end=candidate.line_end,
        excerpt=candidate.excerpt,
        evidence_roles=("corrective_candidate",),
        probe_ids=("judge_gap",),
        retrieval_sources=("corrective_context",),
        rrf_score=0.0,
        route_score=0.0,
        anchor_score=0.0,
        heading_score=0.0,
        agreement_score=0.0,
        noise_penalty=0.0,
        final_score=0.0,
    )


__all__ = ["EvidenceLedger", "planned_hit_from_candidate"]

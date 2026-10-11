"""Validated-only long-term memory service."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import re
from typing import Protocol

from specrag.models import (
    CandidateEvidence,
    EvidenceSynthesis,
    KnowledgeProposal,
    KnowledgeSource,
    MemorySearchResult,
    ResearchTask,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
    ValidatedKnowledge,
)


class KnowledgeStore(Protocol):
    def all(self) -> tuple[ValidatedKnowledge, ...]: ...

    def put(self, knowledge: ValidatedKnowledge) -> None: ...


class MemoryDistiller(Protocol):
    def propose(
        self,
        synthesis: EvidenceSynthesis,
        evidence: tuple[CandidateEvidence, ...],
    ) -> tuple[KnowledgeProposal, ...]: ...


def _key(*parts: str) -> str:
    normalized = "|".join(
        re.sub(r"\s+", " ", part).strip().casefold() for part in parts
    )
    return "K-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24].upper()


class ValidatedMemoryService:
    """Read/search memory freely, but write only from validated evidence."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def search(self, query: str, *, limit: int = 8) -> tuple[MemorySearchResult, ...]:
        if not query.strip():
            raise ValueError("memory search query must not be blank")
        if limit < 1:
            raise ValueError("memory search limit must be positive")
        terms = tuple(
            term for term in re.findall(r"[\w.-]+", query.casefold()) if term
        )
        results: list[MemorySearchResult] = []
        now = datetime.now(timezone.utc)
        for item in self.store.all():
            if item.memory_status != "active":
                continue
            if item.expires_at is not None and item.expires_at <= now:
                continue
            haystack = " ".join((
                item.subject,
                item.predicate,
                item.object,
                *item.conditions,
            )).casefold()
            matches = sum(term in haystack for term in terms)
            if matches:
                results.append(MemorySearchResult(
                    knowledge=item,
                    score=min(1.0, matches / max(1, len(terms))),
                ))
        return tuple(sorted(
            results,
            key=lambda result: (-result.score, result.knowledge.knowledge_id),
        )[:limit])

    def distill_and_store(
        self,
        *,
        synthesis: EvidenceSynthesis,
        tasks: tuple[ResearchTask, ...],
        candidates: tuple[CandidateEvidence, ...],
        validated: tuple[ValidatedEvidence, ...],
        sufficiency: tuple[TaskSufficiencyAssessment, ...],
        distiller: MemoryDistiller,
        now: datetime | None = None,
    ) -> tuple[ValidatedKnowledge, ...]:
        if not tasks or not sufficiency or any(
            item.status != "sufficient" for item in sufficiency
        ):
            raise ValueError("memory requires sufficient evidence for every task")
        eligible = {
            item.evidence_id
            for item in validated
            if item.source_valid and item.validation_status == "supported"
        }
        claim_ids = {claim.claim_id for claim in synthesis.claims}
        proposals = distiller.propose(synthesis, candidates)
        candidate_by_id = {item.evidence_id: item for item in candidates}
        timestamp = now or datetime.now(timezone.utc)
        records: list[ValidatedKnowledge] = []
        for proposal in proposals:
            if proposal.claim_id not in claim_ids:
                raise ValueError("memory proposal references unknown synthesis claim")
            claim = next(
                claim for claim in synthesis.claims if claim.claim_id == proposal.claim_id
            )
            if not set(claim.evidence_ids).issubset(eligible):
                raise ValueError("synthesis claim is not fully validated")
            sources: list[KnowledgeSource] = []
            for evidence_id in claim.evidence_ids:
                try:
                    item = candidate_by_id[evidence_id]
                except KeyError as exc:
                    raise ValueError(
                        f"missing candidate for evidence {evidence_id}"
                    ) from exc
                sources.append(KnowledgeSource(
                    evidence_id=item.evidence_id,
                    specification=item.specification,
                    release=item.release,
                    version=item.version,
                    source_path=item.source_path,
                    document_sha256=item.document_sha256,
                    heading_path=item.heading_path,
                    line_start=item.line_start,
                    line_end=item.line_end,
                ))
            records.append(ValidatedKnowledge(
                knowledge_id=_key(
                    proposal.subject,
                    proposal.predicate,
                    proposal.object,
                    *proposal.conditions,
                    *(source.version for source in sources),
                ),
                claim_id=proposal.claim_id,
                subject=proposal.subject,
                predicate=proposal.predicate,
                object=proposal.object,
                conditions=proposal.conditions,
                sources=tuple(sources),
                validated_at=timestamp,
                expires_at=proposal.expires_at,
            ))

        existing = self.store.all()
        for record in records:
            same_semantics = [
                item for item in existing
                if item.subject == record.subject
                and item.predicate == record.predicate
                and item.object == record.object
                and item.conditions == record.conditions
                and item.memory_status == "active"
            ]
            same_claim = [
                item for item in existing
                if item.subject == record.subject
                and item.predicate == record.predicate
                and item.object != record.object
                and item.memory_status == "active"
            ]
            if same_claim:
                record = record.model_copy(update={"memory_status": "conflicted"})
                for item in same_claim:
                    self.store.put(item.model_copy(update={
                        "memory_status": "conflicted",
                    }))
            elif same_semantics:
                for item in same_semantics:
                    old_versions = {source.version for source in item.sources}
                    new_versions = {source.version for source in record.sources}
                    if old_versions != new_versions:
                        self.store.put(item.model_copy(update={
                            "memory_status": "superseded",
                            "superseded_by": record.knowledge_id,
                        }))
            self.store.put(record)
        return tuple(records)

    def invalidate(self, knowledge_id: str, reason: str) -> ValidatedKnowledge:
        if not reason.strip():
            raise ValueError("invalidation reason must not be blank")
        for item in self.store.all():
            if item.knowledge_id == knowledge_id:
                invalidated = item.model_copy(update={
                    "memory_status": "invalidated",
                    "invalidation_reason": reason,
                })
                self.store.put(invalidated)
                return invalidated
        raise KeyError(f"unknown knowledge ID: {knowledge_id}")

    def expire(self, *, now: datetime | None = None) -> tuple[ValidatedKnowledge, ...]:
        """Mark active records past `expires_at` as invalidated."""

        current = now or datetime.now(timezone.utc)
        expired: list[ValidatedKnowledge] = []
        for item in self.store.all():
            if (
                item.memory_status == "active"
                and item.expires_at is not None
                and item.expires_at <= current
            ):
                updated = item.model_copy(update={
                    "memory_status": "invalidated",
                    "invalidation_reason": "knowledge validity period expired",
                })
                self.store.put(updated)
                expired.append(updated)
        return tuple(expired)


__all__ = ["KnowledgeStore", "MemoryDistiller", "ValidatedMemoryService"]

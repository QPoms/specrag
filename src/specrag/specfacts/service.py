"""SpecFact proposal, review, lifecycle, and hint lookup."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from collections.abc import Callable
from typing import Protocol

from specrag.access import AccessPolicy
from specrag.models import (
    ActorContext,
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchTask,
    SpecFact,
    SpecFactDraft,
    SpecFactProposal,
    SpecFactSearchResult,
    SpecFactSource,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)


class SpecFactStore(Protocol):
    def all_proposals(self) -> tuple[SpecFactProposal, ...]: ...
    def put_proposal(self, proposal: SpecFactProposal) -> None: ...
    def all_facts(self) -> tuple[SpecFact, ...]: ...
    def put_fact(self, fact: SpecFact) -> None: ...


class SpecFactDistiller(Protocol):
    def propose(
        self,
        synthesis: EvidenceSynthesis,
        evidence: tuple[CandidateEvidence, ...],
    ) -> tuple[SpecFactDraft, ...]: ...


class ClaimSpecFactDistiller:
    """Convert already grounded synthesis claims into reviewable drafts."""

    def propose(
        self,
        synthesis: EvidenceSynthesis,
        evidence: tuple[CandidateEvidence, ...],
    ) -> tuple[SpecFactDraft, ...]:
        del evidence
        return tuple(
            SpecFactDraft(
                claim_id=claim.claim_id,
                assertion=claim.text,
                subject="specification claim",
                predicate="states",
                object=claim.text,
            )
            for claim in synthesis.claims
        )


def _key(prefix: str, *parts: str) -> str:
    normalized = "|".join(
        re.sub(r"\s+", " ", part).strip().casefold() for part in parts
    )
    return prefix + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24].upper()


class SpecFactService:
    def __init__(
        self,
        store: SpecFactStore,
        policy: AccessPolicy | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or AccessPolicy()

    def list_facts(self, context: ActorContext) -> tuple[SpecFact, ...]:
        """List every reviewed fact in the current workspace."""
        self.policy.require(context, "specfact.read", workspace_id=context.workspace_id)
        return tuple(
            sorted(
                (
                    item
                    for item in self.store.all_facts()
                    if item.workspace_id == context.workspace_id
                ),
                key=lambda item: item.approved_at,
                reverse=True,
            )
        )

    def list_proposals(self, context: ActorContext) -> tuple[SpecFactProposal, ...]:
        """List reviewable proposals in the current workspace."""
        self.policy.require(context, "specfact.read", workspace_id=context.workspace_id)
        return tuple(
            sorted(
                (
                    item
                    for item in self.store.all_proposals()
                    if item.workspace_id == context.workspace_id
                ),
                key=lambda item: item.proposed_at,
                reverse=True,
            )
        )

    def search(
        self,
        context: ActorContext,
        query: str,
        *,
        release: int,
        limit: int = 8,
        valid_source_paths: frozenset[str] | None = None,
        source_is_valid: Callable[[SpecFactSource], bool] | None = None,
    ) -> tuple[SpecFactSearchResult, ...]:
        self.policy.require(context, "specfact.read", workspace_id=context.workspace_id)
        if not query.strip() or limit < 1:
            raise ValueError("SpecFact query and positive limit are required")
        terms = tuple(re.findall(r"[\w.-]+", query.casefold()))
        now = datetime.now(timezone.utc)
        results: list[SpecFactSearchResult] = []
        for fact in self.store.all_facts():
            if fact.workspace_id != context.workspace_id or fact.status != "approved":
                continue
            if fact.expires_at is not None and fact.expires_at <= now:
                continue
            matching_sources = tuple(source for source in fact.sources if source.release == release)
            if not matching_sources:
                continue
            if valid_source_paths is not None and any(
                source.source_path not in valid_source_paths for source in matching_sources
            ):
                continue
            if source_is_valid is not None and any(
                not source_is_valid(source) for source in matching_sources
            ):
                continue
            haystack = " ".join((
                fact.assertion, fact.subject, fact.predicate, fact.object,
                *fact.conditions,
            )).casefold()
            matches = sum(term in haystack for term in terms)
            if matches:
                results.append(SpecFactSearchResult(
                    fact=fact,
                    score=min(1.0, matches / max(1, len(terms))),
                ))
        return tuple(sorted(results, key=lambda item: (-item.score, item.fact.fact_id))[:limit])

    def propose_from_synthesis(
        self,
        context: ActorContext,
        *,
        run_id: str,
        synthesis: EvidenceSynthesis,
        tasks: tuple[ResearchTask, ...],
        candidates: tuple[CandidateEvidence, ...],
        validated: tuple[ValidatedEvidence, ...],
        sufficiency: tuple[TaskSufficiencyAssessment, ...],
        distiller: SpecFactDistiller,
        now: datetime | None = None,
    ) -> tuple[SpecFactProposal, ...]:
        self.policy.require(context, "specfact.propose", workspace_id=context.workspace_id)
        if not tasks or not sufficiency or any(item.status != "sufficient" for item in sufficiency):
            raise ValueError("SpecFact proposal requires sufficient evidence for every task")
        eligible = {
            item.evidence_id for item in validated
            if item.source_valid and item.validation_status == "supported"
        }
        claims = {claim.claim_id: claim for claim in synthesis.claims}
        candidates_by_id = {item.evidence_id: item for item in candidates}
        timestamp = now or datetime.now(timezone.utc)
        records: list[SpecFactProposal] = []
        for draft in distiller.propose(synthesis, candidates):
            claim = claims.get(draft.claim_id)
            if claim is None:
                raise ValueError("SpecFact draft references unknown synthesis claim")
            if not set(claim.evidence_ids).issubset(eligible):
                raise ValueError("synthesis claim is not fully validated")
            sources = tuple(
                SpecFactSource(
                    evidence_id=item.evidence_id,
                    specification=item.specification,
                    release=item.release,
                    version=item.version,
                    source_path=item.source_path,
                    document_sha256=item.document_sha256,
                    heading_path=item.heading_path,
                    line_start=item.line_start,
                    line_end=item.line_end,
                )
                for evidence_id in claim.evidence_ids
                for item in (candidates_by_id[evidence_id],)
            )
            operation_key = _key(
                "SFP-OP-", context.workspace_id, run_id, draft.claim_id,
                draft.assertion,
                *(source.document_sha256 for source in sources),
                *(source.version for source in sources),
                *(source.source_path for source in sources),
            )
            existing = next((
                item for item in self.store.all_proposals()
                if item.operation_key == operation_key
            ), None)
            if existing is not None:
                records.append(existing)
                continue
            proposal = SpecFactProposal(
                proposal_id=_key("SFP-", operation_key),
                operation_key=operation_key,
                workspace_id=context.workspace_id,
                run_id=run_id,
                claim_id=draft.claim_id,
                assertion=draft.assertion,
                subject=draft.subject,
                predicate=draft.predicate,
                object=draft.object,
                conditions=draft.conditions,
                sources=sources,
                proposed_by=context.user_id,
                proposed_at=timestamp,
                expires_at=draft.expires_at,
            )
            self.store.put_proposal(proposal)
            records.append(proposal)
        return tuple(records)

    def review(
        self,
        context: ActorContext,
        proposal_id: str,
        *,
        approve: bool,
        reason: str | None = None,
        now: datetime | None = None,
    ) -> SpecFactProposal:
        proposal = next((
            item for item in self.store.all_proposals()
            if item.proposal_id == proposal_id
        ), None)
        if proposal is None:
            raise KeyError(f"unknown SpecFact proposal: {proposal_id}")
        self.policy.require(context, "specfact.review", workspace_id=proposal.workspace_id)
        if proposal.status != "pending":
            return proposal
        if not approve and not (reason and reason.strip()):
            raise ValueError("rejection requires a reason")
        timestamp = now or datetime.now(timezone.utc)
        reviewed = proposal.model_copy(update={
            "status": "approved" if approve else "rejected",
            "reviewer_user_id": context.user_id,
            "reviewed_at": timestamp,
            "review_reason": reason.strip() if reason else None,
        })
        self.store.put_proposal(reviewed)
        if approve:
            fact = SpecFact(
                fact_id=_key("SF-", proposal.workspace_id, proposal.proposal_id),
                proposal_id=proposal.proposal_id,
                workspace_id=proposal.workspace_id,
                assertion=proposal.assertion,
                subject=proposal.subject,
                predicate=proposal.predicate,
                object=proposal.object,
                conditions=proposal.conditions,
                sources=proposal.sources,
                reviewer_user_id=context.user_id,
                approved_at=timestamp,
                expires_at=proposal.expires_at,
            )
            self._store_with_lifecycle(fact)
        return reviewed

    def _store_with_lifecycle(self, fact: SpecFact) -> None:
        for existing in self.store.all_facts():
            if existing.workspace_id != fact.workspace_id or existing.status != "approved":
                continue
            same_subject = existing.subject == fact.subject and existing.predicate == fact.predicate
            if same_subject and existing.object != fact.object:
                self.store.put_fact(existing.model_copy(update={"status": "conflicted"}))
                fact = fact.model_copy(update={"status": "conflicted"})
            elif same_subject and existing.object == fact.object and existing.conditions == fact.conditions:
                if {s.version for s in existing.sources} != {s.version for s in fact.sources}:
                    self.store.put_fact(existing.model_copy(update={
                        "status": "superseded", "superseded_by": fact.fact_id,
                    }))
        self.store.put_fact(fact)

    def invalidate(self, context: ActorContext, fact_id: str, reason: str) -> SpecFact:
        if not reason.strip():
            raise ValueError("invalidation reason must not be blank")
        fact = next((item for item in self.store.all_facts() if item.fact_id == fact_id), None)
        if fact is None:
            raise KeyError(f"unknown SpecFact: {fact_id}")
        self.policy.require(context, "specfact.review", workspace_id=fact.workspace_id)
        updated = fact.model_copy(update={
            "status": "invalidated", "invalidation_reason": reason.strip(),
        })
        self.store.put_fact(updated)
        return updated


__all__ = [
    "ClaimSpecFactDistiller", "SpecFactDistiller", "SpecFactService", "SpecFactStore",
]

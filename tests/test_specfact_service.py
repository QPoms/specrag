from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from specrag.access import AuthorizationError
from specrag.infrastructure import InMemorySpecFactStore
from specrag.models import (
    ActorContext,
    CandidateEvidence,
    ResearchTask,
    SpecFactDraft,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)
from specrag.specfacts import SpecFactService


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _context(workspace: str = "WA", role: str = "member", user: str = "alice"):
    return ActorContext(
        user_id=user,
        workspace_id=workspace,
        role=role,
        session_id="session",
        authenticated_at=NOW,
        expires_at=NOW + timedelta(hours=12),
    )


def _candidate(version: str = "17.1.0") -> CandidateEvidence:
    return CandidateEvidence(
        evidence_id="E1", task_id="T1", chunk_id="CH1",
        specification="24.501", release=int(version.split(".", 1)[0]),
        version=version,
        source_path=f"data/3gpp/24.501/v{version}/source.md",
        document_sha256="A" * 64, line_start=2, line_end=2,
        excerpt="The UE shall initiate registration.",
    )


def _validated() -> ValidatedEvidence:
    return ValidatedEvidence(
        evidence_id="E1", task_id="T1", source_valid=True,
        supports_target=True, validation_status="supported", rationale="supported",
    )


def _inputs():
    answer = "The UE initiates registration. [E1]"
    task = ResearchTask(
        task_id="T1", question="What starts registration?", objective="Find trigger.",
    )
    sufficient = TaskSufficiencyAssessment(
        task_id="T1", status="sufficient", supporting_evidence_ids=("E1",),
        rationale="complete",
    )
    return answer, task, sufficient


class _Distiller:
    def __init__(self, object_value: str = "registration") -> None:
        self.object_value = object_value

    def propose(self, answer, evidence):
        return (SpecFactDraft(
            claim_id="C1", assertion=f"UE initiates {self.object_value}.",
            subject="UE", predicate="initiates", object=self.object_value,
            evidence_ids=("E1",),
            conditions=("registration procedure",),
        ),)


def _propose(service, context=None, *, version="17.1.0", distiller=None):
    answer, task, sufficient = _inputs()
    return service.propose_from_answer(
        context or _context(), run_id="RUN-" + "1" * 32,
        answer=answer, tasks=(task,), candidates=(_candidate(version),),
        validated=(_validated(),), sufficiency=(sufficient,),
        distiller=distiller or _Distiller(), now=NOW,
    )[0]


def test_proposal_is_pending_and_idempotent() -> None:
    store = InMemorySpecFactStore()
    service = SpecFactService(store)
    first = _propose(service)
    second = _propose(service)
    assert first == second
    assert first.status == "pending"
    assert service.search(_context(), "UE registration", release=17) == ()


def test_only_reviewer_in_same_workspace_can_approve() -> None:
    service = SpecFactService(InMemorySpecFactStore())
    proposal = _propose(service)
    with pytest.raises(AuthorizationError):
        service.review(_context(), proposal.proposal_id, approve=True)
    with pytest.raises(AuthorizationError):
        service.review(_context("WB", "reviewer", "bob"), proposal.proposal_id, approve=True)
    reviewed = service.review(
        _context("WA", "reviewer", "rachel"), proposal.proposal_id,
        approve=True, now=NOW + timedelta(minutes=1),
    )
    assert reviewed.status == "approved"
    assert len(service.search(_context(), "UE registration", release=17)) == 1
    assert service.search(_context("WB"), "UE registration", release=17) == ()


def test_source_release_and_path_are_checked_on_read() -> None:
    service = SpecFactService(InMemorySpecFactStore())
    proposal = _propose(service)
    service.review(_context(role="reviewer"), proposal.proposal_id, approve=True)
    assert service.search(_context(), "registration", release=18) == ()
    assert service.search(
        _context(), "registration", release=17,
        valid_source_paths=frozenset({"different.md"}),
    ) == ()
    assert service.search(
        _context(), "registration", release=17,
        source_is_valid=lambda source: False,
    ) == ()


def test_rejection_requires_reason_and_never_creates_fact() -> None:
    service = SpecFactService(InMemorySpecFactStore())
    proposal = _propose(service)
    reviewer = _context(role="reviewer")
    with pytest.raises(ValueError, match="reason"):
        service.review(reviewer, proposal.proposal_id, approve=False)
    rejected = service.review(
        reviewer, proposal.proposal_id, approve=False, reason="claim is too broad"
    )
    assert rejected.status == "rejected"
    assert service.store.all_facts() == ()


def test_new_release_supersedes_same_fact_and_conflict_hides_both() -> None:
    store = InMemorySpecFactStore()
    service = SpecFactService(store)
    reviewer = _context(role="reviewer")
    first = _propose(service)
    service.review(reviewer, first.proposal_id, approve=True, now=NOW)
    second = _propose(service, version="18.1.0")
    service.review(
        reviewer, second.proposal_id, approve=True, now=NOW + timedelta(minutes=1)
    )
    assert {fact.status for fact in store.all_facts()} == {"approved", "superseded"}
    assert len(service.search(_context(), "registration", release=18)) == 1
    assert service.search(_context(), "registration", release=17) == ()

    conflict = _propose(service, version="19.1.0", distiller=_Distiller("deregistration"))
    service.review(
        reviewer, conflict.proposal_id, approve=True, now=NOW + timedelta(minutes=2)
    )
    assert service.search(_context(), "registration", release=19) == ()


def test_invalidated_fact_is_not_returned() -> None:
    store = InMemorySpecFactStore()
    service = SpecFactService(store)
    reviewer = _context(role="reviewer")
    proposal = _propose(service)
    service.review(reviewer, proposal.proposal_id, approve=True)
    fact = store.all_facts()[0]
    invalidated = service.invalidate(reviewer, fact.fact_id, "source hash changed")
    assert invalidated.status == "invalidated"
    assert service.search(_context(), "registration", release=17) == ()

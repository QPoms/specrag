from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from types import SimpleNamespace

import pytest

from specrag.access import AuthorizationError, LocalIdentityService
from specrag.evidence import SourceEvidenceValidator
from specrag.infrastructure import (
    InMemoryRunStore,
    InMemorySpecFactStore,
    SQLiteDatabase,
    SQLiteIdentityStore,
)
from specrag.models import (
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchRequest,
    ResearchTask,
    SpecFactDraft,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)
from specrag.orchestration import RunHarness
from specrag.specfacts import SpecFactService


class _Workflow:
    def resume(self, *, thread_id):
        return {"stop_reason": "completed"}

    def checkpoint(self, *, thread_id):
        return SimpleNamespace(values={"thread_id": thread_id})


class _Application:
    workflow = _Workflow()

    def run(self, **kwargs):
        return {"stop_reason": "completed"}


class _Distiller:
    def propose(self, answer, evidence):
        return (SpecFactDraft(
            claim_id="C1", assertion="The UE initiates registration.",
            subject="UE", predicate="initiates", object="registration",
            evidence_ids=("E1",),
        ),)


def test_two_workspace_acceptance_keeps_runs_and_facts_isolated(tmp_path) -> None:
    database = SQLiteDatabase(tmp_path / "state.sqlite3")
    identity = LocalIdentityService(SQLiteIdentityStore(database))
    try:
        identity.bootstrap("alice", "alice secure password")
        alice_token, _ = identity.login("alice", "alice secure password")
        workspace_a = identity.create_workspace(alice_token, "Workspace A")
        alice = identity.context(alice_token, workspace_a.workspace_id)
        identity.add_user(alice, "reviewer", "review secure password", "reviewer")
        identity.add_user(alice, "carol", "carol secure password", "member")
        identity.add_user(alice, "bob", "bob secure password", "member")
        identity.remove_user(alice, "bob")

        reviewer_token, _ = identity.login("reviewer", "review secure password")
        reviewer = identity.context(reviewer_token, workspace_a.workspace_id)
        carol_token, _ = identity.login("carol", "carol secure password")
        carol = identity.context(carol_token, workspace_a.workspace_id)
        bob_token, _ = identity.login("bob", "bob secure password")
        workspace_b = identity.create_workspace(bob_token, "Workspace B")
        bob = identity.context(bob_token, workspace_b.workspace_id)

        run_store = InMemoryRunStore()
        harness = RunHarness(
            run_store,
            lambda release, control, events, research_context: _Application(),
        )
        run = harness.start(alice, ResearchRequest(question="What starts registration?"))
        for operation in (harness.inspect, harness.resume, harness.cancel):
            with pytest.raises(AuthorizationError):
                operation(bob, run.run.run_id)

        relative = "data/3gpp/24.501/v17.1.0/source.md"
        source = tmp_path / relative
        source.parent.mkdir(parents=True)
        text = "The UE shall initiate the registration procedure.\n"
        source.write_text(text, encoding="utf-8")
        candidate = CandidateEvidence(
            evidence_id="E1", task_id="T1", chunk_id="CH1",
            specification="24.501", release=17, version="17.1.0",
            source_path=relative,
            document_sha256=hashlib.sha256(source.read_bytes()).hexdigest().upper(),
            line_start=1, line_end=1,
            excerpt="The UE shall initiate the registration procedure.",
        )
        assert SourceEvidenceValidator(tmp_path).validate(candidate).valid
        synthesis = EvidenceSynthesis(
            answer="The UE initiates registration. [E1]",
            evidence_ids=("E1",),
        )
        task = ResearchTask(task_id="T1", question="Q", objective="O")
        validated = ValidatedEvidence(
            evidence_id="E1", task_id="T1", source_valid=True,
            supports_target=True, validation_status="supported", rationale="source checked",
        )
        sufficient = TaskSufficiencyAssessment(
            task_id="T1", status="sufficient", supporting_evidence_ids=("E1",),
            rationale="complete",
        )
        facts = SpecFactService(InMemorySpecFactStore())
        proposal = facts.propose_from_answer(
            alice, run_id=run.run.run_id, answer=synthesis.answer, tasks=(task,),
            candidates=(candidate,), validated=(validated,), sufficiency=(sufficient,),
            distiller=_Distiller(), now=datetime.now(timezone.utc),
        )[0]
        facts.review(reviewer, proposal.proposal_id, approve=True)
        result = facts.search(carol, "UE registration", release=17)
        assert result[0].fact.sources[0].document_sha256 == candidate.document_sha256
        assert SourceEvidenceValidator(tmp_path).validate(candidate).valid
        assert synthesis.evidence_ids == ("E1",)
        assert facts.search(bob, "UE registration", release=17) == ()
    finally:
        database.close()

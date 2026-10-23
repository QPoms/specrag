from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from specrag.access import AuthorizationError
from specrag.infrastructure import SQLiteResearchResultStore
from specrag.interfaces.api import create_app
from specrag.interfaces.workspace_services import WorkspaceServices
from specrag.models import (
    CandidateEvidence,
    CorpusVersionRef,
    EvidenceSynthesis,
    ResearchInputSnapshot,
    ResearchOutcome,
    ResearchRun,
    RunEvent,
    SpecificationScope,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)


NOW = datetime(2026, 10, 22, 10, tzinfo=UTC)
VERSION = CorpusVersionRef(specification="24.501", version="17.9.0")
SCOPE = SpecificationScope(policy="exact_version", versions=(VERSION,))


def _candidate(excerpt: str) -> CandidateEvidence:
    return CandidateEvidence(
        evidence_id="evidence-1",
        task_id="task-1",
        chunk_id="chunk-1",
        specification="24.501",
        release=17,
        version="17.9.0",
        source_path="corpus/24.501/17.9.0.md",
        document_sha256="A" * 64,
        heading_path=("5", "5.4"),
        line_start=10,
        line_end=14,
        excerpt=excerpt,
    )


def _outcome(run: ResearchRun, answer: str, *, at: datetime) -> ResearchOutcome:
    candidate = _candidate(answer)
    return ResearchOutcome(
        run_id=run.run_id,
        run_attempt=run.run_attempt,
        question=run.question,
        objective=run.input_snapshot.objective,
        conditions=run.input_snapshot.conditions,
        requested_scope=SCOPE,
        corpus_versions=(VERSION,),
        synthesis=EvidenceSynthesis(
            answer=answer, evidence_ids=(candidate.evidence_id,)
        ),
        candidate_evidence=(candidate,),
        validated_evidence=(ValidatedEvidence(
            evidence_id=candidate.evidence_id,
            task_id=candidate.task_id,
            source_valid=True,
            supports_target=True,
            validation_status="supported",
            rationale="The exact cited clause supports the target proposition.",
        ),),
        assessments=(TaskSufficiencyAssessment(
            task_id=candidate.task_id,
            status="sufficient",
            supporting_evidence_ids=(candidate.evidence_id,),
            rationale="The normative requirement is directly supported.",
        ),),
        stop_reason="completed",
        created_at=at,
    )


def _stopped(run: ResearchRun, sequence: int) -> RunEvent:
    return RunEvent(
        event_id="EVT-" + f"{sequence:032X}",
        run_id=run.run_id,
        workspace_id=run.workspace_id,
        sequence=sequence,
        event_type="run_stopped",
        operation_key=f"run:stopped:completed:attempt:{run.run_attempt}",
        occurred_at=run.finished_at,
        payload={"reason": "completed"},
    )


def _authorization(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_results_are_attempt_scoped_idempotent_private_and_restart_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.sqlite3"
    services = WorkspaceServices.create(path)
    services.identity.bootstrap("alice", "correct horse battery")
    token, _ = services.identity.login("alice", "correct horse battery")
    workspace = services.identity.create_workspace(token, "Result Workspace")
    alice = services.identity.context(token, workspace.workspace_id)
    accepted = services.conversations.create(alice, title="Registration behavior")
    conversation_id = accepted.conversation.conversation_id
    message = services.conversations.append_user_message(
        alice,
        conversation_id,
        text="When shall the UE register?",
        operation_key="message-1",
        expected_revision=0,
    )
    snapshot = ResearchInputSnapshot(
        question="When shall the UE register?",
        objective="Establish the normative registration trigger.",
        conditions=("UE is in 5GMM-DEREGISTERED",),
        requested_scope=SCOPE,
        answer_language="English",
        detail_level="standard",
    )
    run = ResearchRun(
        run_id="RUN-" + "A" * 32,
        thread_id="THREAD-" + "A" * 32,
        workspace_id=workspace.workspace_id,
        owner_user_id=alice.user_id,
        question=snapshot.question,
        release=17,
        status="completed",
        stop_reason="completed",
        created_at=NOW,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
        corpus_version_policy="exact_version",
        corpus_versions=(VERSION,),
        conversation_id=conversation_id,
        origin_message_id=message.message.message_id,
        origin_turn_id=message.turn.turn_id,
        submitted_user_sequence=message.message.sequence,
        submission_operation_key="turn-research-1",
        input_snapshot=snapshot,
    )
    services.runs.put(run)
    services.runs.finalize(
        run, _stopped(run, 1), _outcome(run, "The UE shall register.", at=run.finished_at)
    )

    later = NOW + timedelta(seconds=3)
    attempt_one = run.model_copy(update={
        "run_attempt": 1,
        "started_at": NOW + timedelta(seconds=2),
        "finished_at": later,
    })
    services.runs.put(attempt_one)
    services.runs.finalize(
        attempt_one,
        _stopped(attempt_one, 2),
        _outcome(attempt_one, "The updated evidence still requires registration.", at=later),
    )

    first = services.research_results.materialize(run.run_id, 0)
    second = services.research_results.materialize(run.run_id, 1)
    assert first is not None and second is not None
    assert first.run_attempt == 0
    assert first.evidence_status == "sufficient"
    assert first.evidence[0].source.source_path == "corpus/24.501/17.9.0.md"
    assert first.evidence[0].source.document_sha256 == "A" * 64
    assert first.evidence[0].source.line_start == 10
    assert second.supersedes_result_id == first.result_id
    assert services.research_results.materialize(run.run_id, 1) == second
    assert services.research_results.list_for_conversation(
        alice, conversation_id
    ) == (first, second)

    services.identity.add_user(
        alice, "admin", "admin secure password", "admin"
    )
    admin_token, _ = services.identity.login("admin", "admin secure password")
    admin = services.identity.context(admin_token, workspace.workspace_id)
    with pytest.raises(AuthorizationError):
        services.research_results.get(admin, second.result_id)

    with TestClient(create_app(services, worker_mode="external")) as client:
        response = client.get(
            f"/api/v1/workspaces/{workspace.workspace_id}/research-results/"
            f"{second.result_id}",
            headers=_authorization(token),
        )
        assert response.status_code == 200
        assert response.json()["answer"] == second.answer
        listed = client.get(
            f"/api/v1/workspaces/{workspace.workspace_id}/conversations/"
            f"{conversation_id}/results",
            headers=_authorization(token),
        )
        assert [item["result_id"] for item in listed.json()] == [
            first.result_id,
            second.result_id,
        ]
        hidden = client.get(
            f"/api/v1/workspaces/{workspace.workspace_id}/research-results/"
            f"{second.result_id}",
            headers=_authorization(admin_token),
        )
        assert hidden.status_code == 404
    services.close()

    restored = WorkspaceServices.create(path)
    try:
        restored_alice = restored.identity.context(token, workspace.workspace_id)
        persisted = restored.research_results.get(restored_alice, second.result_id)
        assert persisted == second
        assert SQLiteResearchResultStore(restored.database).latest_for_run(
            run.run_id
        ) == second
        assert restored.harness.inspect(
            restored_alice, run.run_id
        ).result_id == second.result_id
    finally:
        restored.close()


def test_atomic_terminal_commit_is_replay_safe(tmp_path: Path) -> None:
    services = WorkspaceServices.create(tmp_path / "atomic.sqlite3")
    services.identity.bootstrap("alice", "correct horse battery")
    token, _ = services.identity.login("alice", "correct horse battery")
    workspace = services.identity.create_workspace(token, "Atomic Workspace")
    alice = services.identity.context(token, workspace.workspace_id)
    snapshot = ResearchInputSnapshot(
        question="Question",
        objective="Objective",
        requested_scope=SCOPE,
        answer_language="English",
        detail_level="standard",
    )
    run = ResearchRun(
        run_id="RUN-" + "B" * 32,
        thread_id="THREAD-" + "B" * 32,
        workspace_id=workspace.workspace_id,
        owner_user_id=alice.user_id,
        question=snapshot.question,
        release=17,
        status="completed",
        stop_reason="completed",
        created_at=NOW,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
        corpus_version_policy="exact_version",
        corpus_versions=(VERSION,),
        input_snapshot=snapshot,
    )
    outcome = _outcome(run, "Supported answer.", at=run.finished_at)
    event = _stopped(run, 1)
    services.runs.put(run)
    services.runs.finalize(run, event, outcome)
    services.runs.finalize(run, event, outcome)

    connection = services.database.connection
    assert connection.execute(
        "SELECT COUNT(*) FROM run_events WHERE run_id=?", (run.run_id,)
    ).fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM run_outcomes WHERE run_id=?", (run.run_id,)
    ).fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM run_completion_jobs WHERE run_id=?", (run.run_id,)
    ).fetchone()[0] == 1
    assert services.runs.get(run.run_id) == run

    attempt_one = run.model_copy(update={
        "run_attempt": 1,
        "started_at": NOW + timedelta(seconds=2),
        "finished_at": NOW + timedelta(seconds=3),
    })
    attempt_one_outcome = _outcome(
        attempt_one,
        "Supported answer after resume.",
        at=attempt_one.finished_at,
    )
    attempt_one_event = _stopped(attempt_one, 2)
    services.runs.put(attempt_one)
    services.runs.finalize(
        attempt_one, attempt_one_event, attempt_one_outcome
    )
    assert services.completion_queue.claim(
        "completion-1", lease_seconds=60
    )[:2] == (run.run_id, 0)
    assert services.completion_queue.claim(
        "completion-2", lease_seconds=60
    ) is None
    assert services.completion_queue.complete(run.run_id, 0, "completion-1")
    services.runs.finalize(run, event, outcome)
    assert services.completion_queue.pending_count() == 1
    assert services.completion_queue.claim(
        "completion-2", lease_seconds=60
    )[:2] == (run.run_id, 1)
    assert services.completion_queue.complete(run.run_id, 1, "completion-2")
    assert connection.execute(
        "SELECT COUNT(*) FROM run_completion_receipts WHERE run_id=?",
        (run.run_id,),
    ).fetchone()[0] == 2
    services.close()

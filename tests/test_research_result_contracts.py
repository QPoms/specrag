from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from specrag.evidence import ResearchResultMaterializer
from specrag.models import (
    CandidateEvidence,
    CorpusVersionRef,
    EvidenceSynthesis,
    ResearchOutcome,
    ResearchResult,
    ResearchResultEvidence,
    ResearchRun,
    SpecificationScope,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)


NOW = datetime(2026, 10, 22, tzinfo=UTC)
RUN_ID = "RUN-" + "A" * 32
RESULT_ID = "RESULT-" + "B" * 32
HASH = "a" * 64


def _scope() -> SpecificationScope:
    return SpecificationScope(
        policy="exact_version",
        versions=(
            CorpusVersionRef(specification="24.501", version="17.9.0"),
        ),
    )


def _candidate() -> CandidateEvidence:
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
        excerpt="The UE shall perform the procedure.",
    )


def _validation(*, source_valid: bool = True) -> ValidatedEvidence:
    return ValidatedEvidence(
        evidence_id="evidence-1",
        task_id="task-1",
        source_valid=source_valid,
        supports_target=True,
        validation_status="supported",
        rationale="The cited clause directly supports the proposition.",
    )


def test_scope_normalizes_exact_versions_and_rejects_ambiguous_shapes() -> None:
    scope = _scope()
    assert scope.versions[0].release == 17

    with pytest.raises(ValidationError, match="Release scope"):
        SpecificationScope(policy="highest_version_within_release")

    with pytest.raises(ValidationError, match="exact scope"):
        SpecificationScope(policy="exact_version", versions=())


def test_outcome_synthesis_can_only_reference_validated_evidence() -> None:
    with pytest.raises(ValidationError, match="unvalidated evidence"):
        ResearchOutcome(
            run_id=RUN_ID,
            question="What does the UE do?",
            objective="Establish the normative UE behavior.",
            requested_scope=_scope(),
            corpus_versions=_scope().versions,
            synthesis=EvidenceSynthesis(
                answer="The UE performs the procedure.",
                evidence_ids=("evidence-1",),
            ),
            candidate_evidence=(_candidate(),),
            stop_reason="completed",
            created_at=NOW,
        )


def test_result_copies_source_valid_evidence_and_checks_citations() -> None:
    evidence = ResearchResultEvidence(
        source=_candidate(), validation=_validation()
    )
    result = ResearchResult(
        result_id=RESULT_ID,
        run_id=RUN_ID,
        workspace_id="workspace-a",
        owner_user_id="alice",
        publication_kind="final",
        evidence_status="sufficient",
        answer="The UE performs the procedure.",
        evidence=(evidence,),
        evidence_ids=("evidence-1",),
        requested_scope=_scope(),
        corpus_versions=_scope().versions,
        source_outcome_sha256=HASH,
        content_sha256=HASH,
        created_at=NOW,
    )
    assert result.evidence[0].source.document_sha256 == "A" * 64

    with pytest.raises(ValidationError, match="not copied"):
        ResearchResult.model_validate({
            **result.model_dump(),
            "evidence_ids": ("evidence-2",),
        })

    with pytest.raises(ValidationError, match="source-valid"):
        ResearchResultEvidence(
            source=_candidate(), validation=_validation(source_valid=False)
        )


def test_insufficient_result_is_valid_without_cited_evidence() -> None:
    result = ResearchResult(
        result_id=RESULT_ID,
        run_id=RUN_ID,
        workspace_id="workspace-a",
        owner_user_id="alice",
        evidence_status="insufficient",
        answer="The available specification evidence is insufficient.",
        requested_scope=_scope(),
        corpus_versions=_scope().versions,
        unresolved_questions=("The triggering condition remains unverified.",),
        source_outcome_sha256=HASH,
        content_sha256=HASH,
        created_at=NOW,
    )
    assert result.evidence_ids == ()


@pytest.mark.parametrize(
    ("status", "validation_status"),
    [
        ("partial", "partial"),
        ("insufficient", "rejected"),
        ("conflict", "conflict"),
    ],
)
def test_materializer_never_promotes_non_sufficient_evidence(
    status: str, validation_status: str
) -> None:
    run = ResearchRun(
        run_id=RUN_ID,
        thread_id="THREAD-" + "A" * 32,
        workspace_id="workspace-a",
        owner_user_id="alice",
        question="What does the UE do?",
        release=17,
        status="completed",
        stop_reason=(
            "insufficient_evidence" if status == "insufficient" else "completed"
        ),
        created_at=NOW,
        started_at=NOW,
        finished_at=NOW,
        corpus_version_policy="exact_version",
        corpus_versions=_scope().versions,
    )
    candidate = _candidate()
    outcome = ResearchOutcome(
        run_id=run.run_id,
        question=run.question,
        objective="Establish the normative UE behavior.",
        requested_scope=_scope(),
        corpus_versions=_scope().versions,
        synthesis=EvidenceSynthesis(
            answer="This assertion must not be promoted.",
            evidence_ids=(candidate.evidence_id,),
        ),
        candidate_evidence=(candidate,),
        validated_evidence=(ValidatedEvidence(
            evidence_id=candidate.evidence_id,
            task_id=candidate.task_id,
            source_valid=validation_status != "rejected",
            supports_target=validation_status != "rejected",
            validation_status=validation_status,
            rationale="Validation did not establish sufficient support.",
        ),),
        assessments=(TaskSufficiencyAssessment(
            task_id=candidate.task_id,
            status=status,
            missing_relation="The decisive condition remains unresolved.",
            rationale="More normative evidence is required.",
        ),),
        stop_reason=run.stop_reason,
        created_at=NOW,
    )

    first = ResearchResultMaterializer().materialize(run, outcome)
    replay = ResearchResultMaterializer().materialize(run, outcome)

    assert first == replay
    assert first.evidence_status == status
    assert first.evidence_ids == ()
    assert "must not be promoted" not in first.answer
    if validation_status == "rejected":
        assert first.evidence == ()

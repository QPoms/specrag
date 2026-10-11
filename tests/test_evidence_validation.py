from __future__ import annotations

import hashlib
from pathlib import Path

from specrag.evidence import (
    EvidenceConflictService,
    EvidenceSufficiencyService,
    EvidenceSynthesisService,
    EvidenceValidationService,
    SourceEvidenceValidator,
    deduplicate_candidates,
)
from specrag.models import (
    CandidateEvidence,
    EvidenceBackedClaim,
    EvidenceConflict,
    EvidenceSynthesis,
    EvidenceSupportAssessment,
    ResearchPlan,
    ResearchTask,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)
from specrag.orchestration import (
    EvidenceSufficiencyHandler,
    EvidenceSynthesisHandler,
    EvidenceValidationHandler,
    ResearchNodeHandlers,
    build_single_research_workflow,
    new_research_state,
)


class _SupportingAssessor:
    def __init__(self) -> None:
        self.calls = 0

    def assess(self, task, evidence):
        self.calls += 1
        return EvidenceSupportAssessment(
            evidence_id=evidence.evidence_id,
            status="supported",
            rationale="The source directly states the requested relation.",
        )


def _source(tmp_path: Path) -> tuple[str, str]:
    relative = "data/3gpp/24.501/v17.1.0/source.md"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# 5 Registration\n"
        "The UE shall initiate the registration procedure.\n"
        "The network shall process the request.\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return relative, digest


def _candidate(tmp_path: Path, **updates) -> CandidateEvidence:
    source_path, digest = _source(tmp_path)
    values = {
        "evidence_id": "E1",
        "task_id": "T1",
        "chunk_id": "C1",
        "specification": "24.501",
        "release": 17,
        "version": "17.1.0",
        "source_path": source_path,
        "document_sha256": digest,
        "heading_path": ("5 Registration",),
        "line_start": 2,
        "line_end": 3,
        "excerpt": (
            "The UE shall initiate the registration procedure.\n"
            "The network shall process the request."
        ),
    }
    values.update(updates)
    return CandidateEvidence(**values)


def _task() -> ResearchTask:
    return ResearchTask(
        task_id="T1",
        question="What starts registration?",
        objective="Find the normative registration trigger.",
    )


def test_source_validation_checks_hash_locator_and_exact_excerpt(
    tmp_path: Path,
) -> None:
    result = SourceEvidenceValidator(tmp_path).validate(_candidate(tmp_path))

    assert result.valid is True
    assert result.path_allowed is True
    assert result.document_hash_valid is True
    assert result.locator_valid is True
    assert result.excerpt_valid is True
    assert result.errors == ()


def test_invalid_source_is_rejected_before_support_assessment(
    tmp_path: Path,
) -> None:
    assessor = _SupportingAssessor()
    service = EvidenceValidationService(
        SourceEvidenceValidator(tmp_path),
        assessor,
    )
    candidate = _candidate(tmp_path, document_sha256="B" * 64)

    batch = service.validate((_task(),), (candidate,))

    assert assessor.calls == 0
    assert batch.source_validations[0].document_hash_valid is False
    assert batch.validated_evidence[0].validation_status == "rejected"
    assert batch.validated_evidence[0].source_valid is False


def test_overlapping_duplicate_candidates_are_collapsed(tmp_path: Path) -> None:
    first = _candidate(tmp_path)
    duplicate = first.model_copy(update={
        "evidence_id": "E2",
        "chunk_id": "C2",
        "line_start": 2,
        "excerpt": "The UE shall initiate the registration procedure.",
    })

    assert deduplicate_candidates((first, duplicate)) == (first,)


def test_validation_handler_updates_graph_state_without_sufficiency_decision(
    tmp_path: Path,
) -> None:
    service = EvidenceValidationService(
        SourceEvidenceValidator(tmp_path),
        _SupportingAssessor(),
    )
    state = new_research_state(
        research_id="R1",
        user_question="What starts registration?",
    )
    state["research_tasks"] = (_task(),)
    state["candidate_evidence"] = (_candidate(tmp_path),)

    update = EvidenceValidationHandler(service)(state)

    assert update["source_validations"][0].valid is True
    assert update["validated_evidence"][0].source_valid is True
    assert update["validated_evidence"][0].supports_target is True
    assert "evidence_sufficient" not in update


def _validated() -> ValidatedEvidence:
    return ValidatedEvidence(
        evidence_id="E1",
        task_id="T1",
        source_valid=True,
        supports_target=True,
        validation_status="supported",
        rationale="The passage supports one required fact.",
    )


def _candidate_placeholder() -> CandidateEvidence:
    return CandidateEvidence(
        evidence_id="E1",
        task_id="T1",
        chunk_id="C1",
        specification="24.501",
        release=17,
        version="17.1.0",
        source_path="data/3gpp/24.501/v17.1.0/source.md",
        document_sha256="A" * 64,
        line_start=1,
        line_end=1,
        excerpt="Registration evidence.",
    )


class _SufficiencyAssessor:
    def __init__(self, assessment: TaskSufficiencyAssessment) -> None:
        self.assessment = assessment

    def assess(self, task, candidates, evidence):
        return self.assessment


def test_supported_item_can_still_leave_task_level_missing_relation() -> None:
    assessment = TaskSufficiencyAssessment(
        task_id="T1",
        status="partial",
        supporting_evidence_ids=("E1",),
        missing_relation="The trigger-to-procedure relation is not explicit.",
        rationale="One fact is supported but the requested relation is absent.",
    )
    service = EvidenceSufficiencyService(_SufficiencyAssessor(assessment))

    batch = service.evaluate(
        (_task(),),
        (_candidate_placeholder(),),
        (_validated(),),
    )

    assert batch.evidence_sufficient is False
    assert batch.tasks[0].status == "running"
    assert batch.gaps[0].kind == "missing_relation"
    assert batch.gaps[0].description == assessment.missing_relation


def test_sufficient_task_requires_and_records_supported_evidence() -> None:
    assessment = TaskSufficiencyAssessment(
        task_id="T1",
        status="sufficient",
        supporting_evidence_ids=("E1",),
        rationale="The evidence directly covers the complete objective.",
    )
    service = EvidenceSufficiencyService(_SufficiencyAssessor(assessment))

    batch = service.evaluate(
        (_task(),),
        (_candidate_placeholder(),),
        (_validated(),),
    )

    assert batch.evidence_sufficient is True
    assert batch.tasks[0].status == "completed"
    assert batch.gaps == ()


def test_unattempted_dependency_task_is_not_assessed_as_a_gap() -> None:
    assessment = TaskSufficiencyAssessment(
        task_id="T1",
        status="sufficient",
        supporting_evidence_ids=("E1",),
        rationale="The predecessor has complete supported evidence.",
    )
    service = EvidenceSufficiencyService(_SufficiencyAssessor(assessment))
    predecessor = _task().model_copy(update={"status": "running"})
    dependent = ResearchTask(
        task_id="T2",
        question="How is the established result used?",
        objective="Establish the dependent relation.",
        dependencies=("T1",),
    )

    batch = service.evaluate(
        (predecessor, dependent),
        (_candidate_placeholder(),),
        (_validated(),),
    )

    assert tuple(item.task_id for item in batch.assessments) == ("T1",)
    assert tuple(item.status for item in batch.tasks) == (
        "completed",
        "pending",
    )
    assert batch.gaps == ()
    assert batch.evidence_sufficient is False


def test_sufficiency_handler_rejects_unvalidated_evidence_reference() -> None:
    assessment = TaskSufficiencyAssessment(
        task_id="T1",
        status="sufficient",
        supporting_evidence_ids=("UNKNOWN",),
        rationale="Invalid synthetic assessment.",
    )
    service = EvidenceSufficiencyService(_SufficiencyAssessor(assessment))
    state = new_research_state(
        research_id="R1",
        user_question="What starts registration?",
    )
    state["research_tasks"] = (_task(),)
    state["validated_evidence"] = (_validated(),)

    try:
        EvidenceSufficiencyHandler(service)(state)
    except ValueError as exc:
        assert "ineligible evidence" in str(exc)
    else:
        raise AssertionError("ineligible evidence reference was accepted")


class _SynthesisGenerator:
    def __init__(self, evidence_id: str = "E1") -> None:
        self.evidence_id = evidence_id

    def synthesize(self, question, tasks, evidence):
        return EvidenceSynthesis(
            answer="The UE initiates the registration procedure.",
            claims=(EvidenceBackedClaim(
                claim_id="C1",
                text="The UE initiates registration.",
                evidence_ids=(self.evidence_id,),
            ),),
        )


def _sufficient_assessment() -> TaskSufficiencyAssessment:
    return TaskSufficiencyAssessment(
        task_id="T1",
        status="sufficient",
        supporting_evidence_ids=("E1",),
        rationale="The complete objective is supported.",
    )


def test_synthesis_rejects_claims_with_unvalidated_evidence(tmp_path: Path) -> None:
    service = EvidenceSynthesisService(_SynthesisGenerator("UNKNOWN"))

    try:
        service.synthesize(
            "What starts registration?",
            (_task(),),
            (_candidate(tmp_path),),
            (_validated(),),
            (_sufficient_assessment(),),
        )
    except ValueError as exc:
        assert "ineligible evidence" in str(exc)
    else:
        raise AssertionError("synthesis accepted an unvalidated evidence ID")


def test_synthesis_handler_records_claim_to_evidence_chain(tmp_path: Path) -> None:
    state = new_research_state(
        research_id="R1",
        user_question="What starts registration?",
    )
    state["research_tasks"] = (
        _task().model_copy(update={"status": "completed"}),
    )
    state["candidate_evidence"] = (_candidate(tmp_path),)
    state["validated_evidence"] = (_validated(),)
    state["sufficiency_assessments"] = (_sufficient_assessment(),)
    state["evidence_sufficient"] = True

    update = EvidenceSynthesisHandler(
        EvidenceSynthesisService(_SynthesisGenerator())
    )(state)

    assert update["stop_reason"] == "completed"
    assert update["final_answer"].startswith("The UE")
    assert update["synthesis"].claims[0].evidence_ids == ("E1",)


def test_validated_conflict_becomes_source_conflict_gap(tmp_path: Path) -> None:
    first = _candidate(
        tmp_path,
        line_end=2,
        excerpt="The UE shall initiate the registration procedure.",
    )
    second = _candidate(
        tmp_path,
        evidence_id="E2",
        chunk_id="C2",
        line_start=3,
        line_end=3,
        excerpt="The network shall process the request.",
    )
    validation = EvidenceValidationService(
        SourceEvidenceValidator(tmp_path),
        _SupportingAssessor(),
    )

    class ConflictAssessor:
        def detect(self, task, evidence):
            return (EvidenceConflict(
                conflict_id="X1",
                task_id=task.task_id,
                evidence_ids=("E1", "E2"),
                kind="version_conflict",
                description="The applicable version conditions disagree.",
            ),)

    state = new_research_state(
        research_id="R1",
        user_question="What starts registration?",
    )
    state["research_tasks"] = (_task(),)
    state["candidate_evidence"] = (first, second)
    validation_update = EvidenceValidationHandler(
        validation,
        EvidenceConflictService(ConflictAssessor()),
    )(state)
    state.update(validation_update)

    sufficiency = EvidenceSufficiencyService(
        _SufficiencyAssessor(_sufficient_assessment())
    )
    update = EvidenceSufficiencyHandler(sufficiency)(state)

    assert update["evidence_sufficient"] is False
    assert update["sufficiency_assessments"][0].status == "conflict"
    assert update["gaps"][0].kind == "source_conflict"
    assert update["research_tasks"][0].status == "blocked"


def test_real_evidence_handlers_complete_single_research_graph(
    tmp_path: Path,
) -> None:
    task = _task()
    plan = ResearchPlan(plan_id="P1", tasks=(task,))
    validator = EvidenceValidationHandler(EvidenceValidationService(
        SourceEvidenceValidator(tmp_path),
        _SupportingAssessor(),
    ))
    sufficiency = EvidenceSufficiencyHandler(EvidenceSufficiencyService(
        _SufficiencyAssessor(_sufficient_assessment())
    ))
    synthesizer = EvidenceSynthesisHandler(EvidenceSynthesisService(
        _SynthesisGenerator()
    ))
    workflow = build_single_research_workflow(ResearchNodeHandlers(
        router=lambda state: {"route": "plan"},
        planner=lambda state: {
            "research_plan": plan,
            "research_tasks": plan.tasks,
        },
        researcher=lambda state: {
            "candidate_evidence": (_candidate(tmp_path),),
            "last_researcher_stop_reason": "agent_finish",
        },
        validator=validator,
        sufficiency_judge=sufficiency,
        replanner=lambda state: {},
        synthesizer=synthesizer,
    ))

    result = workflow.invoke(
        new_research_state(
            research_id="R-E2E",
            user_question="What starts registration?",
        ),
        thread_id="evidence-e2e-test",
    )

    assert result["stop_reason"] == "completed"
    assert result["source_validations"][0].valid is True
    assert result["validated_evidence"][0].validation_status == "supported"
    assert result["sufficiency_assessments"][0].status == "sufficient"
    assert result["synthesis"].claims[0].evidence_ids == ("E1",)

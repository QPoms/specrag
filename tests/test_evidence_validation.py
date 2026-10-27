from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Barrier

import pytest

from specrag.infrastructure.receipt_capture import (
    record_agent_receipt,
    record_failed_agent_call,
)
from specrag.infrastructure.structured_agent import (
    CodexAgentError,
    capture_agent_receipts,
)
from specrag.evidence import (
    EvidenceSufficiencyService,
    EvidenceSynthesisService,
    SourceEvidenceValidator,
    SourceValidationService,
    deduplicate_candidates,
    publication_decision,
)
from specrag.models import (
    CandidateEvidence,
    EvidenceSynthesis,
    ResearchBudget,
    ResearchPlan,
    ResearchTask,
    SourceValidationResult,
    TaskSufficiencyAssessment,
    TaskEvidenceSet,
    ValidatedEvidence,
)
from specrag.orchestration import (
    EvidenceSufficiencyHandler,
    EvidenceSynthesisHandler,
    SourceValidationHandler,
    ResearchNodeHandlers,
    build_single_research_workflow,
    new_research_state,
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


def test_invalid_source_is_rejected_deterministically(
    tmp_path: Path,
) -> None:
    service = SourceValidationService(SourceEvidenceValidator(tmp_path))
    candidate = _candidate(tmp_path, document_sha256="B" * 64)

    batch = service.validate((_task(),), (candidate,))

    assert batch.source_validations[0].document_hash_valid is False


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
    service = SourceValidationService(SourceEvidenceValidator(tmp_path))
    state = new_research_state(
        research_id="R1",
        user_question="What starts registration?",
    )
    state["research_tasks"] = (_task(),)
    state["candidate_evidence"] = (_candidate(tmp_path),)

    update = SourceValidationHandler(service)(state)

    assert update["source_validations"][0].valid is True
    assert "validated_evidence" not in update
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

    def assess(self, task, evidence, original_question=None):
        return self.assessment


class _TrackingSufficiencyAssessor:
    def __init__(self) -> None:
        self.task_ids: list[str] = []

    def assess(self, task, evidence, original_question=None):
        self.task_ids.append(task.task_id)
        return TaskSufficiencyAssessment(
            task_id=task.task_id,
            status="sufficient",
            supporting_evidence_ids=tuple(item.evidence_id for item in evidence),
            rationale="The task has sufficient source evidence.",
        )


class _RecordingEvidenceAssessor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def assess(self, task, evidence, original_question=None):
        evidence_ids = tuple(item.evidence_id for item in evidence)
        self.calls.append(evidence_ids)
        return TaskSufficiencyAssessment(
            task_id=task.task_id,
            status="partial",
            supporting_evidence_ids=evidence_ids[:1],
            missing_relation="One requested relation remains unresolved.",
            rationale="The current evidence only partially supports the task.",
        )


def _source_validation(
    evidence_id: str = "E1",
    *,
    valid: bool = True,
) -> SourceValidationResult:
    return SourceValidationResult(
        evidence_id=evidence_id,
        valid=valid,
        path_allowed=valid,
        document_hash_valid=valid,
        locator_valid=valid,
        excerpt_valid=valid,
        errors=() if valid else ("invalid source",),
    )


def test_sufficiency_handler_does_not_reassess_completed_tasks() -> None:
    assessor = _TrackingSufficiencyAssessor()
    first_task = _task().model_copy(update={"status": "completed"})
    second_task = ResearchTask(
        task_id="T2",
        question="What does the network do?",
        objective="Find the network behavior.",
        status="running",
    )
    first_candidate = _candidate_placeholder()
    second_candidate = first_candidate.model_copy(update={
        "evidence_id": "E2",
        "task_id": "T2",
        "chunk_id": "C2",
        "excerpt": "The network shall process the request.",
    })
    first_assessment = TaskSufficiencyAssessment(
        task_id="T1",
        status="sufficient",
        supporting_evidence_ids=("E1",),
        rationale="The first task was already supported.",
    )
    state = new_research_state(
        research_id="R-INCREMENTAL-JUDGE",
        user_question="Explain both registration behaviors.",
    )
    state["research_tasks"] = (first_task, second_task)
    state["candidate_evidence"] = (first_candidate, second_candidate)
    state["source_validations"] = (
        _source_validation("E1"),
        _source_validation("E2"),
    )
    state["validated_evidence"] = (_validated(),)
    state["sufficiency_assessments"] = (first_assessment,)

    update = EvidenceSufficiencyHandler(
        EvidenceSufficiencyService(assessor)
    )(state)

    assert assessor.task_ids == ["T2"]
    assert [item.task_id for item in update["sufficiency_assessments"]] == [
        "T1",
        "T2",
    ]
    assert {item.evidence_id for item in update["validated_evidence"]} == {
        "E1",
        "E2",
    }
    assert all(item.status == "completed" for item in update["research_tasks"])
    assert update["evidence_sufficient"] is True


def test_judge_reviews_prior_selected_plus_latest_active_not_candidate_history(
) -> None:
    assessor = _RecordingEvidenceAssessor()
    first = _candidate_placeholder()
    historical = first.model_copy(update={
        "evidence_id": "E2",
        "chunk_id": "C2",
        "line_start": 2,
        "line_end": 2,
        "excerpt": "Historical candidate that is no longer active.",
    })
    latest = first.model_copy(update={
        "evidence_id": "E3",
        "chunk_id": "C3",
        "line_start": 3,
        "line_end": 3,
        "excerpt": "New evidence retained by the latest execution.",
    })
    state = new_research_state(
        research_id="R-JUDGE-REVIEW-SET",
        user_question="Explain the complete registration relation.",
    )
    state["research_tasks"] = (_task().model_copy(update={"status": "running"}),)
    state["candidate_evidence"] = (first, historical, latest)
    state["source_validations"] = tuple(
        _source_validation(item.evidence_id) for item in state["candidate_evidence"]
    )
    state["sufficiency_assessments"] = (TaskSufficiencyAssessment(
        task_id="T1",
        status="partial",
        supporting_evidence_ids=("E1",),
        missing_relation="The remaining relation has not been shown.",
        rationale="E1 supports the established part.",
    ),)
    state["task_evidence_sets"] = (TaskEvidenceSet(
        task_id="T1",
        execution_id="EXEC-2",
        active_evidence_ids=("E3",),
    ),)

    update = EvidenceSufficiencyHandler(
        EvidenceSufficiencyService(assessor)
    )(state)

    assert assessor.calls == [("E1", "E3")]
    assert update["judge_review_records"][0].evidence_ids == ("E1", "E3")
    validation_by_id = {
        item.evidence_id: item for item in update["validated_evidence"]
    }
    assert validation_by_id["E2"].validation_status == "rejected"
    assert "latest Researcher-active" in validation_by_id["E2"].rationale


def test_unchanged_judge_review_set_reuses_previous_assessment() -> None:
    assessor = _RecordingEvidenceAssessor()
    state = new_research_state(
        research_id="R-JUDGE-NO-PROGRESS",
        user_question="Explain the complete registration relation.",
    )
    state["research_tasks"] = (_task().model_copy(update={"status": "running"}),)
    state["candidate_evidence"] = (_candidate_placeholder(),)
    state["source_validations"] = (_source_validation(),)
    state["task_evidence_sets"] = (TaskEvidenceSet(
        task_id="T1",
        execution_id="EXEC-1",
        active_evidence_ids=("E1",),
    ),)
    handler = EvidenceSufficiencyHandler(EvidenceSufficiencyService(assessor))

    first = handler(state)
    state.update(first)
    second = handler(state)

    assert assessor.calls == [("E1",)]
    assert second["judge_skipped_task_ids"] == ("T1",)
    assert second["sufficiency_assessments"] == first["sufficiency_assessments"]
    assert second["gaps"] == first["gaps"]


def test_independent_judge_assessments_execute_in_parallel() -> None:
    barrier = Barrier(2, timeout=2)

    class BarrierAssessor:
        def assess(self, task, evidence, original_question=None):
            barrier.wait()
            return TaskSufficiencyAssessment(
                task_id=task.task_id,
                status="sufficient",
                supporting_evidence_ids=tuple(
                    item.evidence_id for item in evidence
                ),
                rationale="The task has sufficient evidence.",
            )

    first_task = _task().model_copy(update={"status": "running"})
    second_task = ResearchTask(
        task_id="T2",
        question="What does the network do?",
        objective="Find the network behavior.",
        status="running",
    )
    first_candidate = _candidate_placeholder()
    second_candidate = first_candidate.model_copy(update={
        "evidence_id": "E2",
        "task_id": "T2",
        "chunk_id": "C2",
        "excerpt": "The network shall process the request.",
    })
    state = new_research_state(
        research_id="R-PARALLEL-JUDGE",
        user_question="Explain both registration behaviors.",
    )
    state["research_tasks"] = (first_task, second_task)
    state["candidate_evidence"] = (first_candidate, second_candidate)
    state["source_validations"] = (
        _source_validation("E1"),
        _source_validation("E2"),
    )
    state["task_evidence_sets"] = (
        TaskEvidenceSet(task_id="T1", active_evidence_ids=("E1",)),
        TaskEvidenceSet(task_id="T2", active_evidence_ids=("E2",)),
    )

    update = EvidenceSufficiencyHandler(
        EvidenceSufficiencyService(BarrierAssessor()),
        max_parallel_tasks=2,
    )(state)

    assert update["evidence_sufficient"] is True
    assert tuple(
        item.task_id for item in update["sufficiency_assessments"]
    ) == ("T1", "T2")


def test_failed_parallel_judge_receipt_returns_to_parent_capture() -> None:
    class FailingAssessor:
        def assess(self, task, evidence, original_question=None):
            error = CodexAgentError(
                "provider unavailable",
                failure_kind="provider_unavailable",
                retryable=True,
                attempt_count=3,
            )
            record_failed_agent_call(
                model="test",
                reasoning_effort="high",
                purpose="judge evidence sufficiency",
                prompt_version="test",
                request_sha256="a" * 64,
                latency_ms=12,
                error=error,
            )
            raise error

    state = new_research_state(
        research_id="R-FAILED-PARALLEL-JUDGE",
        user_question="Explain the complete registration relation.",
    )
    state["research_tasks"] = (_task().model_copy(update={"status": "running"}),)
    state["candidate_evidence"] = (_candidate_placeholder(),)
    state["source_validations"] = (_source_validation(),)
    state["task_evidence_sets"] = (TaskEvidenceSet(
        task_id="T1",
        active_evidence_ids=("E1",),
    ),)
    handler = EvidenceSufficiencyHandler(
        EvidenceSufficiencyService(FailingAssessor()),
        receipt_capture=capture_agent_receipts,
        receipt_recorder=record_agent_receipt,
    )

    with capture_agent_receipts() as receipts:
        with pytest.raises(CodexAgentError, match="provider unavailable"):
            handler(state)

    assert len(receipts) == 1
    assert receipts[0].outcome == "failed"
    assert receipts[0].failure_kind == "provider_unavailable"


def test_supported_item_can_still_leave_task_level_missing_relation() -> None:
    assessment = TaskSufficiencyAssessment(
        task_id="T1",
        status="partial",
        supporting_evidence_ids=("E1",),
        missing_relation="The trigger-to-procedure relation is not explicit.",
        required_proposition=(
            "The source directly links the trigger to registration."
        ),
        missing_dimensions=("condition", "action"),
        evidence_need="relationship",
        trusted_anchors=("registration", "99.999"),
        rationale="One fact is supported but the requested relation is absent.",
    )
    service = EvidenceSufficiencyService(_SufficiencyAssessor(assessment))

    batch = service.evaluate(
        (_task(),),
        (_candidate_placeholder(),),
        (_source_validation(),),
    )

    assert batch.evidence_sufficient is False
    assert batch.tasks[0].status == "running"
    assert batch.gaps[0].kind == "missing_relation"
    assert batch.gaps[0].description == assessment.missing_relation
    assert batch.gaps[0].required_proposition == assessment.required_proposition
    assert batch.gaps[0].missing_dimensions == ("condition", "action")
    assert batch.gaps[0].evidence_need == "relationship"
    assert batch.gaps[0].trusted_anchors == ("registration",)
    assert batch.gaps[0].supporting_evidence_ids == ("E1",)
    assert batch.validated_evidence[0].validation_status == "partial"


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
        (_source_validation(),),
    )

    assert batch.evidence_sufficient is True
    assert batch.tasks[0].status == "completed"
    assert batch.gaps == ()
    assert batch.validated_evidence[0].validation_status == "supported"


def test_source_invalid_candidate_never_reaches_sufficiency_assessor() -> None:
    assessment = TaskSufficiencyAssessment(
        task_id="T1",
        status="insufficient",
        missing_relation="No source-valid evidence was supplied.",
        rationale="The invalid passage cannot support the task.",
    )

    class RecordingAssessor:
        evidence = None

        def assess(self, task, evidence, original_question=None):
            self.evidence = evidence
            return assessment

    assessor = RecordingAssessor()
    batch = EvidenceSufficiencyService(assessor).evaluate(
        (_task(),),
        (_candidate_placeholder(),),
        (_source_validation(valid=False),),
    )

    assert assessor.evidence == ()
    assert batch.validated_evidence[0].source_valid is False
    assert batch.validated_evidence[0].validation_status == "rejected"


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
        (_source_validation(),),
    )

    assert tuple(item.task_id for item in batch.assessments) == ("T1",)
    assert tuple(item.status for item in batch.tasks) == (
        "completed",
        "pending",
    )
    assert batch.gaps == ()
    assert batch.evidence_sufficient is False


def test_sufficiency_handler_downgrades_unvalidated_evidence_reference() -> None:
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
    state["candidate_evidence"] = (_candidate_placeholder(),)
    state["source_validations"] = (_source_validation(),)

    update = EvidenceSufficiencyHandler(service)(state)

    assert update["evidence_sufficient"] is False
    assert update["research_tasks"][0].status == "running"
    assert update["sufficiency_assessments"][0].status == "insufficient"
    assert update["sufficiency_assessments"][0].supporting_evidence_ids == ()
    assert update["gaps"][0].evidence_need == "direct_fact"
    assert "not source-valid" in update["gaps"][0].description


class _SynthesisGenerator:
    def __init__(self, evidence_id: str = "E1") -> None:
        self.evidence_id = evidence_id

    def synthesize(self, question, tasks, evidence, **metadata):
        return EvidenceSynthesis(
            answer=(
                "The UE initiates the registration procedure. "
                f"[{self.evidence_id}]"
            ),
            evidence_ids=(self.evidence_id,),
            **metadata,
        )


class _MissingInlineCitationGenerator:
    def synthesize(self, question, tasks, evidence, **metadata):
        return EvidenceSynthesis(
            answer="The UE initiates the registration procedure.",
            evidence_ids=("E1",),
            **metadata,
        )


class _UnavailableSynthesisGenerator:
    def synthesize(self, question, tasks, evidence, **metadata):
        error = RuntimeError("provider unavailable")
        error.failure_kind = "provider_unavailable"
        raise error


def _sufficient_assessment() -> TaskSufficiencyAssessment:
    return TaskSufficiencyAssessment(
        task_id="T1",
        status="sufficient",
        supporting_evidence_ids=("E1",),
        rationale="The complete objective is supported.",
    )


def test_synthesis_rejects_unvalidated_answer_citations(tmp_path: Path) -> None:
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


def test_synthesis_handler_publishes_evidence_digest_on_provider_failure(
    tmp_path: Path,
) -> None:
    state = new_research_state(
        research_id="R-SYNTHESIS-FALLBACK",
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
        EvidenceSynthesisService(_UnavailableSynthesisGenerator())
    )(state)

    synthesis = update["synthesis"]
    assert synthesis.mode == "complete"
    assert synthesis.evidence_ids == ("E1",)
    assert "[E1]" in synthesis.answer
    assert "model synthesis step was unavailable" in synthesis.answer
    assert update["stop_reason"] == "completed"
    assert update["errors"][-1].startswith("model synthesis unavailable")


def test_partial_synthesis_contract_failure_publishes_safe_digest(
    tmp_path: Path,
) -> None:
    second_task = ResearchTask(
        task_id="T2",
        question="What exception applies?",
        objective="Find the supported part and the unresolved exception.",
        status="completed",
    )
    second_candidate = _candidate(
        tmp_path,
        evidence_id="E2",
        task_id="T2",
        chunk_id="C2",
    )
    second_validated = _validated().model_copy(update={
        "evidence_id": "E2",
        "task_id": "T2",
    })
    partial_assessment = TaskSufficiencyAssessment(
        task_id="T2",
        status="partial",
        supporting_evidence_ids=("E2",),
        missing_relation="The exception remains unsupported.",
        rationale="The base behavior is supported but the exception is not.",
    )
    state = new_research_state(
        research_id="R-PARTIAL-CONTRACT-FALLBACK",
        user_question="What starts registration and what exception applies?",
    )
    state["research_tasks"] = (
        _task().model_copy(update={"status": "completed"}),
        second_task,
    )
    state["candidate_evidence"] = (
        _candidate(tmp_path),
        second_candidate,
    )
    state["validated_evidence"] = (_validated(), second_validated)
    state["sufficiency_assessments"] = (
        _sufficient_assessment(),
        partial_assessment,
    )
    state["evidence_sufficient"] = False
    state["stop_reason"] = "tool_budget"

    update = EvidenceSynthesisHandler(
        EvidenceSynthesisService(_SynthesisGenerator())
    )(state)

    synthesis = update["synthesis"]
    assert synthesis.mode == "partial"
    assert synthesis.covered_task_ids == ("T1", "T2")
    assert synthesis.unresolved_task_ids == ("T2",)
    assert synthesis.evidence_ids == ("E1", "E2")
    assert "[E1]" in synthesis.answer
    assert "[E2]" in synthesis.answer
    assert update["stop_reason"] == "tool_budget"
    assert update["errors"][-1].startswith(
        "model synthesis violated the grounded output contract"
    )


def test_synthesis_rejects_declared_citation_missing_from_answer(
    tmp_path: Path,
) -> None:
    service = EvidenceSynthesisService(_MissingInlineCitationGenerator())

    try:
        service.synthesize(
            "What starts registration?",
            (_task(),),
            (_candidate(tmp_path),),
            (_validated(),),
            (_sufficient_assessment(),),
        )
    except ValueError as exc:
        assert "inline answer citations must match" in str(exc)
    else:
        raise AssertionError("synthesis accepted an answer without inline citations")


def test_synthesis_handler_records_answer_to_evidence_chain(tmp_path: Path) -> None:
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
    assert update["final_answer"] == (
        "The UE initiates the registration procedure. [E1]"
    )
    assert update["synthesis"].evidence_ids == ("E1",)


def test_partial_synthesis_preserves_budget_stop_and_unresolved_tasks(
    tmp_path: Path,
) -> None:
    second_task = ResearchTask(
        task_id="T2",
        question="What exception applies?",
        objective="Find the exception.",
        status="running",
    )
    partial_assessment = TaskSufficiencyAssessment(
        task_id="T1",
        status="partial",
        supporting_evidence_ids=("E1",),
        missing_relation="The exception is not established.",
        rationale="The trigger is supported but its exception is missing.",
    )
    insufficient_assessment = TaskSufficiencyAssessment(
        task_id="T2",
        status="insufficient",
        missing_relation="No source establishes the exception.",
        rationale="No supported evidence was selected.",
    )
    state = new_research_state(
        research_id="R-PARTIAL",
        user_question="What starts registration and what exception applies?",
    )
    state["research_tasks"] = (_task(), second_task)
    state["candidate_evidence"] = (_candidate(tmp_path),)
    state["validated_evidence"] = (_validated(),)
    state["sufficiency_assessments"] = (
        partial_assessment,
        insufficient_assessment,
    )
    state["evidence_sufficient"] = False
    state["stop_reason"] = "iteration_budget"

    update = EvidenceSynthesisHandler(
        EvidenceSynthesisService(_SynthesisGenerator())
    )(state)

    synthesis = update["synthesis"]
    assert update["stop_reason"] == "iteration_budget"
    assert synthesis.mode == "partial"
    assert synthesis.covered_task_ids == ("T1",)
    assert synthesis.unresolved_task_ids == ("T1", "T2")
    assert synthesis.conflicting_task_ids == ()
    assert synthesis.evidence_ids == ("E1",)
    assert "Research limitations (backend-enforced" in synthesis.answer
    assert "T1 (partial)" in synthesis.answer
    assert "T2 (insufficient)" in synthesis.answer


def test_publication_decision_separates_evidence_status_from_generation_path(
) -> None:
    task = _task()
    sufficient = _sufficient_assessment()
    full = publication_decision((task,), (_validated(),), (sufficient,))
    assert full.path == "full_synthesis"
    assert full.synthesis_mode == "complete"

    partial = sufficient.model_copy(update={
        "status": "partial",
        "missing_relation": "One condition remains unsupported.",
    })
    qualified = publication_decision((task,), (_validated(),), (partial,))
    assert qualified.path == "qualified_synthesis"
    assert qualified.synthesis_mode == "partial"

    insufficient = TaskSufficiencyAssessment(
        task_id="T1",
        status="insufficient",
        missing_relation="No supported conclusion was established.",
        rationale="No evidence was selected.",
    )
    status_only = publication_decision((task,), (), (insufficient,))
    assert status_only.path == "status_only"
    assert status_only.synthesis_mode is None

    conflict = insufficient.model_copy(update={
        "status": "conflict",
        "missing_relation": "Two applicable passages remain inconsistent.",
    })
    conflict_only = publication_decision((task,), (), (conflict,))
    assert conflict_only.path == "status_only"
    assert conflict_only.conflicting_task_ids == ("T1",)


def test_partial_synthesis_rejects_evidence_not_selected_by_judge(
    tmp_path: Path,
) -> None:
    assessment = TaskSufficiencyAssessment(
        task_id="T1",
        status="partial",
        supporting_evidence_ids=(),
        missing_relation="The trigger is not sufficiently established.",
        rationale="No evidence was selected as supporting.",
    )

    with pytest.raises(ValueError, match="status-only"):
        EvidenceSynthesisService(_SynthesisGenerator()).synthesize(
            "What starts registration?",
            (_task(),),
            (_candidate(tmp_path),),
            (_validated(),),
            (assessment,),
            mode="partial",
        )


def test_synthesis_rejects_citations_that_omit_a_task(tmp_path: Path) -> None:
    second_task = ResearchTask(
        task_id="T2",
        question="What does the network do?",
        objective="Find the network action.",
        status="completed",
    )
    second_candidate = _candidate(
        tmp_path,
        evidence_id="E2",
        task_id="T2",
        chunk_id="C2",
    )
    second_validated = _validated().model_copy(update={
        "evidence_id": "E2",
        "task_id": "T2",
    })
    second_assessment = TaskSufficiencyAssessment(
        task_id="T2",
        status="sufficient",
        supporting_evidence_ids=("E2",),
        rationale="The network action is supported.",
    )

    service = EvidenceSynthesisService(_SynthesisGenerator())
    try:
        service.synthesize(
            "How do the UE and network act?",
            (_task(), second_task),
            (_candidate(tmp_path), second_candidate),
            (_validated(), second_validated),
            (_sufficient_assessment(), second_assessment),
        )
    except ValueError as exc:
        assert "do not cover tasks: T2" in str(exc)
    else:
        raise AssertionError("synthesis accepted citations that omitted T2")


def test_sufficiency_judge_conflict_becomes_source_conflict_gap(
    tmp_path: Path,
) -> None:
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
    validation = SourceValidationService(SourceEvidenceValidator(tmp_path))

    state = new_research_state(
        research_id="R1",
        user_question="What starts registration?",
    )
    state["research_tasks"] = (_task(),)
    state["candidate_evidence"] = (first, second)
    validation_update = SourceValidationHandler(validation)(state)
    state.update(validation_update)

    sufficiency = EvidenceSufficiencyService(
        _SufficiencyAssessor(TaskSufficiencyAssessment(
            task_id="T1",
            status="conflict",
            supporting_evidence_ids=("E1", "E2"),
            missing_relation="The applicable version conditions disagree.",
            rationale="The two passages directly conflict.",
        ))
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
    source_validator = SourceValidationHandler(SourceValidationService(
        SourceEvidenceValidator(tmp_path),
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
            "last_researcher_stop_reason": "submitted_for_validation",
        },
        source_validator=source_validator,
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
    assert result["synthesis"].evidence_ids == ("E1",)


def test_budget_limited_graph_publishes_cited_partial_synthesis(
    tmp_path: Path,
) -> None:
    task = _task()
    plan = ResearchPlan(plan_id="P-PARTIAL", tasks=(task,))
    partial = TaskSufficiencyAssessment(
        task_id="T1",
        status="partial",
        supporting_evidence_ids=("E1",),
        missing_relation="The exception remains unverified.",
        rationale="The trigger is supported, but the exception is missing.",
    )
    workflow = build_single_research_workflow(ResearchNodeHandlers(
        router=lambda state: {"route": "plan"},
        planner=lambda state: {
            "research_plan": plan,
            "research_tasks": plan.tasks,
        },
        researcher=lambda state: {
            "candidate_evidence": (_candidate(tmp_path),),
            "last_researcher_stop_reason": "submitted_for_validation",
        },
        source_validator=SourceValidationHandler(SourceValidationService(
            SourceEvidenceValidator(tmp_path),
        )),
        sufficiency_judge=EvidenceSufficiencyHandler(
            EvidenceSufficiencyService(_SufficiencyAssessor(partial))
        ),
        replanner=lambda state: {},
        synthesizer=EvidenceSynthesisHandler(EvidenceSynthesisService(
            _SynthesisGenerator()
        )),
    ))

    result = workflow.invoke(
        new_research_state(
            research_id="R-PARTIAL-E2E",
            user_question="What starts registration and what exception applies?",
            budget=ResearchBudget(max_iterations=1),
        ),
        thread_id="partial-evidence-e2e-test",
    )

    assert result["stop_reason"] == "iteration_budget"
    assert result["synthesis"].mode == "partial"
    assert result["synthesis"].evidence_ids == ("E1",)
    assert result["synthesis"].unresolved_task_ids == ("T1",)

from __future__ import annotations

from specrag.agents.models import (
    AgentControlCall,
    AgentEvidenceSelection,
    AgentResearchReview,
    AgentRetrievalCall,
    AgentRetrievalTurn,
    AgentToolOutcome,
    AgenticRetrievalExecution,
)
from specrag.infrastructure.structured_agent import CodexAgentReceipt
from specrag.models import (
    CandidateEvidence,
    ResearchBudget,
    ResearchGap,
    ResearchPlan,
    ResearchTask,
    SourceValidationResult,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)
from specrag.orchestration import (
    ResearchNodeHandlers,
    SingleResearcherHandler,
    build_dependency_context,
    build_single_research_workflow,
    new_research_state,
    researcher_execution_update,
)
from specrag.orchestration.application import ResearchPlannerHandler
from specrag.retrieval.planner import (
    PlannedRetrievalExecution,
    PlannedRetrievalHit,
    PlannedRetrievalTaskResult,
)


def _hit() -> PlannedRetrievalHit:
    return PlannedRetrievalHit(
        hit_id="E-Q1:1",
        chunk_id="24.501:17.1.0:chunk-1",
        collection_rank=1,
        specification="24.501",
        version="17.1.0",
        source_path="data/3gpp/24.501/v17.1.0/source.md",
        document_sha256="A" * 64,
        heading_path=("5 Registration",),
        line_start=10,
        line_end=14,
        excerpt="The UE shall initiate the registration procedure.",
        evidence_roles=("procedure",),
        probe_ids=("P1",),
        retrieval_sources=("tool:hybrid_search",),
        rrf_score=0.02,
        route_score=0.0,
        anchor_score=0.0,
        heading_score=0.0,
        agreement_score=0.0,
        noise_penalty=0.0,
        final_score=0.02,
    )


def _execution() -> AgenticRetrievalExecution:
    call = AgentRetrievalCall(
        call_id="call-A1",
        tool="hybrid_search",
        arguments={
            "query_id": "Q1",
            "lexical_query": "registration",
            "semantic_query": "registration procedure",
            "target_specifications": ["24.501"],
            "max_results": 4,
        },
    )
    outcome = AgentToolOutcome(
        round_number=1,
        call=call,
        status="completed",
        evidence_ids=("E-Q1:1", "E-Q1:2"),
        new_evidence_ids=("E-Q1:1", "E-Q1:2"),
    )
    receipt = CodexAgentReceipt(
        codex_version="test",
        model="test",
        reasoning_effort="test",
        purpose="test",
        prompt_version="test",
        request_sha256="b" * 64,
        response_sha256="c" * 64,
        latency_ms=1,
    )
    task = PlannedRetrievalTaskResult(
        retrieval_id="RET-Q1",
        query_id="Q1",
        planner_action="planned",
        status="completed",
        hits=(_hit(),),
    )
    evidence = PlannedRetrievalExecution(
        status="completed",
        embedding_model="test",
        corpus_version_policy="highest_version_within_release",
        corpus_release=17,
        tasks=(task,),
    )
    candidate_hit = _hit().model_copy(update={
        "hit_id": "E-Q1:2",
        "chunk_id": "23.401:17.1.0:chunk-2",
        "specification": "23.401",
        "source_path": "data/3gpp/23.401/v17.1.0/source.md",
        "line_start": 20,
        "line_end": 24,
        "excerpt": "The MME forwards the bearer request to the eNodeB.",
    })
    candidate_task = task.model_copy(update={
        "hits": (*task.hits, candidate_hit),
    })
    candidate_evidence = evidence.model_copy(update={
        "tasks": (candidate_task,),
    })
    return AgenticRetrievalExecution(
        status="completed",
        stop_reason="submitted_for_validation",
        rounds_used=1,
        search_actions_used=1,
        expansion_actions_used=0,
        turns=(AgentRetrievalTurn(
            round_number=1,
            control_call=AgentControlCall(
                call_id="call-control",
                name="update_research_state",
                review=AgentResearchReview(
                    assessment="Evidence found.",
                    evidence_selections=(AgentEvidenceSelection(
                        query_id="Q1",
                        retained_evidence_ids=("E-Q1:1",),
                    ),),
                    unresolved_gaps=(),
                ),
            ),
            receipt=receipt,
            outcomes=(outcome,),
        ),),
        evidence_execution=evidence,
        candidate_evidence_execution=candidate_evidence,
    )


def _candidate(
    evidence_id: str,
    *,
    task_id: str = "T1",
    line_start: int = 10,
) -> CandidateEvidence:
    return CandidateEvidence(
        evidence_id=evidence_id,
        task_id=task_id,
        chunk_id=f"24.501:17.1.0:{evidence_id}",
        specification="24.501",
        release=17,
        version="17.1.0",
        source_path="data/3gpp/24.501/v17.1.0/source.md",
        document_sha256="A" * 64,
        heading_path=("5 Registration",),
        line_start=line_start,
        line_end=line_start + 1,
        excerpt=f"Supported dependency evidence {evidence_id}.",
    )


def _validated(
    evidence_id: str,
    *,
    task_id: str = "T1",
    status: str = "supported",
) -> ValidatedEvidence:
    return ValidatedEvidence(
        evidence_id=evidence_id,
        task_id=task_id,
        source_valid=True,
        supports_target=status == "supported",
        validation_status=status,
        rationale=f"Validation for {evidence_id}.",
    )


def test_researcher_execution_preserves_provenance_and_accounting() -> None:
    state = new_research_state(
        research_id="R1",
        user_question="What starts registration?",
        budget=ResearchBudget(max_tool_calls=3),
    )
    state["research_tasks"] = (ResearchTask(
        task_id="T1",
        question="What starts registration?",
        objective="Find the normative trigger.",
    ),)

    update = researcher_execution_update(
        state,
        _execution(),
        query_to_task={"Q1": "T1"},
    )

    candidate = update["candidate_evidence"][0]
    assert candidate.task_id == "T1"
    assert candidate.chunk_id == "24.501:17.1.0:chunk-1"
    assert candidate.release == 17
    assert candidate.document_sha256 == "A" * 64
    assert update["tool_calls"][0].evidence_ids[0] == candidate.evidence_id
    assert len(update["tool_calls"][0].evidence_ids) == 2
    assert len(update["tool_calls"][0].observed_evidence) == 2
    assert len(update["tool_calls"][0].new_evidence_ids) == 2
    assert {
        item.specification
        for item in update["tool_calls"][0].observed_evidence
    } == {"23.401", "24.501"}
    assert update["tool_calls"][0].query_id == "Q1"
    assert update["tool_calls"][0].lexical_query == "registration"
    assert update["tool_calls"][0].semantic_query == "registration procedure"
    assert update["tool_calls"][0].target_specifications == ("24.501",)
    assert update["budget"].tool_calls_used == 1
    assert update["last_researcher_stop_reason"] == "submitted_for_validation"
    selection = update["researcher_evidence_selections"][0]
    assert selection.task_id == "T1"
    assert selection.retained_evidence_ids == (candidate.evidence_id,)


def test_planner_handler_reuses_a_frozen_plan() -> None:
    class UnexpectedPlanner:
        def plan(self, **kwargs):
            raise AssertionError("the frozen plan must bypass live planning")

    task = ResearchTask(
        task_id="T1",
        question="Establish the frozen objective.",
        objective="Establish the frozen objective.",
    )
    plan = ResearchPlan(plan_id="P-frozen", tasks=(task,))
    state = new_research_state(
        research_id="R-frozen",
        user_question="Use the frozen plan.",
    )
    state["research_plan"] = plan
    state["research_tasks"] = plan.tasks

    update = ResearchPlannerHandler(UnexpectedPlanner())(state)

    assert update == {"research_plan": plan, "research_tasks": plan.tasks}


def test_single_researcher_handler_runs_one_dependency_ready_task() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.tasks = ()
            self.max_tool_calls = None
            self.shared_context = {}

        def run(self, tasks, *, shared_context, max_tool_calls):
            self.tasks = tasks
            self.max_tool_calls = max_tool_calls
            self.shared_context = shared_context
            return _execution(), {"Q1": tasks[0].task_id}

    state = new_research_state(
        research_id="R1",
        user_question="What starts registration?",
        budget=ResearchBudget(max_tool_calls=3),
    )
    state["research_tasks"] = (
        ResearchTask(
            task_id="T1",
            question="What starts registration?",
            objective="Find the normative trigger.",
        ),
        ResearchTask(
            task_id="T2",
            question="What follows registration?",
            objective="Find the dependent step.",
            dependencies=("T1",),
        ),
    )
    service = FakeService()

    update = SingleResearcherHandler(service)(state)

    assert tuple(task.task_id for task in service.tasks) == ("T1",)
    assert service.max_tool_calls == 3
    assert service.shared_context["dependency_context"] == ()
    assert update["research_tasks"][0].status == "running"
    assert update["research_tasks"][1].status == "pending"


def test_dependent_task_receives_its_own_tool_budget() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.max_tool_calls = None

        def run(self, tasks, *, shared_context, max_tool_calls):
            self.max_tool_calls = max_tool_calls
            return _execution(), {"Q1": tasks[0].task_id}

    budget = ResearchBudget(max_tool_calls=3).record_tool_calls("T1", 3)
    state = new_research_state(
        research_id="R-task-budget",
        user_question="Resolve a dependent task.",
        budget=budget,
    )
    state["research_tasks"] = (
        ResearchTask(
            task_id="T1",
            question="Establish the predecessor.",
            objective="Establish the predecessor.",
            status="completed",
        ),
        ResearchTask(
            task_id="T2",
            question="Use the predecessor result.",
            objective="Establish the dependent result.",
            dependencies=("T1",),
        ),
    )
    state["candidate_evidence"] = (_candidate("E1"),)
    state["validated_evidence"] = (_validated("E1"),)
    state["sufficiency_assessments"] = (TaskSufficiencyAssessment(
        task_id="T1",
        status="sufficient",
        supporting_evidence_ids=("E1",),
        rationale="The predecessor is established.",
    ),)
    service = FakeService()

    update = SingleResearcherHandler(service)(state)

    assert service.max_tool_calls == 3
    assert update["budget"].tool_calls_used == 4
    assert update["budget"].tool_calls_used_for("T1") == 3
    assert update["budget"].tool_calls_used_for("T2") == 1


def test_judge_gap_receives_independent_corrective_budget() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.max_tool_calls = None
            self.shared_context = {}
            self.initial_evidence = ()
            self.initial_active_evidence_ids = ()

        def run(
            self,
            tasks,
            *,
            shared_context,
            max_tool_calls,
            initial_evidence=(),
            initial_active_evidence_ids=(),
        ):
            self.max_tool_calls = max_tool_calls
            self.shared_context = shared_context
            self.initial_evidence = initial_evidence
            self.initial_active_evidence_ids = initial_active_evidence_ids
            return _execution(), {"Q1": tasks[0].task_id}

    gap = ResearchGap(
        gap_id="G1",
        task_id="T1",
        kind="missing_relation",
        description="The trigger relation remains unsupported.",
        required_proposition=(
            "The source directly states the trigger-to-procedure relation."
        ),
        missing_dimensions=("condition", "action"),
        evidence_need="relationship",
        trusted_anchors=("registration",),
        supporting_evidence_ids=("E1",),
    )
    budget = ResearchBudget(
        max_tool_calls=1,
        max_gap_tool_calls=2,
    ).record_tool_calls("T1", 1)
    state = new_research_state(
        research_id="R-gap-budget",
        user_question="What starts registration?",
        budget=budget,
    )
    state["research_tasks"] = (ResearchTask(
        task_id="T1",
        question="What starts registration?",
        objective="Find the normative trigger.",
        status="running",
    ),)
    state["gaps"] = (gap,)
    state["candidate_evidence"] = (_candidate("E1"),)
    state["source_validations"] = (SourceValidationResult(
        evidence_id="E1",
        valid=True,
        path_allowed=True,
        document_hash_valid=True,
        locator_valid=True,
        excerpt_valid=True,
    ),)
    state["validated_evidence"] = (_validated("E1"),)
    service = FakeService()

    update = SingleResearcherHandler(service)(state)

    assert service.max_tool_calls == 2
    assert service.shared_context["gap_retrieval_mode"] == "evidence_gap"
    assert service.shared_context["unresolved_gaps"][0][
        "required_proposition"
    ].startswith("The source directly states")
    assert service.shared_context["prior_tool_calls"] == ()
    assert tuple(item.evidence_id for item in service.initial_evidence) == (
        "E1",
    )
    assert service.initial_active_evidence_ids == ("E1",)
    assert update["budget"].initial_tool_calls_used_for("T1") == 1
    assert update["budget"].gap_tool_calls_used_for("T1") == 1
    assert update["budget"].tool_calls_used_for("T1") == 2


def test_dependency_context_contains_only_selected_supported_evidence() -> None:
    state = new_research_state(
        research_id="R-dependency",
        user_question="How do the dependent findings connect?",
    )
    dependency = ResearchTask(
        task_id="T1",
        question="Which node is involved?",
        objective="Establish the relevant node.",
        status="completed",
    )
    dependent = ResearchTask(
        task_id="T2",
        question="How does that node transform the message?",
        objective="Validate the transformation by the identified node.",
        dependencies=("T1",),
    )
    selected = _candidate("E1")
    unselected = _candidate("E2", line_start=20)
    partial = _candidate("E3", line_start=30)
    state["research_tasks"] = (dependency, dependent)
    state["candidate_evidence"] = (selected, unselected, partial)
    state["validated_evidence"] = (
        _validated("E1"),
        _validated("E2"),
        _validated("E3", status="partial"),
    )
    state["sufficiency_assessments"] = (TaskSufficiencyAssessment(
        task_id="T1",
        status="sufficient",
        supporting_evidence_ids=("E1", "E3"),
        rationale="The supported evidence establishes the relevant node.",
    ),)

    context = build_dependency_context(state, dependent)

    assert len(context) == 1
    assert context[0]["task_id"] == "T1"
    assert context[0]["question"] == dependency.question
    assert context[0]["objective"] == dependency.objective
    assert context[0]["sufficiency_rationale"].startswith("The supported")
    assert [
        item["evidence_id"]
        for item in context[0]["supporting_evidence"]
    ] == ["E1"]
    assert context[0]["supporting_evidence"][0]["excerpt"] == selected.excerpt
    assert "E2" not in str(context)
    assert "E3" not in str(context)


def test_single_researcher_receives_validated_dependency_context() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.shared_context = {}

        def run(self, tasks, *, shared_context, max_tool_calls):
            self.shared_context = shared_context
            return _execution(), {"Q1": tasks[0].task_id}

    state = new_research_state(
        research_id="R-dependent-worker",
        user_question="How is the message transformed?",
        budget=ResearchBudget(max_tool_calls=3),
    )
    dependency = ResearchTask(
        task_id="T1",
        question="Which node is involved?",
        objective="Establish the relevant node.",
        status="completed",
    )
    dependent = ResearchTask(
        task_id="T2",
        question="How does that node transform the message?",
        objective="Validate the transformation by the identified node.",
        dependencies=("T1",),
    )
    state["research_tasks"] = (dependency, dependent)
    state["candidate_evidence"] = (_candidate("E1"),)
    state["validated_evidence"] = (_validated("E1"),)
    state["sufficiency_assessments"] = (TaskSufficiencyAssessment(
        task_id="T1",
        status="sufficient",
        supporting_evidence_ids=("E1",),
        rationale="The node is established.",
    ),)
    service = FakeService()

    SingleResearcherHandler(service)(state)

    dependency_context = service.shared_context["dependency_context"]
    assert dependency_context[0]["task_id"] == "T1"
    assert dependency_context[0]["supporting_evidence"][0][
        "evidence_id"
    ] == "E1"


def test_single_researcher_handler_runs_inside_state_graph() -> None:
    class FakeService:
        def run(self, tasks, *, shared_context, max_tool_calls):
            return _execution(), {"Q1": tasks[0].task_id}

    task = ResearchTask(
        task_id="T1",
        question="What starts registration?",
        objective="Find the normative trigger.",
    )
    plan = ResearchPlan(plan_id="P1", tasks=(task,))

    handlers = ResearchNodeHandlers(
        router=lambda state: {"route": "plan"},
        planner=lambda state: {
            "research_plan": plan,
            "research_tasks": plan.tasks,
        },
        researcher=SingleResearcherHandler(FakeService()),
        source_validator=lambda state: {},
        sufficiency_judge=lambda state: {"evidence_sufficient": True},
        replanner=lambda state: {},
        synthesizer=lambda state: {
            "final_answer": "Grounded answer.",
            "stop_reason": "completed",
        },
    )
    workflow = build_single_research_workflow(handlers)

    result = workflow.invoke(
        new_research_state(
            research_id="R1",
            user_question="What starts registration?",
            budget=ResearchBudget(max_tool_calls=3),
        ),
        thread_id="adapter-graph-test",
    )

    assert result["iteration"] == 1
    assert result["budget"].tool_calls_used == 1
    assert result["candidate_evidence"]
    assert result["tool_calls"]
    assert result["final_answer"] == "Grounded answer."
    assert result["stop_reason"] == "completed"
    assert workflow.checkpoint(thread_id="adapter-graph-test").values


def test_insufficient_evidence_routes_gap_back_to_researcher() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.contexts = []

        def run(
            self,
            tasks,
            *,
            shared_context,
            max_tool_calls,
            initial_evidence=(),
            initial_active_evidence_ids=(),
        ):
            self.contexts.append(shared_context)
            return _execution(), {"Q1": tasks[0].task_id}

    task = ResearchTask(
        task_id="T1",
        question="What starts registration?",
        objective="Find the normative trigger.",
    )
    plan = ResearchPlan(plan_id="P1", tasks=(task,))
    gap = ResearchGap(
        gap_id="G1",
        task_id="T1",
        kind="missing_relation",
        description="The trigger-to-procedure relation is missing.",
    )
    service = FakeService()

    def judge(state):
        if state["iteration"] == 1:
            return {"evidence_sufficient": False, "gaps": (gap,)}
        return {
            "evidence_sufficient": True,
            "gaps": (gap.model_copy(update={"resolved": True}),),
        }

    workflow = build_single_research_workflow(ResearchNodeHandlers(
        router=lambda state: {"route": "plan"},
        planner=lambda state: {
            "research_plan": plan,
            "research_tasks": plan.tasks,
        },
        researcher=SingleResearcherHandler(service),
        source_validator=lambda state: {},
        sufficiency_judge=judge,
        replanner=lambda state: {},
        synthesizer=lambda state: {
            "final_answer": "Grounded answer after gap retrieval.",
            "stop_reason": "completed",
        },
    ))

    result = workflow.invoke(
        new_research_state(
            research_id="R2",
            user_question="What starts registration?",
            budget=ResearchBudget(max_tool_calls=3),
        ),
        thread_id="gap-replan-test",
    )

    assert result["iteration"] == 2
    assert result["budget"].replans_used == 1
    assert result["budget"].tool_calls_used == 2
    assert len(service.contexts) == 2
    assert service.contexts[1]["unresolved_gaps"][0]["gap_id"] == "G1"
    assert result["gaps"][0].resolved is True

from __future__ import annotations

from specrag.agents.models import (
    AgentControlCall,
    AgentResearchReview,
    AgentRetrievalCall,
    AgentRetrievalTurn,
    AgentToolOutcome,
    AgenticRetrievalExecution,
)
from specrag.infrastructure.structured_agent import CodexAgentReceipt
from specrag.models import (
    ResearchBudget,
    ResearchGap,
    ResearchPlan,
    ResearchTask,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)
from specrag.orchestration.application import GapReplanner
from specrag.orchestration import (
    ParallelResearcherHandler,
    ResearchNodeHandlers,
    build_parallel_research_workflow,
    new_parallel_research_state,
)
from specrag.retrieval.planner import (
    PlannedRetrievalExecution,
    PlannedRetrievalHit,
    PlannedRetrievalTaskResult,
)


def _execution(task_id: str) -> AgenticRetrievalExecution:
    query_id = f"Q-{task_id}"
    hit_id = f"E-{task_id}"
    call = AgentRetrievalCall(
        call_id=f"call-{task_id}",
        tool="hybrid_search",
        arguments={
            "query_id": query_id,
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
        evidence_ids=(hit_id,),
        new_evidence_ids=(hit_id,),
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
    hit = PlannedRetrievalHit(
        hit_id=hit_id,
        chunk_id=f"24.501:17.1.0:{task_id}",
        collection_rank=1,
        specification="24.501",
        version="17.1.0",
        source_path="data/3gpp/24.501/v17.1.0/source.md",
        document_sha256="A" * 64,
        heading_path=("5 Registration",),
        line_start=10,
        line_end=10,
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
    return AgenticRetrievalExecution(
        status="completed",
        stop_reason="submitted_for_validation",
        rounds_used=1,
        search_actions_used=1,
        expansion_actions_used=0,
        turns=(AgentRetrievalTurn(
            round_number=1,
            control_call=AgentControlCall(
                call_id=f"control-{task_id}",
                name="update_research_state",
                review=AgentResearchReview(
                    assessment="Evidence found.",
                    evidence_selections=(),
                    unresolved_gaps=(),
                ),
            ),
            receipt=receipt,
            outcomes=(outcome,),
        ),),
        evidence_execution=PlannedRetrievalExecution(
            status="completed",
            embedding_model="test",
            corpus_version_policy="highest_version_within_release",
            corpus_release=17,
            tasks=(PlannedRetrievalTaskResult(
                retrieval_id=f"RET-{task_id}",
                query_id=query_id,
                planner_action="planned",
                status="completed",
                hits=(hit,),
            ),),
        ),
    )


class _FakeService:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.contexts: dict[str, dict[str, object]] = {}

    def run(self, tasks, *, shared_context, max_tool_calls):
        task_id = tasks[0].task_id
        self.calls.append(task_id)
        self.contexts[task_id] = shared_context
        return _execution(task_id), {f"Q-{task_id}": task_id}


def test_parallel_workflow_fans_out_independent_tasks_and_fans_in() -> None:
    service = _FakeService()
    task_a = ResearchTask(
        task_id="T1",
        question="Q1",
        objective="O1",
    )
    task_b = ResearchTask(
        task_id="T2",
        question="Q2",
        objective="O2",
    )
    plan = ResearchPlan(plan_id="P1", tasks=(task_a, task_b))
    workflow = build_parallel_research_workflow(
        ResearchNodeHandlers(
            router=lambda state: {"route": "plan"},
            planner=lambda state: {
                "research_plan": plan,
                "research_tasks": plan.tasks,
            },
            researcher=lambda state: {},
            source_validator=lambda state: {},
            sufficiency_judge=lambda state: {
                "evidence_sufficient": True,
                "research_tasks": tuple(
                    task.model_copy(update={"status": "completed"})
                    for task in state["research_tasks"]
                ),
            },
            replanner=lambda state: {},
            synthesizer=lambda state: {
                "final_answer": "Grounded parallel answer.",
                "stop_reason": "completed",
            },
        ),
        ParallelResearcherHandler(service),
    )

    result = workflow.invoke(
        new_parallel_research_state(
            research_id="parallel-test",
            user_question="What are the two independent facts?",
            budget=ResearchBudget(max_tool_calls=1),
        ),
        thread_id="parallel-test",
    )

    assert sorted(service.calls) == ["T1", "T2"]
    assert result["iteration"] == 1
    assert result["budget"].tool_calls_used == 2
    assert result["budget"].tool_calls_used_for("T1") == 1
    assert result["budget"].tool_calls_used_for("T2") == 1
    assert len(result["candidate_evidence"]) == 2
    assert result["final_answer"] == "Grounded parallel answer."


def test_parallel_workflow_defers_dependent_task_until_next_wave() -> None:
    service = _FakeService()
    first = ResearchTask(task_id="T1", question="Q1", objective="O1")
    dependent = ResearchTask(
        task_id="T2",
        question="Q2",
        objective="O2",
        dependencies=("T1",),
    )
    plan = ResearchPlan(plan_id="P2", tasks=(first, dependent))

    def validate(state):
        return {
            "validated_evidence": tuple(
                ValidatedEvidence(
                    evidence_id=item.evidence_id,
                    task_id=item.task_id,
                    source_valid=True,
                    supports_target=True,
                    validation_status="supported",
                    rationale="The source supports its task.",
                )
                for item in state["candidate_evidence"]
            )
        }

    def judge(state):
        evidence_by_task = {
            item.task_id: item.evidence_id
            for item in state["candidate_evidence"]
        }
        assessments = tuple(
            TaskSufficiencyAssessment(
                task_id=task.task_id,
                status="sufficient",
                supporting_evidence_ids=(evidence_by_task[task.task_id],),
                rationale="The task has supported evidence.",
            )
            for task in state["research_tasks"]
            if task.task_id in evidence_by_task
        )
        if "T2" not in evidence_by_task:
            return {
                "evidence_sufficient": False,
                "research_tasks": (
                    state["research_tasks"][0].model_copy(
                        update={"status": "completed"}
                    ),
                    state["research_tasks"][1],
                ),
                "sufficiency_assessments": assessments,
            }
        return {
            "evidence_sufficient": True,
            "research_tasks": tuple(
                task.model_copy(update={"status": "completed"})
                for task in state["research_tasks"]
            ),
            "sufficiency_assessments": assessments,
        }

    workflow = build_parallel_research_workflow(
        ResearchNodeHandlers(
            router=lambda state: {"route": "plan"},
            planner=lambda state: {
                "research_plan": plan,
                "research_tasks": plan.tasks,
            },
            researcher=lambda state: {},
            source_validator=validate,
            sufficiency_judge=judge,
            replanner=lambda state: {},
            synthesizer=lambda state: {
                "final_answer": "Multi-hop answer.",
                "stop_reason": "completed",
            },
        ),
        ParallelResearcherHandler(service),
    )

    result = workflow.invoke(
        new_parallel_research_state(
            research_id="dependency-test",
            user_question="Resolve a dependent research chain.",
            budget=ResearchBudget(max_tool_calls=1, max_replans=0),
        ),
        thread_id="dependency-test",
    )

    assert service.calls == ["T1", "T2"]
    assert result["iteration"] == 2
    assert result["budget"].tool_calls_used == 2
    assert result["budget"].tool_calls_used_for("T1") == 1
    assert result["budget"].tool_calls_used_for("T2") == 1
    assert result["budget"].replans_used == 0
    assert len(result["researcher_results"]) == 2
    assert result["researcher_results_processed"] == 2
    assert result["final_answer"] == "Multi-hop answer."
    dependency_context = service.contexts["T2"]["dependency_context"]
    assert dependency_context[0]["task_id"] == "T1"
    assert dependency_context[0]["supporting_evidence"][0][
        "evidence_id"
    ].startswith("E:T1:")


def test_incomplete_predecessor_replans_before_dependent_task() -> None:
    service = _FakeService()
    first = ResearchTask(task_id="T1", question="Q1", objective="O1")
    dependent = ResearchTask(
        task_id="T2",
        question="Q2",
        objective="O2",
        dependencies=("T1",),
    )
    plan = ResearchPlan(plan_id="P3", tasks=(first, dependent))

    def validate(state):
        return {
            "validated_evidence": tuple(
                ValidatedEvidence(
                    evidence_id=item.evidence_id,
                    task_id=item.task_id,
                    source_valid=True,
                    supports_target=True,
                    validation_status="supported",
                    rationale="The source supports its task.",
                )
                for item in state["candidate_evidence"]
            )
        }

    def judge(state):
        evidence_by_task = {
            item.task_id: item.evidence_id
            for item in state["candidate_evidence"]
        }
        if service.calls == ["T1"]:
            missing = "The predecessor still lacks its required relation."
            return {
                "evidence_sufficient": False,
                "research_tasks": (
                    state["research_tasks"][0].model_copy(
                        update={"status": "running"}
                    ),
                    state["research_tasks"][1],
                ),
                "sufficiency_assessments": (
                    TaskSufficiencyAssessment(
                        task_id="T1",
                        status="partial",
                        supporting_evidence_ids=(evidence_by_task["T1"],),
                        missing_relation=missing,
                        rationale=missing,
                    ),
                ),
                "gaps": (ResearchGap(
                    gap_id="G:T1:missing_relation",
                    task_id="T1",
                    kind="missing_relation",
                    description=missing,
                ),),
            }
        assessments = tuple(
            TaskSufficiencyAssessment(
                task_id=task_id,
                status="sufficient",
                supporting_evidence_ids=(evidence_id,),
                rationale="The task has supported evidence.",
            )
            for task_id, evidence_id in evidence_by_task.items()
        )
        completed_ids = set(evidence_by_task)
        tasks = tuple(
            task.model_copy(update={"status": "completed"})
            if task.task_id in completed_ids
            else task
            for task in state["research_tasks"]
        )
        return {
            "evidence_sufficient": "T2" in completed_ids,
            "research_tasks": tasks,
            "sufficiency_assessments": assessments,
            "gaps": (),
        }

    workflow = build_parallel_research_workflow(
        ResearchNodeHandlers(
            router=lambda state: {"route": "plan"},
            planner=lambda state: {
                "research_plan": plan,
                "research_tasks": plan.tasks,
            },
            researcher=lambda state: {},
            source_validator=validate,
            sufficiency_judge=judge,
            replanner=GapReplanner(),
            synthesizer=lambda state: {
                "final_answer": "Answer after predecessor retry.",
                "stop_reason": "completed",
            },
        ),
        ParallelResearcherHandler(service),
    )

    result = workflow.invoke(
        new_parallel_research_state(
            research_id="dependency-replan-test",
            user_question="Retry a predecessor before its dependent task.",
            budget=ResearchBudget(
                max_iterations=4,
                max_tool_calls=1,
                max_gap_tool_calls=1,
                max_replans=1,
            ),
        ),
        thread_id="dependency-replan-test",
    )

    assert service.calls == ["T1", "T1", "T2"]
    assert result["iteration"] == 3
    assert result["budget"].replans_used == 1
    assert result["budget"].tool_calls_used == 3
    assert result["budget"].initial_tool_calls_used_for("T1") == 1
    assert result["budget"].gap_tool_calls_used_for("T1") == 1
    assert result["budget"].initial_tool_calls_used_for("T2") == 1
    assert result["final_answer"] == "Answer after predecessor retry."

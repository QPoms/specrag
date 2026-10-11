from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from specrag.cli import (
    BaselineQuestion,
    CuratedRouteLabel,
    run_planning,
    run_routing,
    select_baseline_questions,
)
from specrag.infrastructure.research_planning import (
    RouteGoalAssessment,
    StructuredQuestionRouter,
    StructuredResearchPlanner,
    StructuredRouteResponse,
)
from specrag.models import (
    ResearchPlan,
    ResearchRouteDecision,
    ResearchTask,
)
from specrag.infrastructure.structured_agent import (
    CodexAgentReceipt,
)
from specrag.orchestration.application import (
    PlanningApplication,
    RoutingApplication,
)
from specrag.prompts import (
    PLANNER_INSTRUCTIONS,
    PLANNER_PROMPT_VERSION,
    ROUTER_INSTRUCTIONS,
    ROUTER_PROMPT_VERSION,
)


class _Agent:
    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.payloads: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        self.payloads.append(kwargs["input_payload"])
        return SimpleNamespace(
            output=self.outputs.pop(0),
            receipt=_receipt(prompt_version=kwargs["prompt_version"]),
        )


def _receipt(prompt_version: str = "test") -> CodexAgentReceipt:
    return CodexAgentReceipt(
        agent_runtime="llm_api",
        codex_version="api",
        model="test-model",
        reasoning_effort="high",
        purpose="route",
        prompt_version=prompt_version,
        request_sha256="a" * 64,
        response_sha256="b" * 64,
        latency_ms=1,
        sandbox="api",
    )


def _simple_assessment() -> RouteGoalAssessment:
    return RouteGoalAssessment(
        independent_goal_count=1,
        goal_relationship="single_chain",
        rationale="One coherent research task can answer the question.",
    )


def _complex_assessment() -> RouteGoalAssessment:
    return RouteGoalAssessment(
        independent_goal_count=2,
        goal_relationship="requires_synthesis",
        rationale="Distinct NAS and RRC facts must be researched and related.",
    )


def _simple_route() -> ResearchRouteDecision:
    return ResearchRouteDecision(
        route="single_task",
        complexity="simple",
        rationale="One coherent research task can answer the question.",
    )


def _complex_route() -> ResearchRouteDecision:
    return ResearchRouteDecision(
        route="decompose",
        complexity="complex",
        rationale="Distinct NAS and RRC facts must be researched and related.",
    )


def test_route_and_complexity_must_agree() -> None:
    with pytest.raises(ValueError, match="simple questions require single_task"):
        ResearchRouteDecision(
            route="decompose",
            complexity="simple",
            rationale="Inconsistent on purpose.",
        )


def test_goal_count_and_relationship_must_agree() -> None:
    with pytest.raises(ValueError, match="one independent goal"):
        RouteGoalAssessment(
            independent_goal_count=1,
            goal_relationship="requires_synthesis",
            rationale="Inconsistent on purpose.",
        )


def test_router_routes_without_performing_planner_decomposition() -> None:
    agent = _Agent(_simple_assessment())
    router = StructuredQuestionRouter(agent)

    decision = router.route("What are this service's purpose, inputs, and outputs?")

    instructions = str(agent.calls[0]["instructions"])
    assert decision.complexity == "simple"
    assert agent.calls[0]["output_type"] is RouteGoalAssessment
    assert agent.calls[0]["prompt_version"] == ROUTER_PROMPT_VERSION
    assert instructions == ROUTER_INSTRUCTIONS
    assert "planner-separable normative conclusions" in instructions
    assert "do not need to be logically independent" in instructions
    assert "one retrieval context can answer them" in instructions
    assert "purpose and format" in instructions
    assert "request inputs and response outputs" in instructions
    assert "not separate actor responsibilities" in instructions
    assert "Consecutive steps remain one goal only" in instructions
    assert "separate research entry points or normative contexts" in instructions
    assert "multiple sentences in a possible final answer" in instructions
    assert "Before deciding, apply this test" in instructions
    assert "Do not list the goals or create Planner tasks" in instructions


def test_multiple_independent_goals_map_to_complex_route() -> None:
    agent = _Agent(_complex_assessment())

    response = StructuredQuestionRouter(agent).route_with_receipt(
        "How do two independently specified procedures interact?"
    )

    assert response.output == _complex_route()
    assert response.assessment.independent_goal_count == 2


def test_simple_question_becomes_one_task_without_another_model_call() -> None:
    agent = _Agent()
    planner = StructuredResearchPlanner(agent)

    plan = planner.plan(
        research_id="R1",
        question="When does timer T3512 stop?",
        route=_simple_route(),
    )

    assert len(plan.tasks) == 1
    assert plan.tasks[0].question == "When does timer T3512 stop?"
    assert agent.payloads == []

    response = planner.plan_with_receipt(
        research_id="R1",
        question="When does timer T3512 stop?",
        route=_simple_route(),
    )

    assert response.output == plan
    assert response.mode == "deterministic"
    assert response.contract_version == PLANNER_PROMPT_VERSION
    assert response.prompt_version is None
    assert response.receipt is None
    assert agent.payloads == []


def test_complex_question_is_decomposed_with_explicit_dependencies() -> None:
    proposed = ResearchPlan(
        plan_id="model-plan",
        tasks=(
            ResearchTask(task_id="T1", question="Find NAS rule", objective="NAS"),
            ResearchTask(task_id="T2", question="Find RRC rule", objective="RRC"),
            ResearchTask(
                task_id="T3",
                question="Relate the rules",
                objective="Integration",
                dependencies=("T1", "T2"),
            ),
        ),
    )
    agent = _Agent(proposed)
    planner = StructuredResearchPlanner(agent)

    plan = planner.plan(
        research_id="R2",
        question="How do the NAS and RRC rules work together?",
        route=_complex_route(),
    )

    assert plan.plan_id == "P:R2"
    assert len(plan.tasks) == 3
    assert plan.tasks[2].dependencies == ("T1", "T2")
    assert agent.calls[0]["prompt_version"] == PLANNER_PROMPT_VERSION
    assert agent.calls[0]["instructions"] == PLANNER_INSTRUCTIONS
    assert set(agent.payloads[0]) == {"research_id", "question", "routing"}
    assert agent.payloads[0]["routing"]["complexity"] == "complex"
    assert "release" not in json.dumps(agent.payloads[0])

    response = StructuredResearchPlanner(_Agent(proposed)).plan_with_receipt(
        research_id="R2",
        question="How do the NAS and RRC rules work together?",
        route=_complex_route(),
    )
    assert response.output.plan_id == "P:R2"
    assert response.mode == "model"
    assert response.contract_version == PLANNER_PROMPT_VERSION
    assert response.prompt_version == PLANNER_PROMPT_VERSION
    assert response.receipt is not None
    assert response.receipt.prompt_version == PLANNER_PROMPT_VERSION


def test_planner_prompt_limits_scope_and_defines_dependency_context() -> None:
    assert PLANNER_PROMPT_VERSION == "research-planner/v1.3"
    assert "minimum research scope sufficient" in PLANNER_INSTRUCTIONS
    assert "completed and validated findings" in PLANNER_INSTRUCTIONS
    assert "provided as research context" in PLANNER_INSTRUCTIONS
    assert "Protocol sequence" in PLANNER_INSTRUCTIONS
    assert "complete interface or message definitions" in PLANNER_INSTRUCTIONS
    assert "merely to combine, restate, or summarize" in PLANNER_INSTRUCTIONS


def test_complex_plan_requires_canonical_ordered_task_ids() -> None:
    proposed = ResearchPlan(
        plan_id="model-plan",
        tasks=(
            ResearchTask(task_id="T1", question="Part one", objective="O1"),
            ResearchTask(task_id="T3", question="Part two", objective="O2"),
        ),
    )
    planner = StructuredResearchPlanner(_Agent(proposed))

    with pytest.raises(ValueError, match="ordered as T1, T2"):
        planner.plan(
            research_id="R2",
            question="How do the two parts work together?",
            route=_complex_route(),
        )


def test_research_plan_rejects_dependency_cycles() -> None:
    with pytest.raises(ValueError, match="cycle"):
        ResearchPlan(
            plan_id="P1",
            tasks=(
                ResearchTask(
                    task_id="T1",
                    question="Q1",
                    objective="O1",
                    dependencies=("T2",),
                ),
                ResearchTask(
                    task_id="T2",
                    question="Q2",
                    objective="O2",
                    dependencies=("T1",),
                ),
            ),
        )


def test_research_plan_rejects_forward_dependencies() -> None:
    with pytest.raises(ValueError, match="reference earlier tasks: T2"):
        ResearchPlan(
            plan_id="P1",
            tasks=(
                ResearchTask(
                    task_id="T1",
                    question="Q1",
                    objective="O1",
                    dependencies=("T2",),
                ),
                ResearchTask(task_id="T2", question="Q2", objective="O2"),
            ),
        )


def test_planning_application_passes_only_question_to_router() -> None:
    route = _complex_route()
    plan = ResearchPlan(
        plan_id="P:Q1",
        tasks=(
            ResearchTask(task_id="T1", question="Part one", objective="O1"),
            ResearchTask(task_id="T2", question="Part two", objective="O2"),
        ),
    )
    agent = _Agent(_complex_assessment(), plan)
    application = PlanningApplication(
        router=StructuredQuestionRouter(agent),
        planner=StructuredResearchPlanner(agent),
    )

    actual_route, actual_plan = application.run(
        research_id="Q1",
        question="Compare two procedures.",
    )

    assert actual_route == route
    assert len(actual_plan.tasks) == 2
    assert agent.payloads[0] == {"question": "Compare two procedures."}
    assert agent.payloads[1]["routing"]["route"] == "decompose"
    assert "expected_answer" not in json.dumps(agent.payloads)
    assert "evidence_targets" not in json.dumps(agent.payloads)


def test_planning_run_uses_reviewed_label_without_calling_router(
    tmp_path: Path,
) -> None:
    class FailingRouter:
        def route(self, question: str):
            raise AssertionError("Planner evaluation must not rerun Router")

    proposed = ResearchPlan(
        plan_id="model-plan",
        tasks=(
            ResearchTask(task_id="T1", question="Part one", objective="O1"),
            ResearchTask(task_id="T2", question="Part two", objective="O2"),
        ),
    )
    agent = _Agent(proposed)
    application = PlanningApplication(
        router=FailingRouter(),
        planner=StructuredResearchPlanner(agent),
    )

    status = run_planning(
        (BaselineQuestion(question_id="Q1", question="Compare two parts."),),
        output_directory=tmp_path,
        application=application,
        route_labels={"Q1": CuratedRouteLabel(difficulty="complex")},
    )

    artifact = json.loads((tmp_path / "Q1.json").read_text(encoding="utf-8"))
    assert status == 0
    assert artifact["schema_version"] == "specrag.planning-run/v3"
    assert artifact["route_source"] == "reviewed_label"
    assert artifact["route"]["complexity"] == "complex"
    assert artifact["planner"]["mode"] == "model"
    assert artifact["planner"]["contract_version"] == PLANNER_PROMPT_VERSION
    assert artifact["planner"]["prompt_version"] == PLANNER_PROMPT_VERSION
    assert artifact["planner"]["receipt"]["model"] == "test-model"
    assert artifact["planner"]["receipt"]["prompt_version"] == (
        PLANNER_PROMPT_VERSION
    )
    assert agent.payloads[0]["routing"]["route"] == "decompose"
    assert "release" not in json.dumps(agent.payloads)


def test_simple_planning_artifact_records_deterministic_mode(
    tmp_path: Path,
) -> None:
    application = PlanningApplication(
        router=StructuredQuestionRouter(_Agent()),
        planner=StructuredResearchPlanner(_Agent()),
    )

    status = run_planning(
        (BaselineQuestion(question_id="Q1", question="One focused rule?"),),
        output_directory=tmp_path,
        application=application,
        route_labels={"Q1": CuratedRouteLabel(difficulty="simple")},
    )

    artifact = json.loads((tmp_path / "Q1.json").read_text(encoding="utf-8"))
    assert status == 0
    assert artifact["planner"] == {
        "mode": "deterministic",
        "contract_version": PLANNER_PROMPT_VERSION,
        "prompt_version": None,
        "receipt": None,
    }


def test_question_selection_exposes_only_id_and_question(tmp_path: Path) -> None:
    records = (
        {
            "id": "dev",
            "question": "Development question?",
            "release": 17,
            "split": "development",
            "expected_answer": "secret answer",
            "expected_facts": ["secret fact"],
            "evidence_targets": [{"secret": True}],
        },
        {
            "id": "second",
            "question": "Second question?",
            "release": 17,
            "split": "development",
            "expected_answer": "hidden",
        },
    )
    question_set = tmp_path / "questions.jsonl"
    question_set.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n",
        encoding="utf-8",
    )

    selected = select_baseline_questions(
        identifiers=("dev",),
        select_all=False,
        limit=None,
        question_set=question_set,
    )

    assert selected[0].__dict__ == {
        "question_id": "dev",
        "question": "Development question?",
    }
    with pytest.raises(ValueError, match="unknown"):
        select_baseline_questions(
            identifiers=("missing",),
            select_all=False,
            limit=None,
            question_set=question_set,
        )


def test_routing_artifact_separates_model_output_from_curated_label(
    tmp_path: Path,
) -> None:
    class Router:
        def route_with_receipt(self, question: str):
            assert question == "One question?"
            return StructuredRouteResponse(
                output=_simple_route(),
                assessment=_simple_assessment(),
                receipt=_receipt(),
            )

    status = run_routing(
        (BaselineQuestion(question_id="Q1", question="One question?"),),
        output_directory=tmp_path,
        application=RoutingApplication(router=Router()),
        labels={
            "Q1": CuratedRouteLabel(
                difficulty="simple",
            )
        },
    )

    route_artifact = json.loads(
        (tmp_path / "Q1.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (tmp_path / "comparison.json").read_text(encoding="utf-8")
    )
    assert status == 0
    assert route_artifact["schema_version"] == "specrag.routing-run/v2"
    assert set(route_artifact["route"]) == {
        "route",
        "complexity",
        "rationale",
    }
    assert route_artifact["routing_assessment"] == {
        "independent_goal_count": 1,
        "goal_relationship": "single_chain",
        "rationale": "One coherent research task can answer the question.",
    }
    assert "expected_difficulty" not in json.dumps(route_artifact)
    assert "release" not in json.dumps(route_artifact)
    assert comparison["questions"][0]["route_match"] is True
    assert comparison["questions"][0]["difficulty_match"] is True


def test_select_all_includes_every_curated_question(tmp_path: Path) -> None:
    records = (
        {
            "id": "dev",
            "question": "Development question?",
            "split": "development",
        },
        {
            "id": "second",
            "question": "Second question?",
            "split": "development",
        },
    )
    question_set = tmp_path / "questions.jsonl"
    question_set.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n",
        encoding="utf-8",
    )

    selected = select_baseline_questions(
        identifiers=(),
        select_all=True,
        limit=None,
        question_set=question_set,
    )

    assert [question.question_id for question in selected] == ["dev", "second"]

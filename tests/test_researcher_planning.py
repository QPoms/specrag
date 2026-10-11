from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from specrag.agents.models import AgentRetrievalAction, AgentRetrievalDecision
from specrag.agents.planning import ResearcherPlanningService
from specrag.cli import (
    BaselineQuestion,
    CuratedRouteLabel,
    run_researcher_planning,
)
from specrag.infrastructure.research_planning import StructuredResearchPlanner
from specrag.infrastructure.researcher_controller import (
    StructuredResearcherController,
)
from specrag.infrastructure.structured_agent import CodexAgentReceipt
from specrag.models import ResearchPlan, ResearchRouteDecision, ResearchTask
from specrag.orchestration.application import ResearcherPlanningApplication
from specrag.prompts import RESEARCHER_CONTROLLER_PROMPT_VERSION
from specrag.retrieval.models import CorpusScope
from specrag.tools import ToolCapability


def _receipt(prompt_version: str) -> CodexAgentReceipt:
    return CodexAgentReceipt(
        agent_runtime="llm_api",
        codex_version="api",
        model="test-model",
        reasoning_effort="high",
        purpose="test",
        prompt_version=prompt_version,
        request_sha256="a" * 64,
        response_sha256="b" * 64,
        latency_ms=1,
        sandbox="api",
    )


class _Agent:
    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output=self.outputs.pop(0),
            receipt=_receipt(kwargs["prompt_version"]),
        )


def _action(task_id: str, *, specification: str = "24.501"):
    return AgentRetrievalAction(
        action_id=f"A-{task_id}",
        tool="hybrid_search",
        query_id=task_id,
        lexical_query="registration security context",
        semantic_query=(
            "Normative requirements for the registration security context"
        ),
        target_specifications=(specification,),
        seed_evidence_ids=(),
        max_results=5,
        rationale="Locate the governing normative requirement.",
        expected_evidence="A source clause stating the required condition.",
    )


def _decision(task_id: str, *, specification: str = "24.501"):
    return AgentRetrievalDecision(
        assessment="Search directly for the normative rule.",
        actions=(_action(task_id, specification=specification),),
    )


def _researcher(*decisions: AgentRetrievalDecision) -> tuple[
    ResearcherPlanningService,
    _Agent,
]:
    agent = _Agent(*decisions)
    capability = ToolCapability(
        name="hybrid_search",
        description="Combined lexical and semantic retrieval.",
        required_fields=("query", "semantic_query"),
        optional_fields=("target_specifications", "limit"),
        execution_kind="model_scored",
    )
    return ResearcherPlanningService(
        controller_agent=StructuredResearcherController(agent),
        corpus_scope=CorpusScope(release=17),
        available_specifications=("24.501", "38.331"),
        available_tools=(capability,),
        max_rounds=3,
        max_search_actions=4,
        max_expansion_actions=2,
        max_total_actions=6,
        hits_per_search=5,
        max_evidence_per_question=12,
    ), agent


def test_planning_only_artifact_contains_validated_calls_without_execution(
    tmp_path: Path,
) -> None:
    researcher, controller_agent = _researcher(_decision("T1"))
    application = ResearcherPlanningApplication(
        planner=StructuredResearchPlanner(_Agent()),
        researcher=researcher,
        release=17,
    )

    status = run_researcher_planning(
        (BaselineQuestion(question_id="Q1", question="One focused rule?"),),
        output_directory=tmp_path,
        route_labels={"Q1": CuratedRouteLabel(difficulty="simple")},
        application=application,
    )

    artifact = json.loads((tmp_path / "Q1.json").read_text(encoding="utf-8"))
    proposal = artifact["researcher"]["first_turns"][0]["proposal"]
    assert status == 0
    assert artifact["researcher"]["execution_performed"] is False
    assert proposal["accepted"] is True
    assert proposal["decision"]["actions"][0]["tool"] == "hybrid_search"
    assert proposal["decision"]["actions"][0]["lexical_query"] == (
        "registration security context"
    )
    assert proposal["validations"] == [{
        "action_id": "A-T1",
        "accepted": True,
        "errors": [],
    }]
    assert proposal["receipt"]["prompt_version"] == (
        RESEARCHER_CONTROLLER_PROMPT_VERSION
    )
    assert controller_agent.calls[0]["input_payload"]["evidence_catalog"] == []


def test_complex_planning_only_defers_dependency_tasks() -> None:
    plan = ResearchPlan(
        plan_id="model-plan",
        tasks=(
            ResearchTask(task_id="T1", question="First fact?", objective="O1"),
            ResearchTask(
                task_id="T2",
                question="Dependent relation?",
                objective="O2",
                dependencies=("T1",),
            ),
            ResearchTask(task_id="T3", question="Independent fact?", objective="O3"),
        ),
    )
    planner_agent = _Agent(plan)
    researcher, controller_agent = _researcher(
        _decision("T1"),
        _decision("T3", specification="38.331"),
    )
    application = ResearcherPlanningApplication(
        planner=StructuredResearchPlanner(planner_agent),
        researcher=researcher,
        release=17,
    )

    response = application.run_with_route(
        research_id="Q2",
        question="Compare the facts and then establish their relation.",
        route=ResearchRouteDecision(
            route="decompose",
            complexity="complex",
            rationale="Reviewed complex label.",
        ),
    )

    assert tuple(task_id for task_id, _ in response.first_turns) == ("T1", "T3")
    assert response.deferred_task_ids == ("T2",)
    assert len(controller_agent.calls) == 2
    assert controller_agent.calls[0]["input_payload"]["research_tasks"][0][
        "task_id"
    ] == "T1"
    assert controller_agent.calls[1]["input_payload"]["research_tasks"][0][
        "task_id"
    ] == "T3"


def test_planning_only_rejects_unknown_target_specification() -> None:
    researcher, _ = _researcher(_decision("T1", specification="99.999"))

    proposal = researcher.propose_first_turn((ResearchTask(
        task_id="T1",
        question="One focused rule?",
        objective="Establish the rule.",
    ),))

    assert proposal.accepted is False
    assert proposal.validations[0].accepted is False
    assert proposal.validations[0].errors == (
        "unknown specifications: 99.999",
    )


def test_planning_only_rejects_tool_missing_from_supplied_catalog() -> None:
    action = _action("T1").model_copy(update={
        "tool": "keyword_search",
        "semantic_query": "",
    })
    researcher, _ = _researcher(AgentRetrievalDecision(
        assessment="Use a lexical search.",
        actions=(action,),
    ))

    proposal = researcher.propose_first_turn((ResearchTask(
        task_id="T1",
        question="One focused rule?",
        objective="Establish the rule.",
    ),))

    assert proposal.accepted is False
    assert proposal.validations[0].errors == (
        "tool is not available: keyword_search",
    )

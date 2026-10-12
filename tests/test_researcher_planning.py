from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from specrag.agents.models import (
    AgentEvidenceSelection,
    AgentRetrievalAction,
    AgentRetrievalDecision,
)
from specrag.agents.planning import (
    ResearcherPlanningService,
    propose_researcher_turn,
)
from specrag.agents.state import EvidenceLedger
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
from specrag.retrieval.planner import PlannedRetrievalHit
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


def _action(task_id: str, *, specification: str = ""):
    return AgentRetrievalAction(
        action_id=f"A-{task_id}",
        tool="hybrid_search",
        query_id=task_id,
        lexical_query="registration security context",
        semantic_query=(
            "Normative requirements for the registration security context"
        ),
        hypothetical_document="",
        target_specifications=(specification,) if specification else (),
        seed_evidence_ids=(),
        max_results=5,
    )


def _decision(task_id: str, *, specification: str = ""):
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
    payload = controller_agent.calls[0]["input_payload"]
    assert payload["active_evidence"] == []
    assert payload["pending_candidates"] == []
    assert payload["inactive_candidates"] == []


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
        _decision("T3"),
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


def test_planning_only_rejects_known_but_ungrounded_specification() -> None:
    researcher, _ = _researcher(_decision("T1", specification="24.501"))

    proposal = researcher.propose_first_turn((ResearchTask(
        task_id="T1",
        question="One focused rule?",
        objective="Establish the rule.",
    ),))

    assert proposal.accepted is False
    assert proposal.validations[0].errors == (
        "specifications are not grounded in the task, shared context, or "
        "observed evidence: 24.501",
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


def test_first_turn_may_use_entire_total_action_budget_for_searches() -> None:
    actions = tuple(
        _action("T1").model_copy(update={
            "action_id": f"A{i}",
            "lexical_query": f"registration anchor {i}",
            "semantic_query": f"registration condition {i}",
        })
        for i in range(1, 7)
    )
    researcher, _ = _researcher(AgentRetrievalDecision(
        assessment="Cover four distinct substantive units.",
        actions=actions,
    ))

    proposal = researcher.propose_first_turn((ResearchTask(
        task_id="T1",
        question="Establish four related registration requirements.",
        objective="Establish every requested requirement.",
    ),))

    assert proposal.accepted is True
    assert len(proposal.validations) == 6
    assert proposal.input_payload["budget"] == {
        "rounds_remaining_including_this": 3,
        "tool_actions_remaining": 6,
        "hits_per_search_ceiling": 5,
        "max_active_evidence_per_question": 12,
    }


def test_first_turn_rejects_only_actions_beyond_total_budget() -> None:
    actions = tuple(
        _action("T1").model_copy(update={
            "action_id": f"A{i}",
            "lexical_query": f"registration anchor {i}",
            "semantic_query": f"registration condition {i}",
        })
        for i in range(1, 8)
    )
    researcher, _ = _researcher(AgentRetrievalDecision(
        assessment="Propose more actions than the total budget permits.",
        actions=actions,
    ))

    proposal = researcher.propose_first_turn((ResearchTask(
        task_id="T1",
        question="Establish several registration requirements.",
        objective="Establish every requested requirement.",
    ),))

    assert [item.accepted for item in proposal.validations] == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert proposal.validations[-1].errors == (
        "total tool action budget exhausted",
    )


def test_planning_only_accepts_well_formed_hyde_action() -> None:
    action = _action("T1").model_copy(update={
        "tool": "hyde_search",
        "lexical_query": "",
        "semantic_query": "",
        "hypothetical_document": (
            "The UE shall retain the security context after registration."
        ),
    })
    controller = _Agent(AgentRetrievalDecision(
        assessment="Use hypothetical normative wording for semantic retrieval.",
        actions=(action,),
    ))
    researcher = ResearcherPlanningService(
        controller_agent=StructuredResearcherController(controller),
        corpus_scope=CorpusScope(release=17),
        available_specifications=("24.501",),
        available_tools=(ToolCapability(
            name="hyde_search",
            description="Retrieve using a hypothetical normative passage.",
            required_fields=("hypothetical_document",),
            optional_fields=("target_specifications", "limit"),
            execution_kind="model_scored",
        ),),
    )

    proposal = researcher.propose_first_turn((ResearchTask(
        task_id="T1",
        question="When is the security context retained?",
        objective="Establish the retention rule.",
    ),))

    assert proposal.accepted is True


def _observed_cross_reference() -> PlannedRetrievalHit:
    return PlannedRetrievalHit(
        hit_id="SOURCE",
        chunk_id="38.331:17.17.0:chunk-1",
        collection_rank=1,
        specification="38.331",
        version="17.17.0",
        source_path="data/3gpp/38.331/v17.17.0/source.md",
        document_sha256="A" * 64,
        heading_path=("5.8.9.3Sidelink radio link failure related actions",),
        line_start=100,
        line_end=110,
        excerpt=(
            "The UE shall release the SRBs of this destination according to "
            "clause 5.8.9.1a.3."
        ),
        evidence_roles=("failure handling",),
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


def test_evidence_ledger_keeps_candidates_while_replacing_active_evidence() -> None:
    ledger = EvidenceLedger(("T1",), max_per_question=2)
    hits = tuple(
        _observed_cross_reference().model_copy(update={
            "chunk_id": f"38.331:17.17.0:chunk-{index}",
            "excerpt": f"Candidate {index}",
        })
        for index in range(1, 4)
    )

    _, added = ledger.add("T1", hits, action_id="A-search")
    ledger.select("T1", ("E-T1:1", "E-T1:3"))

    assert added == ("E-T1:1", "E-T1:2", "E-T1:3")
    assert len(ledger.candidate_hits("T1")) == 3
    assert tuple(hit.hit_id for hit in ledger.active_hits("T1")) == (
        "E-T1:1",
        "E-T1:3",
    )
    views = ledger.selection_payload()
    assert views["pending_candidates"] == []
    assert [item["evidence_id"] for item in views["inactive_candidates"]] == [
        "E-T1:2"
    ]


def _clause_action(clause: str) -> AgentRetrievalAction:
    return AgentRetrievalAction(
        action_id="A-clause",
        tool="clause_lookup",
        query_id="T1",
        lexical_query="",
        semantic_query="",
        hypothetical_document="",
        clause=clause,
        target_specifications=("38.331",),
        seed_evidence_ids=(),
        max_results=5,
    )


def test_clause_lookup_accepts_reference_and_specification_from_evidence() -> None:
    ledger = EvidenceLedger(("T1",), max_per_question=12)
    ledger.add("T1", (_observed_cross_reference(),), action_id="A-search")
    controller = _Agent(AgentRetrievalDecision(
        assessment="Follow the answer-bearing normative cross-reference.",
        evidence_selections=(AgentEvidenceSelection(
            query_id="T1",
            retained_evidence_ids=("E-T1:1",),
        ),),
        actions=(_clause_action("5.8.9.1a.3"),),
    ))

    proposal = propose_researcher_turn(
        (ResearchTask(
            task_id="T1",
            question="What must the UE do after sidelink radio link failure?",
            objective="Establish every required UE action.",
        ),),
        controller_agent=StructuredResearcherController(controller),
        corpus_scope=CorpusScope(release=17),
        available_specifications=("38.331",),
        available_tools=(ToolCapability(
            name="clause_lookup",
            description="Retrieve a grounded clause.",
            required_fields=("clause",),
            optional_fields=("target_specifications", "limit"),
            execution_kind="deterministic",
        ),),
        ledger=ledger,
    )

    assert proposal.accepted is True


def test_clause_lookup_rejects_clause_not_present_in_trusted_inputs() -> None:
    ledger = EvidenceLedger(("T1",), max_per_question=12)
    ledger.add("T1", (_observed_cross_reference(),), action_id="A-search")
    controller = _Agent(AgentRetrievalDecision(
        assessment="Do not accept an inferred neighboring clause number.",
        evidence_selections=(AgentEvidenceSelection(
            query_id="T1",
            retained_evidence_ids=("E-T1:1",),
        ),),
        actions=(_clause_action("5.8.9.1a.4"),),
    ))

    proposal = propose_researcher_turn(
        (ResearchTask(
            task_id="T1",
            question="What must the UE do after sidelink radio link failure?",
            objective="Establish every required UE action.",
        ),),
        controller_agent=StructuredResearcherController(controller),
        corpus_scope=CorpusScope(release=17),
        available_specifications=("38.331",),
        available_tools=(ToolCapability(
            name="clause_lookup",
            description="Retrieve a grounded clause.",
            required_fields=("clause",),
            optional_fields=("target_specifications", "limit"),
            execution_kind="deterministic",
        ),),
        ledger=ledger,
    )

    assert proposal.accepted is False
    assert proposal.validations[0].errors == (
        "clause_lookup clause is not grounded in the task, shared context, or "
        "observed evidence",
    )

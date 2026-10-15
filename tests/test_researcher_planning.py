from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from specrag.agents.planning import (
    ResearcherPlanningService,
    propose_researcher_turn,
)
from specrag.agents.state import EvidenceLedger
from specrag.cli import BaselineQuestion, CuratedRouteLabel, run_researcher_planning
from specrag.infrastructure.research_planning import StructuredResearchPlanner
from specrag.infrastructure.researcher_controller import (
    StructuredResearcherController,
)
from specrag.infrastructure.structured_agent import (
    CodexAgentReceipt,
    CodexAgentToolResponse,
)
from specrag.models import (
    NativeFunctionCall,
    ResearchPlan,
    ResearchRouteDecision,
    ResearchTask,
)
from specrag.orchestration.application import ResearcherPlanningApplication
from specrag.prompts import RESEARCHER_CONTROLLER_PROMPT_VERSION
from specrag.retrieval.models import CorpusScope
from specrag.retrieval.planner import PlannedRetrievalHit
from specrag.tools import retrieval_tool_definitions


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


class _PlanningAgent:
    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output=self.outputs.pop(0),
            receipt=_receipt(kwargs["prompt_version"]),
        )


class _NativeAgent:
    def __init__(self, *batches: tuple[NativeFunctionCall, ...]) -> None:
        self.batches = list(batches)
        self.calls: list[dict[str, object]] = []

    def call_tools(self, **kwargs):
        self.calls.append(kwargs)
        calls = self.batches.pop(0)
        return CodexAgentToolResponse(
            calls=calls,
            replay_items=(),
            receipt=_receipt(kwargs["prompt_version"]),
        )


def _control(*, retained: tuple[str, ...] = ()) -> NativeFunctionCall:
    return NativeFunctionCall(
        call_id="control",
        name="update_research_state",
        arguments={
            "assessment": "Search directly for the normative rule.",
            "evidence_selections": (
                [{
                    "query_id": "T1",
                    "retained_evidence_ids": list(retained),
                }]
                if retained else []
            ),
            "unresolved_gaps": [{
                "query_id": "T1",
                "description": "The requested normative rule is unsupported.",
            }],
        },
    )


def _hybrid_call(
    task_id: str,
    *,
    call_id: str = "call-hybrid",
    specification: str = "",
) -> NativeFunctionCall:
    return NativeFunctionCall(
        call_id=call_id,
        name="hybrid_search",
        arguments={
            "query_id": task_id,
            "lexical_query": "registration security context",
            "semantic_query": (
                "Normative requirements for the registration security context"
            ),
            "target_specifications": [specification] if specification else [],
            "max_results": 5,
        },
    )


def _researcher(*batches: tuple[NativeFunctionCall, ...]):
    agent = _NativeAgent(*batches)
    service = ResearcherPlanningService(
        controller_agent=StructuredResearcherController(agent),
        corpus_scope=CorpusScope(release=17),
        available_specifications=("24.501", "38.331"),
        available_tools=retrieval_tool_definitions(),
        max_rounds=3,
        max_total_actions=6,
        hits_per_search=5,
        max_evidence_per_question=12,
    )
    return service, agent


def test_planning_artifact_contains_native_calls_without_execution(
    tmp_path: Path,
) -> None:
    researcher, agent = _researcher((_control(), _hybrid_call("T1")))
    application = ResearcherPlanningApplication(
        planner=StructuredResearchPlanner(_PlanningAgent()),
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
    assert proposal["control_call"]["name"] == "update_research_state"
    assert proposal["retrieval_calls"][0]["tool"] == "hybrid_search"
    assert proposal["validations"] == [{
        "call_id": "call-hybrid",
        "accepted": True,
        "errors": [],
    }]
    assert proposal["receipt"]["prompt_version"] == (
        RESEARCHER_CONTROLLER_PROMPT_VERSION
    )
    payload = agent.calls[0]["input_payload"]
    assert "available_tools" not in payload
    assert payload["active_evidence"] == []


def test_complex_planning_defers_dependency_tasks() -> None:
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
    planner_agent = _PlanningAgent(plan)
    t3_control = _control().model_copy(update={
        "arguments": {
            **_control().arguments,
            "unresolved_gaps": [{
                "query_id": "T3",
                "description": "The independent fact is unsupported.",
            }],
        }
    })
    researcher, agent = _researcher(
        (_control(), _hybrid_call("T1")),
        (t3_control, _hybrid_call("T3", call_id="call-t3")),
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
    assert len(agent.calls) == 2


def test_unknown_and_ungrounded_specifications_are_rejected() -> None:
    task = (ResearchTask(
        task_id="T1",
        question="One focused rule?",
        objective="Establish the rule.",
    ),)
    unknown, _ = _researcher((
        _control(),
        _hybrid_call("T1", specification="99.999"),
    ))
    ungrounded, _ = _researcher((
        _control(),
        _hybrid_call("T1", specification="24.501"),
    ))

    unknown_plan = unknown.propose_first_turn(task)
    ungrounded_plan = ungrounded.propose_first_turn(task)

    assert unknown_plan.validations[0].accepted is False
    assert "unknown specifications: 99.999" in unknown_plan.validations[0].errors
    assert ungrounded_plan.validations[0].errors == (
        "specifications are not grounded in the task, shared context, or "
        "observed evidence: 24.501",
    )


def test_tool_missing_from_catalog_is_rejected() -> None:
    service, _ = _researcher((
        _control(),
        NativeFunctionCall(
            call_id="call-missing",
            name="not_registered",
            arguments={"query_id": "T1"},
        ),
    ))

    proposal = service.propose_first_turn((ResearchTask(
        task_id="T1",
        question="One focused rule?",
        objective="Establish the rule.",
    ),))

    assert proposal.validations[0].errors == (
        "tool is not available: not_registered",
    )


def test_first_turn_budget_accepts_six_and_rejects_seventh_call() -> None:
    task = (ResearchTask(
        task_id="T1",
        question="Establish several registration requirements.",
        objective="Establish every requested requirement.",
    ),)
    calls = tuple(
        _hybrid_call("T1", call_id=f"call-{index}").model_copy(update={
            "arguments": {
                **_hybrid_call("T1").arguments,
                "lexical_query": f"registration anchor {index}",
                "semantic_query": f"registration condition {index}",
            }
        })
        for index in range(1, 8)
    )
    six, _ = _researcher((_control(), *calls[:6]))
    seven, _ = _researcher((_control(), *calls))

    accepted = six.propose_first_turn(task)
    over_budget = seven.propose_first_turn(task)

    assert accepted.accepted is True
    assert [item.accepted for item in over_budget.validations] == [
        True, True, True, True, True, True, False,
    ]
    assert over_budget.validations[-1].errors == (
        "total tool action budget exhausted",
    )


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


def test_evidence_ledger_keeps_complete_candidates_and_replaces_active() -> None:
    ledger = EvidenceLedger(("T1",), max_per_question=2)
    hits = tuple(
        _observed_cross_reference().model_copy(update={
            "chunk_id": f"38.331:17.17.0:chunk-{index}",
            "excerpt": f"start-{index}-" + ("x" * 2400) + f"-end-{index}",
        })
        for index in range(1, 4)
    )
    _, added = ledger.add("T1", hits, action_id="call-search")
    ledger.select("T1", ("E-T1:1", "E-T1:3"))

    assert added == ("E-T1:1", "E-T1:2", "E-T1:3")
    assert len(ledger.candidate_hits("T1")) == 3
    views = ledger.selection_payload()
    assert [item["excerpt"] for item in views["active_evidence"]] == [
        hits[0].excerpt,
        hits[2].excerpt,
    ]
    assert views["inactive_candidates"][0]["excerpt"] == hits[1].excerpt


def test_corrective_ledger_restores_candidates_and_judge_active_set() -> None:
    ledger = EvidenceLedger(("T1",), max_per_question=3)
    seeded = tuple(
        _observed_cross_reference().model_copy(update={
            "hit_id": f"E-T1:{index}",
            "chunk_id": f"38.331:17.17.0:seed-{index}",
            "excerpt": f"Restored source-valid candidate {index}.",
        })
        for index in range(1, 3)
    )

    ledger.seed("T1", seeded, active_evidence_ids=("E-T1:2",))
    _, added = ledger.add(
        "T1",
        (_observed_cross_reference().model_copy(update={
            "chunk_id": "38.331:17.17.0:new",
            "excerpt": "New corrective evidence.",
        }),),
        action_id="call-corrective",
    )

    views = ledger.selection_payload()
    assert [item["evidence_id"] for item in views["active_evidence"]] == [
        "E-T1:2",
    ]
    assert [item["evidence_id"] for item in views["inactive_candidates"]] == [
        "E-T1:1",
    ]
    assert added == ("E-T1:3",)


def test_grounded_clause_call_is_accepted_from_observed_evidence() -> None:
    ledger = EvidenceLedger(("T1",), max_per_question=12)
    ledger.add("T1", (_observed_cross_reference(),), action_id="call-search")
    control = _control(retained=("E-T1:1",))
    clause = NativeFunctionCall(
        call_id="call-clause",
        name="clause_lookup",
        arguments={
            "query_id": "T1",
            "clause": "5.8.9.1a.3",
            "target_specifications": ["38.331"],
            "max_results": 5,
        },
    )
    agent = _NativeAgent((control, clause))

    proposal = propose_researcher_turn(
        (ResearchTask(
            task_id="T1",
            question="What must the UE do after sidelink radio link failure?",
            objective="Establish every required UE action.",
        ),),
        controller_agent=StructuredResearcherController(agent),
        corpus_scope=CorpusScope(release=17),
        available_specifications=("38.331",),
        available_tools=retrieval_tool_definitions(),
        ledger=ledger,
    )

    assert proposal.accepted is True

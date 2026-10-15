from __future__ import annotations

import json
from types import SimpleNamespace

from specrag.agents.models import AgentRetrievalCall
from specrag.agents.researcher import (
    RegistryResearcherService,
    build_retrieval_tool_request,
    run_agentic_retrieval_loop,
)
from specrag.agents.state import EvidenceLedger
from specrag.config import AppConfig, RetrievalCfg
from specrag.infrastructure.structured_agent import (
    CodexAgentReceipt,
    CodexAgentToolResponse,
)
from specrag.infrastructure.researcher_controller import (
    StructuredResearcherController,
)
from specrag.models import NativeFunctionCall, ResearchTask
from specrag.prompts import (
    RESEARCHER_CONTROLLER_INSTRUCTIONS,
    RESEARCHER_CONTROLLER_PROMPT_VERSION,
    RESEARCHER_GAP_INSTRUCTIONS,
    RESEARCHER_GAP_PROMPT_VERSION,
)
from specrag.retrieval.models import CorpusScope, SourceDocument
from specrag.tools import (
    HybridSearchArguments,
    RetrievalToolRequest,
    RetrievalToolResult,
    ToolCapability,
    ToolEvidence,
    ToolRegistry,
)


def _receipt(prompt_version: str = "test") -> CodexAgentReceipt:
    return CodexAgentReceipt(
        agent_runtime="llm_api",
        codex_version="api",
        model="test",
        reasoning_effort="test",
        purpose="test",
        prompt_version=prompt_version,
        request_sha256="a" * 64,
        response_sha256="b" * 64,
        latency_ms=0,
        sandbox="api",
    )


def _review_call(
    *,
    submit: bool = False,
    retained: tuple[str, ...] = (),
    assessment: str = "Review the evidence and address the remaining gap.",
) -> NativeFunctionCall:
    return NativeFunctionCall(
        call_id="control-submit" if submit else "control-update",
        name=("submit_for_validation" if submit else "update_research_state"),
        arguments={
            "assessment": assessment,
            "evidence_selections": (
                [{
                    "query_id": "Q1",
                    "retained_evidence_ids": list(retained),
                }]
                if retained else []
            ),
            "unresolved_gaps": (
                []
                if submit else [{
                    "query_id": "Q1",
                    "description": "The normative procedure remains unsupported.",
                }]
            ),
        },
    )


def _search_call(
    *,
    query_id: str = "Q1",
    tool: str = "hybrid_search",
    call_id: str = "call-search",
) -> NativeFunctionCall:
    arguments: dict[str, object] = {
        "query_id": query_id,
        "target_specifications": [],
        "max_results": 1,
    }
    if tool == "hybrid_search":
        arguments.update({
            "lexical_query": "registration",
            "semantic_query": "registration procedure",
        })
    elif tool == "clause_lookup":
        arguments["clause"] = "5.1"
    return NativeFunctionCall(
        call_id=call_id,
        name=tool,
        arguments=arguments,
    )


def _tasks() -> tuple[ResearchTask, ...]:
    return (ResearchTask(
        task_id="Q1",
        question="What is the registration procedure?",
        objective="Establish the normative registration procedure.",
    ),)


def _retriever() -> SimpleNamespace:
    return SimpleNamespace(
        documents=(SourceDocument(
            source_collection="3gpp_markdown",
            specification="24.501",
            version="17.1.0",
            source_path="data/3gpp/24.501/v17.1.0/source.md",
            document_sha256="A" * 64,
        ),),
        fingerprint="FINGERPRINT",
        index_manifest_path=".local/index/manifest.json",
    )


class _SequenceAgent:
    def __init__(self, *batches: tuple[NativeFunctionCall, ...]) -> None:
        self._batches = iter(batches)
        self.calls: list[dict[str, object]] = []

    def call_tools(self, **kwargs):
        self.calls.append(kwargs)
        batch = next(self._batches)
        replay_items = tuple({
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": json.dumps(call.arguments),
        } for call in batch)
        return CodexAgentToolResponse(
            calls=batch,
            replay_items=replay_items,
            receipt=_receipt(kwargs["prompt_version"]),
        )


class _FailingAgent:
    def call_tools(self, **kwargs):
        raise RuntimeError("synthetic controller failure")


class _NoHitsHybridTool:
    name = "hybrid_search"
    argument_type = HybridSearchArguments
    capability = ToolCapability(
        name="hybrid_search",
        description="Search with lexical and semantic text.",
        execution_kind="deterministic",
    )

    def __init__(self) -> None:
        self.requests: list[RetrievalToolRequest] = []

    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        self.requests.append(request)
        return RetrievalToolResult(
            request_id=request.request_id,
            query_id=request.query_id,
            tool_name=self.name,
            status="no_hits",
            corpus_scope=CorpusScope(release=17),
        )


class _EvidenceHybridTool(_NoHitsHybridTool):
    def invoke(self, request: RetrievalToolRequest) -> RetrievalToolResult:
        self.requests.append(request)
        evidence = tuple(
            ToolEvidence(
                evidence_id=f"RAW-{index}",
                chunk_id=f"24.501:17.1.0:chunk-{index}",
                rank=index,
                match_kind="hybrid_search",
                score=1.0 / index,
                score_kind="rrf",
                specification="24.501",
                release=17,
                version="17.1.0",
                source_path="data/3gpp/24.501/v17.1.0/source.md",
                document_sha256="A" * 64,
                heading_path=(f"5.{index} Registration",),
                line_start=index,
                line_end=index,
                excerpt=f"Candidate evidence {index}.",
            )
            for index in (1, 2)
        )
        return RetrievalToolResult(
            request_id=request.request_id,
            query_id=request.query_id,
            tool_name=self.name,
            status="completed",
            corpus_scope=CorpusScope(release=17),
            evidence=evidence,
        )


def _config() -> AppConfig:
    return AppConfig(retrieval=RetrievalCfg(
        embedding_model="test",
        specification_root="data/3gpp",
        index_dir=".local/test",
    ))


def test_researcher_prompt_uses_evidence_gap_paths_in_order() -> None:
    assert RESEARCHER_CONTROLLER_PROMPT_VERSION == "researcher-controller/v2.3"
    assert "Judge-gap corrective rounds" not in RESEARCHER_CONTROLLER_INSTRUCTIONS
    assert "Update Active Evidence" in RESEARCHER_CONTROLLER_INSTRUCTIONS
    assert "Corrective input and evidence semantics" in RESEARCHER_GAP_INSTRUCTIONS
    assert "Mandatory corrective sequence" in RESEARCHER_GAP_INSTRUCTIONS
    assert "### Path A: existing evidence" in RESEARCHER_GAP_INSTRUCTIONS
    assert "### Path B: source navigation" in RESEARCHER_GAP_INSTRUCTIONS
    assert "### Path C: new search" in RESEARCHER_GAP_INSTRUCTIONS
    assert RESEARCHER_GAP_PROMPT_VERSION == "researcher-controller-gap/v1.0"


def test_evidence_gap_uses_dedicated_prompt_contract() -> None:
    agent = _SequenceAgent((_review_call(submit=True),))
    service = RegistryResearcherService(
        controller_agent=StructuredResearcherController(agent),
        retriever=_retriever(),
        app_config=_config(),
    )

    service.run(
        _tasks(),
        shared_context={
            "gap_retrieval_mode": "evidence_gap",
            "unresolved_gaps": ({
                "query_id": "Q1",
                "required_proposition": "The procedure is explicitly defined.",
                "missing_dimensions": ["relation"],
                "evidence_need": "direct_fact",
                "trusted_anchors": ["registration"],
            },),
        },
    )

    call = agent.calls[0]
    assert call["prompt_version"] == RESEARCHER_GAP_PROMPT_VERSION
    assert call["instructions"] == RESEARCHER_GAP_INSTRUCTIONS


def test_submission_records_stop_reason_without_retrieval() -> None:
    agent = _SequenceAgent((_review_call(submit=True),))

    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(agent),
        retriever=_retriever(),
        app_config=_config(),
    )

    assert result.stop_reason == "submitted_for_validation"
    assert result.rounds_used == 1
    assert result.search_actions_used == 0
    assert result.status == "unavailable"


def test_hyde_call_maps_hypothetical_document_into_request() -> None:
    call = AgentRetrievalCall(
        call_id="call-hyde",
        tool="hyde_search",
        arguments={
            "query_id": "Q1",
            "lexical_query": "required action",
            "hypothetical_document": "The UE shall perform the required action.",
            "target_specifications": [],
            "max_results": 5,
        },
    )

    request = build_retrieval_tool_request(
        call,
        round_number=1,
        ledger=EvidenceLedger(("Q1",), max_per_question=4),
        result_limit=5,
    )

    assert request.query == "required action"
    assert request.semantic_query is None
    assert request.hypothetical_document.startswith("The UE shall")


def test_controller_receives_tools_separately_from_state_payload() -> None:
    agent = _SequenceAgent((_review_call(submit=True),))
    service = RegistryResearcherService(
        controller_agent=StructuredResearcherController(agent),
        retriever=_retriever(),
        app_config=_config(),
    )

    execution, mapping = service.run(_tasks())

    assert execution.stop_reason == "submitted_for_validation"
    assert mapping == {"Q1": "Q1"}
    call = agent.calls[0]
    payload = call["input_payload"]
    assert "available_tools" not in payload
    assert payload["history"] == []
    assert payload["pending_candidates"] == []
    assert {tool.name for tool in call["tools"]} >= {
        "update_research_state",
        "submit_for_validation",
        "hybrid_search",
    }
    assert call["prompt_version"] == RESEARCHER_CONTROLLER_PROMPT_VERSION
    assert call["instructions"] == RESEARCHER_CONTROLLER_INSTRUCTIONS


def test_native_tool_result_is_replayed_on_next_round() -> None:
    agent = _SequenceAgent(
        (_review_call(), _search_call()),
        (_review_call(submit=True),),
    )
    registry = ToolRegistry()
    tool = _NoHitsHybridTool()
    registry.register(tool)

    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(agent),
        retriever=_retriever(),
        app_config=_config(),
        tool_registry=registry,
    )

    assert len(tool.requests) == 1
    assert result.search_actions_used == 1
    assert result.stop_reason == "submitted_for_validation"
    outputs = agent.calls[1]["prior_outputs"]
    assert {output.call_id for output in outputs} == {
        "control-update",
        "call-search",
    }
    search_output = next(
        output for output in outputs if output.call_id == "call-search"
    )
    assert search_output.output["status"] == "no_hits"


def test_budget_exhaustion_gets_one_final_selection_turn() -> None:
    agent = _SequenceAgent(
        (_review_call(), _search_call()),
        (_review_call(submit=True, retained=("E-Q1:2",)),),
    )
    registry = ToolRegistry()
    registry.register(_EvidenceHybridTool())

    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(agent),
        retriever=_retriever(),
        app_config=_config(),
        tool_registry=registry,
        max_rounds=1,
        max_total_actions=1,
        max_evidence_per_question=1,
    )

    assert result.stop_reason == "submitted_for_validation"
    assert tuple(turn.phase for turn in result.turns) == (
        "retrieval",
        "finalization",
    )
    assert agent.calls[1]["input_payload"]["execution_mode"] == (
        "finalization_only"
    )
    assert result.evidence_execution.tasks[0].hits[0].hit_id == "E-Q1:2"


def test_dependency_context_reaches_native_controller() -> None:
    agent = _SequenceAgent((_review_call(submit=True),))
    dependency_context = ({
        "task_id": "T1",
        "supporting_evidence": ({
            "evidence_id": "E1",
            "excerpt": "The intermediate node is identified.",
        },),
    },)

    run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(agent),
        retriever=_retriever(),
        app_config=_config(),
        shared_context={"dependency_context": dependency_context},
    )

    payload = agent.calls[0]["input_payload"]
    assert payload["shared_context"]["dependency_context"] == dependency_context
    assert "completed predecessor tasks" in agent.calls[0]["instructions"]


def test_invalid_calls_are_rejected_without_spending_budget() -> None:
    invalid_batch = (_review_call(), _search_call(query_id="UNKNOWN"))
    agent = _SequenceAgent(invalid_batch, invalid_batch)

    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(agent),
        retriever=_retriever(),
        app_config=_config(),
        max_rounds=2,
        max_finalization_turns=0,
    )

    assert result.stop_reason == "round_budget"
    assert result.search_actions_used == 0
    assert all(
        outcome.status == "rejected"
        for turn in result.turns
        for outcome in turn.outcomes
    )
    assert result.errors == ("unknown query_id: UNKNOWN",) * 2


def test_oversized_native_arguments_are_rejected() -> None:
    oversized = _search_call(tool="clause_lookup")
    oversized = oversized.model_copy(update={
        "arguments": {
            **oversized.arguments,
            "clause": "x" * 65,
        }
    })
    agent = _SequenceAgent(
        (_review_call(), oversized),
        (_review_call(submit=True),),
    )

    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(agent),
        retriever=_retriever(),
        app_config=_config(),
        max_rounds=2,
    )

    first = result.turns[0].outcomes[0]
    assert first.status == "rejected"
    assert "at most 64 characters" in first.errors[0]
    assert result.search_actions_used == 0
    assert result.stop_reason == "submitted_for_validation"


def test_missing_control_rejects_entire_native_batch() -> None:
    agent = _SequenceAgent(
        (_search_call(),),
        (_review_call(submit=True),),
    )

    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(agent),
        retriever=_retriever(),
        app_config=_config(),
        max_rounds=2,
    )

    assert result.turns[0].outcomes[0].status == "rejected"
    assert result.search_actions_used == 0
    assert "exactly one Researcher control call" in result.errors[0]


def test_controller_failure_records_agent_error() -> None:
    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(_FailingAgent()),
        retriever=_retriever(),
        app_config=_config(),
    )

    assert result.stop_reason == "agent_error"
    assert result.rounds_used == 0
    assert "synthetic controller failure" in result.errors[0]

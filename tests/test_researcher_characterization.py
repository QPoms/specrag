from __future__ import annotations

from types import SimpleNamespace

from specrag.agents.researcher import (
    AgentRetrievalAction,
    AgentRetrievalDecision,
    RegistryResearcherService,
    build_retrieval_tool_request,
    run_agentic_retrieval_loop,
)
from specrag.agents.models import AgentEvidenceSelection
from specrag.agents.state import EvidenceLedger
from specrag.config import AppConfig, RetrievalCfg
from specrag.infrastructure.structured_agent import (
    CodexAgentReceipt,
    CodexAgentResponse,
)
from specrag.infrastructure.researcher_controller import (
    StructuredResearcherController,
)
from specrag.models import ResearchTask
from specrag.prompts import (
    RESEARCHER_CONTROLLER_INSTRUCTIONS,
    RESEARCHER_CONTROLLER_PROMPT_VERSION,
)
from specrag.retrieval.models import CorpusScope, SourceDocument
from specrag.tools import (
    RetrievalToolRequest,
    RetrievalToolResult,
    ToolEvidence,
    ToolCapability,
    ToolRegistry,
)


def _receipt() -> CodexAgentReceipt:
    return CodexAgentReceipt(
        codex_version="test",
        model="test",
        reasoning_effort="test",
        purpose="test",
        prompt_version="test",
        request_sha256="a" * 64,
        response_sha256="b" * 64,
        latency_ms=0,
    )


def _action(tool: str, *, query_id: str = "") -> AgentRetrievalAction:
    is_submission = tool == "submit_for_validation"
    return AgentRetrievalAction(
        action_id=f"ACTION-{tool}",
        tool=tool,
        query_id=query_id,
        lexical_query="" if is_submission else "registration",
        semantic_query="" if is_submission else "registration procedure",
        hypothetical_document="",
        target_specifications=(),
        seed_evidence_ids=(),
        max_results=1,
    )


def _tasks() -> tuple[ResearchTask, ...]:
    return (ResearchTask(
        task_id="Q1",
        question="What is the registration procedure?",
        objective="Establish the normative registration procedure.",
    ),)


def _retriever() -> SimpleNamespace:
    return SimpleNamespace(
        documents=(
            SourceDocument(
                source_collection="3gpp_markdown",
                specification="24.501",
                version="17.1.0",
                source_path="data/3gpp/24.501/v17.1.0/source.md",
                document_sha256="A" * 64,
            ),
        ),
        fingerprint="FINGERPRINT",
        index_manifest_path=".local/index/manifest.json",
    )


class _SequenceController:
    def __init__(self, *decisions: AgentRetrievalDecision) -> None:
        self._decisions = iter(decisions)
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return CodexAgentResponse(output=next(self._decisions), receipt=_receipt())


class _FailingController:
    def parse(self, **kwargs):
        raise RuntimeError("synthetic controller failure")


class _NoHitsHybridTool:
    name = "hybrid_search"
    capability = ToolCapability(
        name="hybrid_search",
        description="Search the active corpus with lexical and semantic text.",
        required_fields=("query", "semantic_query"),
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


def test_submission_records_explicit_stop_reason_without_using_tools() -> None:
    controller = _SequenceController(AgentRetrievalDecision(
        assessment="Evidence is sufficient.",
        actions=(_action("submit_for_validation"),),
    ))

    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(controller),
        retriever=_retriever(),
        app_config=_config(),
    )

    assert result.stop_reason == "submitted_for_validation"
    assert result.rounds_used == 1
    assert result.search_actions_used == 0
    assert result.expansion_actions_used == 0
    assert result.status == "unavailable"


def test_hyde_action_maps_hypothetical_document_into_tool_request() -> None:
    action = _action("semantic_search", query_id="Q1").model_copy(update={
        "tool": "hyde_search",
        "lexical_query": "",
        "semantic_query": "",
        "hypothetical_document": "The UE shall perform the required action.",
    })

    request = build_retrieval_tool_request(
        action,
        round_number=1,
        ledger=EvidenceLedger(("Q1",), max_per_question=4),
        result_limit=5,
    )

    assert request.query == ""
    assert request.semantic_query is None
    assert request.hypothetical_document == (
        "The UE shall perform the required action."
    )


def test_first_controller_round_is_the_only_model_planning_entry() -> None:
    controller = _SequenceController(AgentRetrievalDecision(
        assessment="Plan the first executable retrieval actions.",
        actions=(_action("submit_for_validation"),),
    ))
    service = RegistryResearcherService(
        controller_agent=StructuredResearcherController(controller),
        retriever=_retriever(),
        app_config=_config(),
    )

    execution, mapping = service.run((ResearchTask(
        task_id="T1",
        question="What is the registration procedure?",
        objective="Establish the normative procedure.",
    ),))

    assert execution.stop_reason == "submitted_for_validation"
    assert mapping == {"T1": "T1"}
    assert len(controller.calls) == 1
    first_payload = controller.calls[0]["input_payload"]
    assert first_payload["history"] == []
    assert first_payload["active_evidence"] == []
    assert first_payload["pending_candidates"] == []
    assert first_payload["inactive_candidates"] == []
    assert first_payload["available_tools"]
    assert first_payload["research_tasks"] == [{
        "task_id": "T1",
        "question": "What is the registration procedure?",
        "objective": "Establish the normative procedure.",
        "dependencies": [],
        "status": "pending",
    }]
    assert "deterministic_retrieval_hints" not in first_payload
    assert first_payload["corpus_scope"]["release"] == 17
    assert controller.calls[0]["prompt_version"] == (
        RESEARCHER_CONTROLLER_PROMPT_VERSION
    )
    assert controller.calls[0]["instructions"] == (
        RESEARCHER_CONTROLLER_INSTRUCTIONS
    )


def test_first_controller_round_can_emit_and_execute_a_tool_call() -> None:
    controller = _SequenceController(
        AgentRetrievalDecision(
            assessment="Start with the strongest combined search.",
            actions=(_action("hybrid_search", query_id="Q1"),),
        ),
        AgentRetrievalDecision(
            assessment="No hits remain within this bounded test.",
            actions=(_action("submit_for_validation"),),
        ),
    )
    registry = ToolRegistry()
    tool = _NoHitsHybridTool()
    registry.register(tool)

    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(controller),
        retriever=_retriever(),
        app_config=_config(),
        tool_registry=registry,
    )

    assert len(tool.requests) == 1
    assert tool.requests[0].query_id == "Q1"
    assert result.turns[0].outcomes[0].status == "no_hits"
    assert result.search_actions_used == 1
    assert result.stop_reason == "submitted_for_validation"
    assert controller.calls[0]["input_payload"]["history"] == []
    assert controller.calls[1]["input_payload"]["history"][0][
        "tool"
    ] == "hybrid_search"


def test_budget_exhaustion_gets_one_final_evidence_selection_turn() -> None:
    controller = _SequenceController(
        AgentRetrievalDecision(
            assessment="Retrieve candidate passages.",
            actions=(_action("hybrid_search", query_id="Q1"),),
        ),
        AgentRetrievalDecision(
            assessment="Retain the direct passage and submit it.",
            evidence_selections=(AgentEvidenceSelection(
                query_id="Q1",
                retained_evidence_ids=("E-Q1:2",),
            ),),
            actions=(_action("submit_for_validation"),),
        ),
    )
    registry = ToolRegistry()
    tool = _EvidenceHybridTool()
    registry.register(tool)

    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(controller),
        retriever=_retriever(),
        app_config=_config(),
        tool_registry=registry,
        max_rounds=1,
        max_total_actions=1,
        max_evidence_per_question=1,
    )

    assert result.stop_reason == "submitted_for_validation"
    assert result.rounds_used == 1
    assert tuple(turn.phase for turn in result.turns) == (
        "retrieval",
        "finalization",
    )
    assert controller.calls[1]["input_payload"]["execution_mode"] == (
        "finalization_only"
    )
    assert len(controller.calls[1]["input_payload"]["pending_candidates"]) == 2
    assert result.evidence_execution.tasks[0].hits[0].hit_id == "E-Q1:2"
    assert len(result.evidence_execution.tasks[0].hits) == 1
    assert len(result.candidate_evidence_execution.tasks[0].hits) == 2


def test_controller_receives_and_is_instructed_to_use_dependency_context() -> None:
    controller = _SequenceController(AgentRetrievalDecision(
        assessment="Use the predecessor finding to scope the next search.",
        actions=(_action("submit_for_validation"),),
    ))
    dependency_context = ({
        "task_id": "T1",
        "supporting_evidence": ({
            "evidence_id": "E1",
            "excerpt": "The intermediate node is identified.",
        },),
    },)

    run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(controller),
        retriever=_retriever(),
        app_config=_config(),
        shared_context={"dependency_context": dependency_context},
    )

    call = controller.calls[0]
    assert call["prompt_version"] == RESEARCHER_CONTROLLER_PROMPT_VERSION
    assert call["input_payload"]["shared_context"][
        "dependency_context"
    ] == dependency_context
    assert "completed predecessor tasks" in call["instructions"]
    assert "seed_evidence_ids" in call["instructions"]
    normalized_instructions = " ".join(call["instructions"].split())
    assert "first round is the initial retrieval-planning round" in (
        normalized_instructions
    )


def test_invalid_actions_stop_at_round_budget_and_remain_auditable() -> None:
    invalid = AgentRetrievalDecision(
        assessment="Try an unknown question.",
        actions=(_action("hybrid_search", query_id="UNKNOWN"),),
    )
    controller = _SequenceController(invalid, invalid)

    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(controller),
        retriever=_retriever(),
        app_config=_config(),
        max_rounds=2,
    )

    assert result.stop_reason == "round_budget"
    assert result.rounds_used == 2
    assert result.search_actions_used == 0
    assert all(
        outcome.status == "rejected"
        for turn in result.turns
        for outcome in turn.outcomes
    )
    assert result.errors == ("unknown query_id: UNKNOWN",) * 2


def test_controller_failure_records_agent_error_stop_reason() -> None:
    result = run_agentic_retrieval_loop(
        _tasks(),
        controller_agent=StructuredResearcherController(_FailingController()),
        retriever=_retriever(),
        app_config=_config(),
    )

    assert result.stop_reason == "agent_error"
    assert result.rounds_used == 0
    assert result.status == "unavailable"
    assert "synthetic controller failure" in result.errors[0]

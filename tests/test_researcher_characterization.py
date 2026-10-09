from __future__ import annotations

from types import SimpleNamespace

from specrag.agents.researcher import (
    AgentRetrievalAction,
    AgentRetrievalDecision,
    run_agentic_retrieval_loop,
)
from specrag.config import AppConfig, RetrievalCfg
from specrag.infrastructure.structured_agent import (
    CodexAgentReceipt,
    CodexAgentResponse,
)
from specrag.retrieval.models import SourceDocument
from specrag.retrieval.planner import (
    DiscoveryRetrievalPlanV3,
    PlannedDiscoveryRetrieval,
    ReferenceExpansionPolicy,
    RetrievalProbe,
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
    is_finish = tool == "finish"
    return AgentRetrievalAction(
        action_id=f"ACTION-{tool}",
        tool=tool,
        query_id=query_id,
        lexical_query="" if is_finish else "registration",
        semantic_query="" if is_finish else "registration procedure",
        target_specifications=(),
        seed_evidence_ids=(),
        max_results=1,
        rationale="characterization",
        expected_evidence="normative text",
    )


def _plan() -> DiscoveryRetrievalPlanV3:
    item = PlannedDiscoveryRetrieval(
        retrieval_id="RET-Q1",
        query_id="Q1",
        claim_id="C1",
        angle="procedure",
        original_question="What is the registration procedure?",
        planner_action="planned",
        reference_expansion=ReferenceExpansionPolicy(enabled=False),
        retrieval_probes=(
            RetrievalProbe(
                probe_id="P1",
                evidence_role="procedure",
                lexical_query="registration",
                semantic_query="registration procedure",
            ),
        ),
    )
    return DiscoveryRetrievalPlanV3(retrievals=(item,))


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

    def parse(self, **kwargs):
        return CodexAgentResponse(output=next(self._decisions), receipt=_receipt())


class _FailingController:
    def parse(self, **kwargs):
        raise RuntimeError("synthetic controller failure")


def _config() -> AppConfig:
    return AppConfig(retrieval=RetrievalCfg(
        embedding_model="test",
        specification_root="data/3gpp",
        index_dir=".local/test",
    ))


def test_agent_finish_records_explicit_stop_reason_without_using_tools() -> None:
    controller = _SequenceController(AgentRetrievalDecision(
        assessment="Evidence is sufficient.",
        actions=(_action("finish"),),
    ))

    result = run_agentic_retrieval_loop(
        _plan(),
        controller_agent=controller,
        retriever=_retriever(),
        app_config=_config(),
    )

    assert result.stop_reason == "agent_finish"
    assert result.rounds_used == 1
    assert result.search_actions_used == 0
    assert result.expansion_actions_used == 0
    assert result.status == "unavailable"


def test_invalid_actions_stop_at_round_budget_and_remain_auditable() -> None:
    invalid = AgentRetrievalDecision(
        assessment="Try an unknown question.",
        actions=(_action("hybrid_search", query_id="UNKNOWN"),),
    )
    controller = _SequenceController(invalid, invalid)

    result = run_agentic_retrieval_loop(
        _plan(),
        controller_agent=controller,
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
        _plan(),
        controller_agent=_FailingController(),
        retriever=_retriever(),
        app_config=_config(),
    )

    assert result.stop_reason == "agent_error"
    assert result.rounds_used == 0
    assert result.status == "unavailable"
    assert "synthetic controller failure" in result.errors[0]

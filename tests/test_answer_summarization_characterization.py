from __future__ import annotations

from specrag.infrastructure.structured_agent import (
    CodexAgentReceipt,
    CodexAgentResponse,
)
from specrag.retrieval.planner import (
    AgentQuestionAnswerProposal,
    DiscoveryEvidenceSelection,
    EvidenceBackedAnswer,
    QuestionEvidenceSelection,
    SelectedEvidenceGroup,
    SelectedSpecificationEvidence,
    summarize_selected_discovery_evidence,
)


def _receipt() -> CodexAgentReceipt:
    return CodexAgentReceipt(
        codex_version="test",
        model="test",
        reasoning_effort="none",
        purpose="test",
        prompt_version="test/v1",
        request_sha256="a" * 64,
        response_sha256="b" * 64,
        latency_ms=1,
    )


class _SummaryAgent:
    def parse(self, **kwargs):
        return CodexAgentResponse(
            output=AgentQuestionAnswerProposal(
                query_id="Q1",
                answers=(
                    EvidenceBackedAnswer(
                        answer_id="agent-valid",
                        hypothesis_id=None,
                        coverage_status="supported",
                        answer_type="protocol_message",
                        statement="The Registration Request carries the object.",
                        support_kind="direct_statement",
                        evidence_ids=("E1",),
                    ),
                    EvidenceBackedAnswer(
                        answer_id="agent-unknown",
                        hypothesis_id=None,
                        coverage_status="supported",
                        answer_type="other",
                        statement="An unsupported answer.",
                        support_kind="direct_statement",
                        evidence_ids=("UNKNOWN",),
                    ),
                ),
            ),
            receipt=_receipt(),
        )


def test_selected_summary_rejects_unknown_evidence_ids() -> None:
    evidence = SelectedSpecificationEvidence(
        evidence_id="E1",
        source_stage="first_stage",
        chunk_id="C1",
        specification="24.501",
        version="17.1.0",
        source_path="TS_24.501.md",
        document_sha256="A" * 64,
        heading_path=("5 Procedure",),
        line_start=10,
        line_end=12,
        excerpt="The Registration Request carries the object.",
    )
    selection = DiscoveryEvidenceSelection(
        status="completed",
        max_evidence_groups_per_question=4,
        questions=(
            QuestionEvidenceSelection(
                query_id="Q1",
                claim_id="C1",
                angle="carrier",
                original_question="Which message carries the object?",
                status="completed",
                evidence_groups=(
                    SelectedEvidenceGroup(
                        group_id="G1",
                        source_stages=("first_stage",),
                        evidence=(evidence,),
                    ),
                ),
            ),
        ),
    )

    summary = summarize_selected_discovery_evidence(
        selection,
        summary_agent=_SummaryAgent(),
    )

    assert summary.status == "completed"
    assert [answer.evidence_ids for answer in summary.questions[0].answers] == [
        ("E1",),
    ]
    assert any(
        "unknown evidence IDs" in warning
        for warning in summary.questions[0].warnings
    )

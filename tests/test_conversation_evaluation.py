from specrag.evaluation import (
    ConversationEvaluationCase,
    evaluate_conversation_case,
    summarize_conversation_evaluations,
)
from specrag.models import ConversationMessageContent


RESULT_ID = "RESULT-" + "A" * 32


def test_conversation_evaluation_covers_direct_and_research_routing() -> None:
    direct = evaluate_conversation_case(ConversationEvaluationCase(
        case_id="ordinary-chat",
        expected_action="direct",
        response=ConversationMessageContent(
            text="How can I help?", source_kind="direct_conversation"
        ),
    ))
    researched = evaluate_conversation_case(ConversationEvaluationCase(
        case_id="normative-research",
        expected_action="start_research",
        invoked_tools=("start_research",),
        response=ConversationMessageContent(
            text="The evidence supports this conclusion. [E1]",
            source_kind="research_result",
            research_result_ids=(RESULT_ID,),
            run_ids=("RUN-" + "B" * 32,),
            citations=("E1",),
        ),
        expected_conditions=("UE is registered",),
        submitted_conditions=("UE is registered",),
        expected_result_ids=(RESULT_ID,),
        expected_citations=("E1",),
        available_evidence_ids=("E1", "E2"),
        required_context_refs=("MSG-1", RESULT_ID),
        recovered_context_refs=("MSG-1", RESULT_ID),
        compacted=True,
    ))

    assert direct.passed
    assert researched.passed
    assert summarize_conversation_evaluations((direct, researched)) == {
        "cases": 2,
        "passed": 2,
        "pass_rate": 1.0,
    }


def test_conversation_evaluation_detects_unsupported_and_drifted_trace() -> None:
    evaluated = evaluate_conversation_case(ConversationEvaluationCase(
        case_id="unsupported-direct",
        expected_action="start_research",
        response=ConversationMessageContent(
            text="The protocol requires it.", source_kind="direct_conversation"
        ),
        expected_conditions=("Release 17",),
        submitted_conditions=(),
        expected_result_ids=(RESULT_ID,),
        expected_citations=("E1",),
        available_evidence_ids=(),
        required_context_refs=(RESULT_ID,),
        recovered_context_refs=(),
        compacted=True,
    ))

    assert not evaluated.passed
    assert not evaluated.routing_correct
    assert not evaluated.normative_source_valid
    assert not evaluated.conditions_preserved
    assert not evaluated.result_selection_correct
    assert not evaluated.citation_fidelity
    assert not evaluated.compaction_recovery

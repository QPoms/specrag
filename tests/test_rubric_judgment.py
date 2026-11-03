from __future__ import annotations

import pytest
from pydantic import ValidationError

from specrag.evaluation import (
    BenchmarkRubric,
    FinalAnswerJudgmentDraft,
    JudgeRubricAssessment,
    RubricItem,
    ScoreBand,
    score_rubric_judgment,
)
from specrag.evaluation.rubric_judgment import FinalAnswerJudgeInput
from specrag.prompts import FINAL_ANSWER_JUDGE_PROMPT_VERSION


def _band(ratio: float, description: str) -> ScoreBand:
    return ScoreBand(ratio=ratio, description=description)


def _rubric() -> BenchmarkRubric:
    return BenchmarkRubric(
        items=(
            RubricItem(
                rubric_id="R1",
                criterion="State the first conclusion and its condition.",
                weight=60,
                full=_band(1.0, "Conclusion and condition are correct."),
                partial=_band(0.5, "Conclusion is correct but condition is absent."),
                none=_band(0.0, "Conclusion is absent or incorrect."),
            ),
            RubricItem(
                rubric_id="R2",
                criterion="State the second relationship.",
                weight=40,
                full=_band(1.0, "Relationship is correct."),
                partial=_band(0.5, "Relationship direction is incomplete."),
                none=_band(0.0, "Relationship is absent or incorrect."),
            ),
        )
    )


def _judge_input() -> FinalAnswerJudgeInput:
    return FinalAnswerJudgeInput(
        question="What happens in both cases?",
        reference_answer="The first conclusion is conditional; B follows A.",
        rubric=_rubric(),
        candidate_answer=(
            "The first conclusion applies\nonly under condition X. "
            "The answer also mentions A."
        ),
    )


def test_judge_input_contains_exactly_four_semantic_fields() -> None:
    payload = _judge_input().model_dump(mode="json")

    assert list(payload) == [
        "question",
        "reference_answer",
        "rubric",
        "candidate_answer",
    ]
    assert "gold_evidence" not in payload
    assert "environment" not in payload


def test_judge_output_schema_requires_every_nested_field() -> None:
    schema = FinalAnswerJudgmentDraft.model_json_schema()
    assessment = schema["$defs"]["JudgeRubricAssessment"]

    assert set(schema["required"]) == {
        "rubric_assessments",
        "overall_reasoning",
    }
    assert set(assessment["required"]) == {
        "rubric_id",
        "level",
        "answer_support",
        "reason",
    }
    assert assessment["additionalProperties"] is False


def test_program_scores_ordered_judgment_and_places_score_before_assessments() -> None:
    judgment = FinalAnswerJudgmentDraft(
        rubric_assessments=(
            JudgeRubricAssessment(
                rubric_id="R1",
                level="full",
                answer_support=(
                    "The first conclusion applies only under condition X.",
                ),
                reason="Both the conclusion and condition are present.",
            ),
            JudgeRubricAssessment(
                rubric_id="R2",
                level="partial",
                answer_support=("mentions A.",),
                reason="A is present but the relationship to B is missing.",
            ),
        ),
        overall_reasoning="The first item is complete and the second is partial.",
    )

    evaluation = score_rubric_judgment(
        question_id="r17-qa-001",
        run_id="run-001",
        judge_input=_judge_input(),
        judgment=judgment,
        judge_model="judge-model",
        judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
    )

    assert evaluation.score == 80.0
    assert [item.awarded_score for item in evaluation.rubric_assessments] == [
        60.0,
        20.0,
    ]
    assert [item.score_ratio for item in evaluation.rubric_assessments] == [
        1.0,
        0.5,
    ]
    assert list(evaluation.model_dump(mode="json")) == [
        "schema_version",
        "question_id",
        "run_id",
        "score",
        "rubric_assessments",
        "overall_reasoning",
        "judge_model",
        "judge_prompt_version",
    ]


@pytest.mark.parametrize(
    "assessments",
    [
        (
            JudgeRubricAssessment(
                rubric_id="R2",
                level="none",
                answer_support=(),
                reason="Absent.",
            ),
            JudgeRubricAssessment(
                rubric_id="R1",
                level="none",
                answer_support=(),
                reason="Absent.",
            ),
        ),
        (
            JudgeRubricAssessment(
                rubric_id="R1",
                level="none",
                answer_support=(),
                reason="Absent.",
            ),
        ),
    ],
)
def test_scoring_rejects_missing_or_out_of_order_rubric_ids(
    assessments: tuple[JudgeRubricAssessment, ...],
) -> None:
    judgment = FinalAnswerJudgmentDraft(
        rubric_assessments=assessments,
        overall_reasoning="Nothing was covered.",
    )

    with pytest.raises(ValueError, match="every rubric ID once"):
        score_rubric_judgment(
            question_id="r17-qa-001",
            run_id="run-001",
            judge_input=_judge_input(),
            judgment=judgment,
            judge_model="judge-model",
            judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
        )


def test_scoring_accepts_semantic_support_not_copied_from_candidate_answer() -> None:
    judgment = FinalAnswerJudgmentDraft(
        rubric_assessments=(
            JudgeRubricAssessment(
                rubric_id="R1",
                level="full",
                answer_support=("The answer gives the conclusion and condition.",),
                reason="The Candidate Answer semantically covers the item.",
            ),
            JudgeRubricAssessment(
                rubric_id="R2",
                level="none",
                answer_support=("The answer merely mentions A.",),
                reason="Absent.",
            ),
        ),
        overall_reasoning="One item was purportedly covered.",
    )

    evaluation = score_rubric_judgment(
        question_id="r17-qa-001",
        run_id="run-001",
        judge_input=_judge_input(),
        judgment=judgment,
        judge_model="judge-model",
        judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
    )

    assert evaluation.score == 60.0
    assert evaluation.rubric_assessments[0].answer_support == (
        "The answer gives the conclusion and condition.",
    )
    assert evaluation.rubric_assessments[1].answer_support == (
        "The answer merely mentions A.",
    )


def test_judge_draft_rejects_scores_but_support_is_optional_audit_text() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FinalAnswerJudgmentDraft.model_validate(
            {
                "rubric_assessments": [
                    {
                        "rubric_id": "R1",
                        "level": "none",
                        "answer_support": [],
                        "reason": "Absent.",
                        "awarded_score": 60,
                    }
                ],
                "overall_reasoning": "Absent.",
            }
        )

    partial = JudgeRubricAssessment(
        rubric_id="R1",
        level="partial",
        answer_support=(),
        reason="Incomplete.",
    )
    contradicted = JudgeRubricAssessment(
        rubric_id="R1",
        level="none",
        answer_support=("The Candidate Answer reverses the relationship.",),
        reason="Contradicted.",
    )

    assert partial.answer_support == ()
    assert contradicted.answer_support

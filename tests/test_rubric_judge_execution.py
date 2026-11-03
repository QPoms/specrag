from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from specrag.evaluation import (
    FinalAnswerJudgeInput,
    FinalAnswerJudgmentDraft,
    JudgeCallResult,
    JudgeRubricAssessment,
    execute_final_answer_judge,
    load_end_to_end_benchmark,
)
from specrag.prompts import FINAL_ANSWER_JUDGE_PROMPT_VERSION


ROOT = Path(__file__).resolve().parents[1]


class _SequenceJudge:
    def __init__(self, outcomes: Iterable[JudgeCallResult | Exception]) -> None:
        self._outcomes = iter(outcomes)
        self.inputs: list[FinalAnswerJudgeInput] = []
        self.corrections: list[str | None] = []

    def __call__(
        self,
        judge_input: FinalAnswerJudgeInput,
        *,
        correction: str | None = None,
    ) -> JudgeCallResult:
        self.inputs.append(judge_input)
        self.corrections.append(correction)
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _ProviderFailure(RuntimeError):
    failure_kind = "provider_unavailable"
    public_message = "The Judge provider is temporarily unavailable."


def _draft(*, support: str, level: str = "full") -> FinalAnswerJudgmentDraft:
    case = load_end_to_end_benchmark(ROOT).cases[0]
    assessments = []
    for index, item in enumerate(case.rubric.items):
        item_level = level if index == 0 else "none"
        assessments.append(
            JudgeRubricAssessment(
                rubric_id=item.rubric_id,
                level=item_level,
                answer_support=(support,) if item_level != "none" else (),
                reason="The first item is assessed; remaining items are absent.",
            )
        )
    return FinalAnswerJudgmentDraft(
        rubric_assessments=tuple(assessments),
        overall_reasoning="Only the first Rubric Item is covered.",
    )


def test_empty_candidate_is_deterministic_zero_without_judge_call() -> None:
    case = load_end_to_end_benchmark(ROOT).cases[0]
    judge = _SequenceJudge(())

    result = execute_final_answer_judge(
        case=case,
        run_id="run-empty",
        candidate_answer="  \n",
        judge=judge,
        judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
    )

    assert result.status == "no_answer"
    assert result.score == 0.0
    assert result.attempt_count == 0
    assert result.attempt_failures == ()
    assert judge.inputs == []
    assert result.evaluation is not None
    assert result.evaluation.judge_model is None
    assert all(
        item.level == "none" and item.awarded_score == 0.0
        for item in result.evaluation.rubric_assessments
    )


def test_invalid_first_output_retries_same_input_then_scores() -> None:
    case = load_end_to_end_benchmark(ROOT).cases[0]
    candidate = "The candidate states the supported conclusion."
    incomplete = _draft(support="A semantic summary.")
    incomplete = FinalAnswerJudgmentDraft(
        rubric_assessments=incomplete.rubric_assessments[:-1],
        overall_reasoning=incomplete.overall_reasoning,
    )
    judge = _SequenceJudge((
        JudgeCallResult(judgment=incomplete, judge_model="judge-model"),
        JudgeCallResult(
            judgment=_draft(support="supported conclusion"),
            judge_model="judge-model",
        ),
    ))

    result = execute_final_answer_judge(
        case=case,
        run_id="run-retry",
        candidate_answer=candidate,
        judge=judge,
        judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
    )

    assert result.status == "scored"
    assert result.score == float(case.rubric.items[0].weight)
    assert result.attempt_count == 2
    assert len(result.attempt_failures) == 1
    assert result.attempt_failures[0].failure_kind == "invalid_output"
    assert len(judge.inputs) == 2
    assert judge.inputs[0] is judge.inputs[1]
    assert judge.corrections[0] is None
    assert "every Rubric Item exactly once" in str(judge.corrections[1])


def test_timeout_retries_once_and_can_recover() -> None:
    case = load_end_to_end_benchmark(ROOT).cases[0]
    candidate = "The candidate states the supported conclusion."
    judge = _SequenceJudge((
        TimeoutError("sensitive timeout detail"),
        JudgeCallResult(
            judgment=_draft(support="supported conclusion"),
            judge_model="judge-model",
        ),
    ))

    result = execute_final_answer_judge(
        case=case,
        run_id="run-timeout",
        candidate_answer=candidate,
        judge=judge,
        judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
    )

    assert result.status == "scored"
    assert result.attempt_count == 2
    assert result.attempt_failures[0].failure_kind == "timeout"
    assert result.attempt_failures[0].message == "Judge call timed out."
    assert judge.corrections == [None, None]


def test_two_provider_failures_produce_no_score() -> None:
    case = load_end_to_end_benchmark(ROOT).cases[0]
    judge = _SequenceJudge((_ProviderFailure(), _ProviderFailure()))

    result = execute_final_answer_judge(
        case=case,
        run_id="run-failed",
        candidate_answer="A non-empty Candidate Answer.",
        judge=judge,
        judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
    )

    assert result.status == "evaluation_failed"
    assert result.score is None
    assert result.evaluation is None
    assert result.attempt_count == 2
    assert [item.failure_kind for item in result.attempt_failures] == [
        "provider_error",
        "provider_error",
    ]
    assert all(
        item.message == "The Judge provider is temporarily unavailable."
        for item in result.attempt_failures
    )


def test_programming_error_is_not_hidden_as_evaluation_failure() -> None:
    case = load_end_to_end_benchmark(ROOT).cases[0]
    judge = _SequenceJudge((RuntimeError("programming defect"),))

    with pytest.raises(RuntimeError, match="programming defect"):
        execute_final_answer_judge(
            case=case,
            run_id="run-defect",
            candidate_answer="A non-empty Candidate Answer.",
            judge=judge,
            judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
        )

    assert len(judge.inputs) == 1

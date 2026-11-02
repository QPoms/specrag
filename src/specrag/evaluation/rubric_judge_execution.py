"""Bounded execution policy for offline final-answer judgment."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .benchmark_models import BenchmarkCase
from .rubric_judgment import (
    FinalAnswerEvaluation,
    FinalAnswerJudgeInput,
    FinalAnswerJudgmentDraft,
    ScoredRubricAssessment,
    build_final_answer_judge_input,
    score_rubric_judgment,
)


MAX_JUDGE_ATTEMPTS = 2

JudgeExecutionStatus = Literal["scored", "no_answer", "evaluation_failed"]
JudgeFailureKind = Literal[
    "invalid_output",
    "timeout",
    "provider_error",
    "unknown",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JudgeCallResult(_StrictModel):
    """Provider-neutral successful response from one Judge call."""

    judgment: FinalAnswerJudgmentDraft
    judge_model: str = Field(min_length=1)

    @model_validator(mode="after")
    def _nonblank_model(self) -> "JudgeCallResult":
        if not self.judge_model.strip():
            raise ValueError("judge_model must not be blank")
        return self


class FinalAnswerJudge(Protocol):
    """Injected one-call boundary; provider adapters implement this protocol."""

    def __call__(self, judge_input: FinalAnswerJudgeInput) -> JudgeCallResult: ...


class JudgeAttemptFailure(_StrictModel):
    attempt: int = Field(ge=1, le=MAX_JUDGE_ATTEMPTS)
    failure_kind: JudgeFailureKind
    error_type: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=1000)


class FinalAnswerJudgeExecution(_StrictModel):
    """Outcome of the bounded Judge policy, including non-scoring failures."""

    schema_version: Literal["specrag.final-answer-judge-execution/v1"] = (
        "specrag.final-answer-judge-execution/v1"
    )
    question_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    status: JudgeExecutionStatus
    score: float | None = Field(default=None, ge=0.0, le=100.0)
    attempt_count: int = Field(ge=0, le=MAX_JUDGE_ATTEMPTS)
    attempt_failures: tuple[JudgeAttemptFailure, ...] = ()
    evaluation: FinalAnswerEvaluation | None = None
    judge_prompt_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_execution(self) -> "FinalAnswerJudgeExecution":
        if not self.question_id.strip() or not self.run_id.strip():
            raise ValueError("question_id and run_id must not be blank")
        if not self.judge_prompt_version.strip():
            raise ValueError("judge_prompt_version must not be blank")
        expected_attempts = tuple(
            range(1, len(self.attempt_failures) + 1)
        )
        actual_attempts = tuple(item.attempt for item in self.attempt_failures)
        if actual_attempts != expected_attempts:
            raise ValueError("attempt failures must be ordered from attempt 1")

        if self.status == "evaluation_failed":
            if self.score is not None or self.evaluation is not None:
                raise ValueError("failed Judge execution cannot contain a score")
            if self.attempt_count != MAX_JUDGE_ATTEMPTS:
                raise ValueError("failed Judge execution must exhaust both attempts")
            if len(self.attempt_failures) != MAX_JUDGE_ATTEMPTS:
                raise ValueError("failed Judge execution requires both failures")
            return self

        if self.evaluation is None or self.score != self.evaluation.score:
            raise ValueError("scored execution must contain its matching evaluation")
        if self.evaluation.question_id != self.question_id:
            raise ValueError("execution and evaluation question IDs must match")
        if self.evaluation.run_id != self.run_id:
            raise ValueError("execution and evaluation Run IDs must match")

        if self.status == "no_answer":
            if self.score != 0.0 or self.attempt_count != 0:
                raise ValueError("no-answer execution must score zero without a call")
            if self.attempt_failures:
                raise ValueError("no-answer execution cannot contain Judge failures")
            if self.evaluation.judge_model is not None:
                raise ValueError("no-answer evaluation cannot identify a Judge model")
            return self

        if self.attempt_count not in {1, 2}:
            raise ValueError("scored Judge execution requires one or two attempts")
        if len(self.attempt_failures) != self.attempt_count - 1:
            raise ValueError("scored execution failure count disagrees with attempts")
        if self.evaluation.judge_model is None:
            raise ValueError("model-scored evaluation requires Judge metadata")
        return self


def execute_final_answer_judge(
    *,
    case: BenchmarkCase,
    run_id: str,
    candidate_answer: str | None,
    judge: FinalAnswerJudge,
    judge_prompt_version: str,
) -> FinalAnswerJudgeExecution:
    """Run at most two Judge calls, never rerunning the Research Workflow."""

    question_id = case.task_metadata.task_id
    if candidate_answer is None or not candidate_answer.strip():
        evaluation = _no_answer_evaluation(case=case, run_id=run_id)
        return FinalAnswerJudgeExecution(
            question_id=question_id,
            run_id=run_id,
            status="no_answer",
            score=0.0,
            attempt_count=0,
            evaluation=evaluation,
            judge_prompt_version=judge_prompt_version,
        )

    judge_input = build_final_answer_judge_input(
        case,
        candidate_answer=candidate_answer,
    )
    failures: list[JudgeAttemptFailure] = []
    for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
        try:
            result = judge(judge_input)
        except Exception as exc:  # noqa: BLE001 - provider boundary is injected
            if not _is_expected_judge_call_failure(exc):
                raise
            failures.append(_attempt_failure(attempt=attempt, exc=exc))
            continue
        try:
            evaluation = score_rubric_judgment(
                question_id=question_id,
                run_id=run_id,
                judge_input=judge_input,
                judgment=result.judgment,
                judge_model=result.judge_model,
                judge_prompt_version=judge_prompt_version,
            )
            return FinalAnswerJudgeExecution(
                question_id=question_id,
                run_id=run_id,
                status="scored",
                score=evaluation.score,
                attempt_count=attempt,
                attempt_failures=tuple(failures),
                evaluation=evaluation,
                judge_prompt_version=judge_prompt_version,
            )
        except ValueError as exc:
            failures.append(_attempt_failure(attempt=attempt, exc=exc))

    return FinalAnswerJudgeExecution(
        question_id=question_id,
        run_id=run_id,
        status="evaluation_failed",
        score=None,
        attempt_count=MAX_JUDGE_ATTEMPTS,
        attempt_failures=tuple(failures),
        evaluation=None,
        judge_prompt_version=judge_prompt_version,
    )


def _no_answer_evaluation(
    *,
    case: BenchmarkCase,
    run_id: str,
) -> FinalAnswerEvaluation:
    assessments = tuple(
        ScoredRubricAssessment(
            rubric_id=item.rubric_id,
            weight=item.weight,
            level="none",
            score_ratio=0.0,
            awarded_score=0.0,
            answer_support=(),
            reason="Workflow produced no Candidate Answer.",
        )
        for item in case.rubric.items
    )
    return FinalAnswerEvaluation(
        question_id=case.task_metadata.task_id,
        run_id=run_id,
        score=0.0,
        rubric_assessments=assessments,
        overall_reasoning=(
            "Workflow produced no Candidate Answer; every Rubric Item received none."
        ),
        judge_model=None,
        judge_prompt_version=None,
    )


def _attempt_failure(*, attempt: int, exc: Exception) -> JudgeAttemptFailure:
    failure_kind = _failure_kind(exc)
    if failure_kind == "invalid_output":
        message = str(exc).strip()[:1000] or "Judge output failed validation."
    else:
        public_message = getattr(exc, "public_message", None)
        message = (
            public_message.strip()[:1000]
            if isinstance(public_message, str) and public_message.strip()
            else {
                "timeout": "Judge call timed out.",
                "provider_error": "Judge provider call failed.",
                "unknown": "Judge call failed.",
            }[failure_kind]
        )
    return JudgeAttemptFailure(
        attempt=attempt,
        failure_kind=failure_kind,
        error_type=type(exc).__name__[:200],
        message=message,
    )


def _failure_kind(exc: Exception) -> JudgeFailureKind:
    provider_kind = getattr(exc, "failure_kind", None)
    if isinstance(exc, ValueError) or provider_kind == "structured_output":
        return "invalid_output"
    if isinstance(exc, TimeoutError) or provider_kind == "timeout":
        return "timeout"
    if provider_kind in {
        "authentication_or_configuration",
        "context_overflow",
        "invalid_request",
        "provider_unavailable",
        "rate_limited",
        "retryable_transient",
    }:
        return "provider_error"
    return "unknown"


def _is_expected_judge_call_failure(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or isinstance(
        getattr(exc, "failure_kind", None),
        str,
    )


__all__ = [
    "FinalAnswerJudge",
    "FinalAnswerJudgeExecution",
    "JudgeAttemptFailure",
    "JudgeCallResult",
    "JudgeExecutionStatus",
    "JudgeFailureKind",
    "MAX_JUDGE_ATTEMPTS",
    "execute_final_answer_judge",
]

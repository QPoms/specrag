from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_final_answer_benchmark import (
    _parser,
    _candidate_answer,
    _run_id,
    _selected_cases,
    _summary,
    _validate_artifact_case,
)
from specrag.evaluation import execute_final_answer_judge, load_end_to_end_benchmark
from specrag.prompts import FINAL_ANSWER_JUDGE_PROMPT_VERSION


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/runs/end-to-end/20261018-formal-50"


class _ForbiddenJudge:
    def __call__(self, judge_input):
        raise AssertionError("Judge must not be called for an empty answer")


def test_runner_selects_simple_cases_in_benchmark_order() -> None:
    cases = _selected_cases(difficulty="simple", identifiers=None, limit=3)

    assert len(cases) == 3
    assert all(case.task_metadata.difficulty == "simple" for case in cases)
    assert [case.task_metadata.task_id for case in cases] == sorted(
        case.task_metadata.task_id for case in cases
    )


def test_runner_exposes_explicit_api_and_codex_backends() -> None:
    args = _parser().parse_args(["runs", "outputs", "--backend", "codex"])

    assert args.backend == "codex"


def test_runner_extracts_saved_answer_and_run_identity() -> None:
    case = load_end_to_end_benchmark(ROOT).cases[0]
    path = RUNS / f"{case.task_metadata.task_id}.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))

    _validate_artifact_case(artifact, case)

    assert _candidate_answer(artifact) == artifact["state"]["final_answer"]
    assert _run_id(artifact, question_id=case.task_metadata.task_id) == artifact[
        "thread_id"
    ]


def test_runner_rejects_question_text_drift() -> None:
    case = load_end_to_end_benchmark(ROOT).cases[0]
    artifact = {
        "question": {"id": case.task_metadata.task_id, "text": "changed"},
        "state": {},
    }

    with pytest.raises(ValueError, match="text does not match"):
        _validate_artifact_case(artifact, case)


def test_summary_includes_zero_but_suppresses_average_after_judge_failure() -> None:
    cases = _selected_cases(difficulty="all", identifiers=None, limit=2)
    no_answer = execute_final_answer_judge(
        case=cases[0],
        run_id="run-empty",
        candidate_answer=None,
        judge=_ForbiddenJudge(),
        judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
    )
    failed_payload = no_answer.model_dump(mode="python")
    failed_payload.update({
        "question_id": cases[1].task_metadata.task_id,
        "run_id": "run-failed",
        "status": "evaluation_failed",
        "score": None,
        "attempt_count": 2,
        "attempt_failures": [
            {
                "attempt": 1,
                "failure_kind": "provider_error",
                "error_type": "ProviderError",
                "message": "Provider failed.",
            },
            {
                "attempt": 2,
                "failure_kind": "provider_error",
                "error_type": "ProviderError",
                "message": "Provider failed.",
            },
        ],
        "evaluation": None,
    })
    from specrag.evaluation import FinalAnswerJudgeExecution

    failed = FinalAnswerJudgeExecution.model_validate(failed_payload)
    summary = _summary(cases=cases, executions=(no_answer, failed))

    assert summary["score_count"] == 1
    assert summary["no_answer_count"] == 1
    assert summary["evaluation_failed_count"] == 1
    assert summary["average_score"] is None
    assert summary["score_by_difficulty"] == {
        "simple": {
            "requested_count": 2,
            "score_count": 1,
            "evaluation_failed_count": 1,
            "average_score": None,
        }
    }

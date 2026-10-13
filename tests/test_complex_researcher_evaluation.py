from __future__ import annotations

import json
from argparse import Namespace

from scripts.evaluate_complex_researcher import (
    _atomic_write_json,
    _load_resume_result,
    _run_batch,
)


def _artifact(question_id: str = "r17-qa-008") -> dict[str, object]:
    return {
        "schema_version": "specrag.complex-researcher-evaluation/v1",
        "question": {"id": question_id},
        "state": {
            "stop_reason": "completed",
            "evidence_sufficient": True,
            "budget": {
                "tool_calls_used": 10,
                "tool_calls_used_by_task": [["T1", 2], ["T2", 2], ["T3", 6]],
            },
        },
    }


def test_resume_accepts_valid_atomic_artifact(tmp_path) -> None:
    output = tmp_path / "r17-qa-008.json"
    _atomic_write_json(output, _artifact())

    result = _load_resume_result(
        output,
        expected_question_id="r17-qa-008",
    )

    assert result == {
        "question_id": "r17-qa-008",
        "execution": "skipped",
        "stop_reason": "completed",
        "evidence_sufficient": True,
        "tool_calls_used": 10,
        "tool_calls_used_by_task": [["T1", 2], ["T2", 2], ["T3", 6]],
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_retries_truncated_or_mismatched_artifacts(tmp_path) -> None:
    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"schema_version":', encoding="utf-8")
    mismatched = tmp_path / "mismatched.json"
    mismatched.write_text(json.dumps(_artifact("r17-qa-009")), encoding="utf-8")

    assert _load_resume_result(
        truncated,
        expected_question_id="r17-qa-008",
    ) is None
    assert _load_resume_result(
        mismatched,
        expected_question_id="r17-qa-008",
    ) is None


def test_batch_resume_skips_existing_artifacts_and_writes_summary(tmp_path) -> None:
    planning_directory = tmp_path / "planning"
    output_directory = tmp_path / "output"
    planning_directory.mkdir()
    output_directory.mkdir()
    for question_id in ("r17-qa-008", "r17-qa-009"):
        _atomic_write_json(
            planning_directory / f"{question_id}.json",
            {"question": {"id": question_id}},
        )
        _atomic_write_json(
            output_directory / f"{question_id}.json",
            _artifact(question_id),
        )

    exit_code = _run_batch(Namespace(
        planning_artifact=planning_directory,
        output=output_directory,
        batch=True,
        resume=True,
        release=17,
        max_iterations=4,
        max_tool_calls=6,
        max_replans=2,
    ))

    summary = json.loads(
        (output_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert summary["recorded_questions"] == 2
    assert summary["successful_questions"] == 2
    assert summary["current_question_id"] is None
    assert [item["execution"] for item in summary["results"]] == [
        "skipped",
        "skipped",
    ]

"""Run the real workflow from frozen Planner artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import uuid
from typing import Any

from pydantic import TypeAdapter

from specrag.models import ResearchBudget, ResearchPlan
from specrag.orchestration import ResearchState
from specrag.orchestration.application import build_research_application


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("planning_artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="treat planning_artifact and output as input/output directories",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="in batch mode, skip valid artifacts already present in output",
    )
    parser.add_argument("--release", type=int, default=17)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--max-gap-tool-calls", type=int, default=6)
    parser.add_argument("--max-replans", type=int, default=2)
    return parser


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _result_from_artifact(
    artifact: dict[str, Any],
    *,
    execution: str,
) -> dict[str, Any]:
    state = artifact["state"]
    budget = state["budget"]
    return {
        "question_id": str(artifact["question"]["id"]),
        "execution": execution,
        "stop_reason": state["stop_reason"],
        "evidence_sufficient": state["evidence_sufficient"],
        "tool_calls_used": budget["tool_calls_used"],
        "tool_calls_used_by_task": budget["tool_calls_used_by_task"],
    }


def _load_resume_result(
    path: Path,
    *,
    expected_question_id: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact["schema_version"] != "specrag.complex-researcher-evaluation/v1":
            return None
        if str(artifact["question"]["id"]) != expected_question_id:
            return None
        return _result_from_artifact(artifact, execution="skipped")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _run_one(
    planning_path: Path,
    output_path: Path,
    *,
    application: Any,
    max_iterations: int,
    max_tool_calls: int,
    max_gap_tool_calls: int,
    max_replans: int,
) -> dict[str, Any]:
    planning = json.loads(planning_path.read_text(encoding="utf-8"))
    question = planning["question"]
    plan = ResearchPlan.model_validate(planning["plan"])
    started = datetime.now(UTC)
    state = application.run(
        research_id=str(question["id"]),
        question=str(question["text"]),
        thread_id=f"{question['id']}-{uuid.uuid4().hex}",
        budget=ResearchBudget(
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            max_gap_tool_calls=max_gap_tool_calls,
            max_replans=max_replans,
        ),
        plan=plan,
    )
    artifact = {
        "schema_version": "specrag.complex-researcher-evaluation/v1",
        "planning_artifact": planning_path.as_posix(),
        "question": question,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "state": TypeAdapter(ResearchState).dump_python(state, mode="json"),
    }
    _atomic_write_json(output_path, artifact)
    return _result_from_artifact(artifact, execution="executed")


def _write_batch_summary(
    output_directory: Path,
    *,
    planning_directory: Path,
    started_at: str,
    total_questions: int,
    current_question_id: str | None,
    results: list[dict[str, Any]],
) -> None:
    successful = sum(
        item.get("stop_reason") == "completed"
        and item.get("evidence_sufficient") is True
        for item in results
    )
    summary = {
        "schema_version": "specrag.complex-researcher-batch/v1",
        "planning_directory": planning_directory.as_posix(),
        "started_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "total_questions": total_questions,
        "recorded_questions": len(results),
        "successful_questions": successful,
        "current_question_id": current_question_id,
        "results": results,
    }
    _atomic_write_json(output_directory / "summary.json", summary)


def _run_batch(args: argparse.Namespace) -> int:
    if not args.planning_artifact.is_dir():
        raise ValueError("--batch requires planning_artifact to be a directory")
    if args.output.exists() and not args.output.is_dir():
        raise ValueError("--batch requires output to be a directory")
    planning_paths = sorted(args.planning_artifact.glob("r17-qa-*.json"))
    if not planning_paths:
        raise ValueError("planning directory contains no r17-qa-*.json artifacts")

    args.output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    results: list[dict[str, Any]] = []
    application: Any | None = None
    for planning_path in planning_paths:
        planning = json.loads(planning_path.read_text(encoding="utf-8"))
        question_id = str(planning["question"]["id"])
        output_path = args.output / planning_path.name
        if args.resume:
            existing = _load_resume_result(
                output_path,
                expected_question_id=question_id,
            )
            if existing is not None:
                results.append(existing)
                print(json.dumps(existing, ensure_ascii=False), flush=True)
                _write_batch_summary(
                    args.output,
                    planning_directory=args.planning_artifact,
                    started_at=started_at,
                    total_questions=len(planning_paths),
                    current_question_id=None,
                    results=results,
                )
                continue

        _write_batch_summary(
            args.output,
            planning_directory=args.planning_artifact,
            started_at=started_at,
            total_questions=len(planning_paths),
            current_question_id=question_id,
            results=results,
        )
        if application is None:
            application = build_research_application(release=args.release)
        try:
            result = _run_one(
                planning_path,
                output_path,
                application=application,
                max_iterations=args.max_iterations,
                max_tool_calls=args.max_tool_calls,
                max_gap_tool_calls=getattr(args, "max_gap_tool_calls", 6),
                max_replans=args.max_replans,
            )
        except Exception as exc:  # keep the remaining batch resumable
            result = {
                "question_id": question_id,
                "execution": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        _write_batch_summary(
            args.output,
            planning_directory=args.planning_artifact,
            started_at=started_at,
            total_questions=len(planning_paths),
            current_question_id=None,
            results=results,
        )

    return 0 if all(
        item.get("stop_reason") == "completed"
        and item.get("evidence_sufficient") is True
        for item in results
    ) else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume and not args.batch:
        raise ValueError("--resume is only valid with --batch")
    if args.batch:
        return _run_batch(args)

    application = build_research_application(release=args.release)
    result = _run_one(
        args.planning_artifact,
        args.output,
        application=application,
        max_iterations=args.max_iterations,
        max_tool_calls=args.max_tool_calls,
        max_gap_tool_calls=args.max_gap_tool_calls,
        max_replans=args.max_replans,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["stop_reason"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())

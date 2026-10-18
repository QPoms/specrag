"""Draft and summarize binary end-to-end Task Success evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid
from typing import Any, Mapping

from specrag.config import load_config
from specrag.evaluation import (
    TaskSuccessEvaluation,
    TaskSuccessReviewDecision,
    TaskSuccessSemanticDraft,
    apply_task_success_review,
    evaluate_task_success,
    summarize_end_to_end,
)
from specrag.evaluation.component_metrics import load_component_annotations
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.prompts import (
    END_TO_END_EVALUATION_INSTRUCTIONS,
    END_TO_END_EVALUATION_PROMPT_VERSION,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate complete-workflow Task Success and efficiency."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft", help="create proposed per-case judgments")
    draft.add_argument("run_directory", type=Path)
    draft.add_argument("qa_file", type=Path)
    draft.add_argument("annotation_file", type=Path)
    draft.add_argument("output_directory", type=Path)
    draft.add_argument("--id", dest="identifiers", action="append")
    draft.add_argument("--resume", action="store_true")

    review = commands.add_parser(
        "review", help="apply auditable human decisions to proposed judgments"
    )
    review.add_argument("evaluation_directory", type=Path)
    review.add_argument("decision_file", type=Path)

    summary = commands.add_parser(
        "summarize", help="summarize only human-reviewed judgments"
    )
    summary.add_argument("evaluation_directory", type=Path)
    return parser


def _records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        question_id = str(value["id"])
        if question_id in records:
            raise ValueError(f"duplicate question ID on line {line_number}")
        records[question_id] = value
    return records


def _review_decisions(path: Path) -> dict[str, TaskSuccessReviewDecision]:
    decisions: dict[str, TaskSuccessReviewDecision] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        decision = TaskSuccessReviewDecision.model_validate_json(line)
        if decision.question_id in decisions:
            raise ValueError(
                f"duplicate review decision on line {line_number}: "
                f"{decision.question_id}"
            )
        decisions[decision.question_id] = decision
    if not decisions:
        raise ValueError("review decision file is empty")
    return decisions


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _batch_manifest(
    args: argparse.Namespace,
    identifiers: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": "specrag.end-to-end-evaluation-batch/v1",
        "run_directory": args.run_directory.resolve().as_posix(),
        "qa_file": args.qa_file.resolve().as_posix(),
        "qa_file_sha256": hashlib.sha256(args.qa_file.read_bytes()).hexdigest(),
        "annotation_file": args.annotation_file.resolve().as_posix(),
        "annotation_file_sha256": hashlib.sha256(
            args.annotation_file.read_bytes()
        ).hexdigest(),
        "question_ids": list(identifiers),
    }


def _freeze_batch_manifest(
    args: argparse.Namespace,
    identifiers: tuple[str, ...],
) -> None:
    path = args.output_directory / "manifest.json"
    expected = _batch_manifest(args, identifiers)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError(
                "evaluation batch manifest conflicts with the requested inputs"
            )
        return
    _write_json(path, expected)


def _evaluation_paths(directory: Path) -> tuple[Path, ...]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("evaluation directory is missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identifiers = tuple(str(value) for value in manifest.get("question_ids", ()))
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("evaluation batch manifest has invalid question IDs")
    expected = {f"{question_id}.json" for question_id in identifiers}
    actual = {path.name for path in directory.glob("r17-qa-*.json")}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError("evaluation batch is incomplete (" + "; ".join(details) + ")")
    return tuple(directory / f"{question_id}.json" for question_id in identifiers)


def _state(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    value = artifact.get("state", {})
    return value if isinstance(value, Mapping) else {}


def _semantic_payload(
    artifact: Mapping[str, Any],
    qa: Mapping[str, Any],
    answer_units: tuple[object, ...],
) -> dict[str, Any]:
    state = _state(artifact)
    synthesis_value = state.get("synthesis") or artifact.get("synthesis") or {}
    synthesis = synthesis_value if isinstance(synthesis_value, Mapping) else {}
    candidates = {
        str(item["evidence_id"]): item
        for item in state.get("candidate_evidence", ())
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    cited_ids = tuple(str(value) for value in synthesis.get("evidence_ids", ()))
    return {
        "question_id": qa["id"],
        "original_question": qa["question"],
        "expected_answer": qa["expected_answer"],
        "expected_facts": qa["expected_facts"],
        "reviewed_answer_units": [
            item.model_dump(mode="json") for item in answer_units
        ],
        "generated_answer": synthesis.get("answer") or state.get("final_answer"),
        "cited_evidence_ids": cited_ids,
        "cited_evidence": [
            candidates[evidence_id]
            for evidence_id in cited_ids if evidence_id in candidates
        ],
    }


def _runtime_failed_semantic() -> TaskSuccessSemanticDraft:
    return TaskSuccessSemanticDraft(
        core_conclusion_answered=False,
        no_material_error=False,
        no_material_unsupported_conclusion=False,
        core_citations_correct=False,
        rationale=(
            "The complete workflow did not produce a normal final answer; "
            "semantic answer gates were not evaluated."
        ),
    )


def _draft(args: argparse.Namespace) -> int:
    qa = _records(args.qa_file)
    annotations = load_component_annotations(args.annotation_file)
    identifiers = tuple(args.identifiers or sorted(qa))
    missing = tuple(
        question_id for question_id in identifiers
        if question_id not in qa or question_id not in annotations
    )
    if missing:
        raise ValueError("unknown or unannotated question IDs: " + ", ".join(missing))
    proposed_annotations = tuple(
        question_id for question_id in identifiers
        if annotations[question_id].review_status != "reviewed"
    )
    if proposed_annotations:
        raise ValueError(
            "end-to-end evaluation requires reviewed Answer Units: "
            + ", ".join(proposed_annotations)
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    _freeze_batch_manifest(args, identifiers)
    agent: LLMApiStructuredAgent | None = None
    error_count = 0
    for question_id in identifiers:
        artifact_path = args.run_directory / f"{question_id}.json"
        output_path = args.output_directory / f"{question_id}.json"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing Run artifact: {artifact_path}")
        if args.resume and output_path.is_file():
            existing = TaskSuccessEvaluation.model_validate_json(
                output_path.read_text(encoding="utf-8")
            )
            current_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if existing.source_artifact_sha256 != current_sha256:
                raise ValueError(
                    f"stale resumed evaluation for {question_id}: Run hash changed"
                )
            print(json.dumps({
                "question_id": existing.question_id,
                "status": "resumed",
                "review_status": existing.review_status,
                "task_success": existing.task_success,
            }, ensure_ascii=False), flush=True)
            continue
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            state = _state(artifact)
            complete = (
                state.get("stop_reason") == "completed"
                and bool(state.get("synthesis") or state.get("final_answer"))
            )
            judge_model = None
            if complete:
                if agent is None:
                    agent = LLMApiStructuredAgent(
                        role="summary",
                        reasoning_effort="high",
                        config=load_config(),
                    )
                response = agent.parse(
                    purpose="evaluate end-to-end Task Success",
                    prompt_version=END_TO_END_EVALUATION_PROMPT_VERSION,
                    instructions=END_TO_END_EVALUATION_INSTRUCTIONS,
                    input_payload=_semantic_payload(
                        artifact,
                        qa[question_id],
                        annotations[question_id].answer_units,
                    ),
                    output_type=TaskSuccessSemanticDraft,
                )
                semantic = response.output
                judge_model = response.receipt.model
            else:
                semantic = _runtime_failed_semantic()
            evaluation = evaluate_task_success(
                artifact_path,
                qa[question_id],
                semantic,
                judge_prompt_version=(
                    END_TO_END_EVALUATION_PROMPT_VERSION if complete else None
                ),
                judge_model=judge_model,
            )
            _write_json(output_path, evaluation.model_dump(mode="json"))
            result = {
                "question_id": question_id,
                "status": "completed",
                "review_status": evaluation.review_status,
                "task_success": evaluation.task_success,
                "failure_reasons": evaluation.failure_reasons,
                "total_tokens": evaluation.efficiency.total_tokens,
                "llm_call_count": evaluation.efficiency.llm_call_count,
                "tool_call_count": evaluation.efficiency.tool_call_count,
            }
        except Exception as exc:  # noqa: BLE001 - keep the batch resumable
            error_count += 1
            result = {
                "question_id": question_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:1600]}",
            }
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 1 if error_count else 0


def _summarize(directory: Path) -> int:
    paths = _evaluation_paths(directory)
    evaluations = tuple(
        TaskSuccessEvaluation.model_validate_json(path.read_text(encoding="utf-8"))
        for path in paths
    )
    summary = summarize_end_to_end(evaluations)
    _write_json(directory / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _review(directory: Path, decision_file: Path) -> int:
    decisions = _review_decisions(decision_file)
    for question_id, decision in decisions.items():
        path = directory / f"{question_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing proposed evaluation: {path}")
        evaluation = TaskSuccessEvaluation.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        reviewed = apply_task_success_review(evaluation, decision)
        _write_json(path, reviewed.model_dump(mode="json"))
        print(json.dumps({
            "question_id": question_id,
            "review_status": reviewed.review_status,
            "task_success": reviewed.task_success,
            "failure_reasons": reviewed.failure_reasons,
        }, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "draft":
            return _draft(args)
        if args.command == "review":
            return _review(args.evaluation_directory, args.decision_file)
        return _summarize(args.evaluation_directory)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"end-to-end evaluation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

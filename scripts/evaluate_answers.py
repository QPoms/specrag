"""Run deterministic answer/evidence evaluation over synthesis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from specrag.evaluation import (
    AnswerEvaluation,
    evaluate_answer_artifact,
    summarize_answer_evaluations,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate answer citation and gold-target alignment."
    )
    parser.add_argument("synthesis_directory", type=Path)
    parser.add_argument("qa_file", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def _qa_records(path: Path) -> dict[str, dict[str, object]]:
    return {
        str(item["id"]): item
        for item in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    qa = _qa_records(args.qa_file)
    evaluations: list[AnswerEvaluation] = []
    for artifact_path in sorted(args.synthesis_directory.glob("r17-qa-*.json")):
        output_path = args.output_directory / artifact_path.name
        if args.resume and output_path.is_file():
            evaluation = AnswerEvaluation.model_validate_json(
                output_path.read_text(encoding="utf-8")
            )
        else:
            question_id = artifact_path.stem
            try:
                evaluation = evaluate_answer_artifact(
                    artifact_path,
                    qa[question_id],
                )
            except Exception as exc:  # noqa: BLE001 - persist bounded failure
                evaluation = AnswerEvaluation(
                    question_id=question_id,
                    difficulty=str(qa.get(question_id, {}).get("difficulty", "unknown")),
                    reasoning_scope=str(
                        qa.get(question_id, {}).get("reasoning_scope", "unknown")
                    ),
                    status="error",
                    reason=f"{type(exc).__name__}: {str(exc)[:1600]}",
                )
            _write(output_path, evaluation.model_dump(mode="json"))
        evaluations.append(evaluation)
        print(json.dumps({
            "question_id": evaluation.question_id,
            "status": evaluation.status,
            "target_recall": evaluation.target_recall,
            "citation_validity": evaluation.citation_validity,
        }, ensure_ascii=False))

    summary = summarize_answer_evaluations(tuple(evaluations))
    _write(args.output_directory / "summary.json", summary)
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    sys.exit(main())

"""Run deterministic component metrics from frozen artifacts and judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from specrag.evaluation.component_metrics import (
    PlannerEvaluation,
    RouterEvaluation,
    evaluate_planner_artifact,
    evaluate_router_artifact,
    load_component_annotations,
    load_planner_judgments,
    summarize_planner_evaluations,
    summarize_router_evaluations,
)
from specrag.evaluation.component_models import ComponentQuestionAnnotation
from specrag.evaluation.component_evidence_metrics import (
    evaluate_final_evidence_artifact,
    evaluate_researcher_artifact,
    evaluate_retrieval_artifact,
    evaluate_synthesizer_artifact,
)
from specrag.evaluation.component_evidence_summary import summarize_evaluations
from specrag.evaluation.component_evidence_io import (
    load_final_evidence_judgments,
    load_researcher_judgments,
    load_retrieval_judgments,
    load_synthesizer_judgments,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one SpecRAG component from frozen artifacts."
    )
    subparsers = parser.add_subparsers(dest="component", required=True)

    seed_router = subparsers.add_parser("seed-router")
    seed_router.add_argument("--questions", required=True, type=Path)
    seed_router.add_argument("--route-review", required=True, type=Path)
    seed_router.add_argument("--base-annotations", type=Path)
    seed_router.add_argument("--output", required=True, type=Path)

    router = subparsers.add_parser("router")
    _common_arguments(router)

    planner = subparsers.add_parser("planner")
    _common_arguments(planner)
    planner.add_argument(
        "--judgments",
        required=True,
        type=Path,
        help="JSONL structured Planner semantic judgments",
    )
    for name in ("retriever", "researcher", "judge", "synthesizer"):
        component = subparsers.add_parser(name)
        _common_arguments(component)
        component.add_argument(
            "--judgments",
            required=True,
            type=Path,
            help=f"JSONL structured {name} semantic judgments",
        )
    return parser


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        action="append",
        help="artifact directory; repeat to combine non-overlapping frozen batches",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-proposed",
        action="store_true",
        help="allow calibration runs using annotations not yet human-reviewed",
    )


def _write_json(path: Path, value: object) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite evaluation report: {path}")
        return
    path.write_text(serialized, encoding="utf-8")


def _write_text(path: Path, serialized: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != serialized:
            raise FileExistsError(f"refusing to overwrite curated annotations: {path}")
        return
    path.write_text(serialized, encoding="utf-8")


def _seed_router_annotations(args: argparse.Namespace) -> int:
    questions = _load_jsonl_objects(args.questions, key="id")
    reviews = _load_jsonl_objects(args.route_review, key="id")
    if set(questions) != set(reviews):
        raise ValueError(
            "route review must cover exactly the question set; "
            f"missing={sorted(set(questions) - set(reviews))}, "
            f"extra={sorted(set(reviews) - set(questions))}"
        )
    base = (
        load_component_annotations(args.base_annotations)
        if args.base_annotations
        else {}
    )
    if not set(base).issubset(questions):
        raise ValueError("base annotations contain questions outside the question set")
    annotations: list[ComponentQuestionAnnotation] = []
    for question_id, question in questions.items():
        review = reviews[question_id]
        route = str(review.get("proposed_route", ""))
        if route not in {"single_task", "decompose"}:
            raise ValueError(f"unsupported proposed route for {question_id}: {route!r}")
        if review.get("review_status") != "proposed":
            raise ValueError(
                "legacy route seeds must remain proposed until auditable review: "
                f"{question_id}"
            )
        existing = base.get(question_id)
        if existing is not None:
            if existing.expected_route != route:
                raise ValueError(f"base route differs from review seed for {question_id}")
            annotations.append(existing)
            continue
        annotations.append(ComponentQuestionAnnotation(
            question_id=question_id,
            review_status="proposed",
            expected_route=route,
            route_rationale=str(review.get("rationale", "")).strip(),
        ))
        if str(question.get("split", "")) != "development":
            raise ValueError(f"component seed requires development question: {question_id}")
    serialized = "".join(
        item.model_dump_json() + "\n" for item in annotations
    )
    _write_text(args.output, serialized)
    summary = {
        "schema_version": "specrag.component-annotation-seed-summary/v1",
        "question_count": len(annotations),
        "single_task_count": sum(
            item.expected_route == "single_task" for item in annotations
        ),
        "decompose_count": sum(
            item.expected_route == "decompose" for item in annotations
        ),
        "records_with_research_units": sum(
            bool(item.research_units) for item in annotations
        ),
        "records_with_evidence_units": sum(
            bool(item.evidence_units) for item in annotations
        ),
        "records_with_answer_units": sum(
            bool(item.answer_units) for item in annotations
        ),
        "output": args.output.as_posix(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_jsonl_objects(path: Path, *, key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        record_key = str(record.get(key, ""))
        if not record_key or record_key in records:
            raise ValueError(f"invalid or duplicate {key} at line {line_number}")
        records[record_key] = record
    if not records:
        raise ValueError(f"JSONL input is empty: {path}")
    return records


def _require_reviewed(statuses: set[str], *, allow_proposed: bool) -> None:
    if "proposed" in statuses and not allow_proposed:
        raise ValueError(
            "proposed annotations or judgments require --allow-proposed; "
            "their reports are calibration-only"
        )


def _router(args: argparse.Namespace) -> int:
    annotations = load_component_annotations(args.annotations)
    _require_reviewed(
        {item.review_status for item in annotations.values()},
        allow_proposed=args.allow_proposed,
    )
    evaluations: list[RouterEvaluation] = []
    errors: list[dict[str, str]] = []
    for question_id, annotation in sorted(annotations.items()):
        try:
            artifact_path = _artifact_path(args.artifacts, question_id)
            evaluation = evaluate_router_artifact(artifact_path, annotation)
        except Exception as exc:  # persist bounded per-case failure
            errors.append(_error(question_id, exc))
            continue
        evaluations.append(evaluation)
        _write_json(
            args.output / f"{question_id}.json",
            evaluation.model_dump(mode="json"),
        )

    summary: dict[str, Any] = (
        summarize_router_evaluations(tuple(evaluations)) if evaluations else {
            "schema_version": "specrag.router-evaluation-summary/v1",
            "question_count": 0,
            "calibration_only": True,
        }
    )
    summary["requested_question_count"] = len(annotations)
    summary["error_count"] = len(errors)
    summary["errors"] = errors
    _add_input_metadata(summary, args)
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


def _planner(args: argparse.Namespace) -> int:
    annotations = load_component_annotations(args.annotations)
    judgments = load_planner_judgments(args.judgments)
    missing_annotations = set(judgments) - set(annotations)
    if missing_annotations:
        raise ValueError(
            f"Planner judgments lack annotations: {sorted(missing_annotations)}"
        )
    _require_reviewed(
        {
            *(annotations[item].review_status for item in judgments),
            *(item.review_status for item in judgments.values()),
        },
        allow_proposed=args.allow_proposed,
    )
    evaluations: list[PlannerEvaluation] = []
    errors: list[dict[str, str]] = []
    for question_id, judgment in sorted(judgments.items()):
        try:
            artifact_path = _artifact_path(args.artifacts, question_id)
            evaluation = evaluate_planner_artifact(
                artifact_path,
                annotations[question_id],
                judgment,
            )
        except Exception as exc:  # persist bounded per-case failure
            errors.append(_error(question_id, exc))
            continue
        evaluations.append(evaluation)
        _write_json(
            args.output / f"{question_id}.json",
            evaluation.model_dump(mode="json"),
        )

    summary: dict[str, Any] = (
        summarize_planner_evaluations(tuple(evaluations)) if evaluations else {
            "schema_version": "specrag.planner-evaluation-summary/v1",
            "question_count": 0,
            "calibration_only": True,
        }
    )
    summary["requested_question_count"] = len(judgments)
    summary["error_count"] = len(errors)
    summary["errors"] = errors
    _add_input_metadata(summary, args)
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


def _retriever(args: argparse.Namespace) -> int:
    judgments = load_retrieval_judgments(args.judgments)
    _require_reviewed(
        {item.review_status for item in judgments.values()},
        allow_proposed=args.allow_proposed,
    )
    evaluations = []
    errors: list[dict[str, str]] = []
    for (question_id, call_id), judgment in sorted(judgments.items()):
        try:
            artifact_path = _artifact_path(args.artifacts, question_id)
            evaluation = evaluate_retrieval_artifact(artifact_path, judgment)
        except Exception as exc:  # persist bounded per-case failure
            errors.append(_error(f"{question_id}:{call_id}", exc))
            continue
        evaluations.append(evaluation)
        _write_json(
            args.output / f"{question_id}--{call_id}.json",
            evaluation.model_dump(mode="json"),
        )
    return _finish_evidence_component(
        args, "retriever", evaluations, len(judgments), errors
    )


def _researcher(args: argparse.Namespace) -> int:
    annotations = load_component_annotations(args.annotations)
    judgments = load_researcher_judgments(args.judgments)
    _require_known_annotations(annotations, {key[0] for key in judgments})
    _require_reviewed(
        {
            *(annotations[key[0]].review_status for key in judgments),
            *(item.review_status for item in judgments.values()),
        },
        allow_proposed=args.allow_proposed,
    )
    evaluations = []
    errors: list[dict[str, str]] = []
    for (question_id, task_id), judgment in sorted(judgments.items()):
        try:
            artifact_path = _artifact_path(args.artifacts, question_id)
            evaluation = evaluate_researcher_artifact(
                artifact_path,
                annotations[question_id],
                judgment,
            )
        except Exception as exc:  # persist bounded per-case failure
            errors.append(_error(f"{question_id}:{task_id}", exc))
            continue
        evaluations.append(evaluation)
        _write_json(
            args.output / f"{question_id}--{task_id}.json",
            evaluation.model_dump(mode="json"),
        )
    return _finish_evidence_component(
        args, "researcher", evaluations, len(judgments), errors
    )


def _question_level_evidence_component(
    args: argparse.Namespace,
    *,
    component: str,
) -> int:
    annotations = load_component_annotations(args.annotations)
    if component == "judge":
        judgments = load_final_evidence_judgments(args.judgments)
        evaluator = evaluate_final_evidence_artifact
    else:
        judgments = load_synthesizer_judgments(args.judgments)
        evaluator = evaluate_synthesizer_artifact
    _require_known_annotations(annotations, set(judgments))
    _require_reviewed(
        {
            *(annotations[key].review_status for key in judgments),
            *(item.review_status for item in judgments.values()),
        },
        allow_proposed=args.allow_proposed,
    )
    evaluations = []
    errors: list[dict[str, str]] = []
    for question_id, judgment in sorted(judgments.items()):
        try:
            artifact_path = _artifact_path(args.artifacts, question_id)
            evaluation = evaluator(
                artifact_path,
                annotations[question_id],
                judgment,
            )
        except Exception as exc:  # persist bounded per-case failure
            errors.append(_error(question_id, exc))
            continue
        evaluations.append(evaluation)
        _write_json(
            args.output / f"{question_id}.json",
            evaluation.model_dump(mode="json"),
        )
    return _finish_evidence_component(
        args, component, evaluations, len(judgments), errors
    )


def _finish_evidence_component(
    args: argparse.Namespace,
    component: str,
    evaluations: list,
    requested_count: int,
    errors: list[dict[str, str]],
) -> int:
    summary = (
        summarize_evaluations(tuple(evaluations), component)
        if evaluations
        else {
            "schema_version": f"specrag.{component}-evaluation-summary/v1",
            "case_count": 0,
            "calibration_only": True,
        }
    )
    summary["requested_case_count"] = requested_count
    summary["error_count"] = len(errors)
    summary["errors"] = errors
    _add_input_metadata(summary, args)
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


def _require_known_annotations(annotations: dict, question_ids: set[str]) -> None:
    missing = question_ids - set(annotations)
    if missing:
        raise ValueError(f"judgments lack component annotations: {sorted(missing)}")


def _add_input_metadata(summary: dict[str, Any], args: argparse.Namespace) -> None:
    for name in ("annotations", "judgments"):
        path = getattr(args, name, None)
        if path is None:
            continue
        summary[f"{name}_path"] = path.as_posix()
        summary[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def _error(question_id: str, exc: Exception) -> dict[str, str]:
    return {
        "question_id": question_id,
        "error_type": type(exc).__name__,
        "reason": str(exc)[:1600],
    }


def _artifact_path(directories: list[Path], question_id: str) -> Path:
    matches = tuple(
        directory / f"{question_id}.json"
        for directory in directories
        if (directory / f"{question_id}.json").is_file()
    )
    if not matches:
        raise FileNotFoundError(f"no artifact found for {question_id}")
    if len(matches) > 1:
        raise ValueError(
            f"multiple artifacts found for {question_id}: "
            + ", ".join(path.as_posix() for path in matches)
        )
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.component == "seed-router":
        return _seed_router_annotations(args)
    if args.component == "router":
        return _router(args)
    if args.component == "planner":
        return _planner(args)
    if args.component == "retriever":
        return _retriever(args)
    if args.component == "researcher":
        return _researcher(args)
    if args.component in {"judge", "synthesizer"}:
        return _question_level_evidence_component(args, component=args.component)
    raise AssertionError(f"unsupported component {args.component}")


if __name__ == "__main__":
    sys.exit(main())

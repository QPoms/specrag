"""Draft full-set component annotations and Planner judgments with an LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from specrag.config import load_config
from specrag.evaluation.component_metrics import (
    evaluate_planner_artifact,
    load_component_annotations,
    load_planner_judgments,
)
from specrag.evaluation.component_models import (
    ComponentQuestionAnnotation,
    ComponentUnitDraft,
    PlannerJudgmentDraft,
    PlannerSemanticJudgment,
)
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.prompts import (
    COMPONENT_ANNOTATION_INSTRUCTIONS,
    COMPONENT_ANNOTATION_PROMPT_VERSION,
    COMPONENT_PLANNER_JUDGE_INSTRUCTIONS,
    COMPONENT_PLANNER_JUDGE_PROMPT_VERSION,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draft component gold with versioned offline-judge prompts."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    annotations = commands.add_parser("annotations")
    annotations.add_argument("--questions", required=True, type=Path)
    annotations.add_argument("--base-annotations", required=True, type=Path)
    annotations.add_argument("--output-dir", required=True, type=Path)
    annotations.add_argument("--compiled-output", required=True, type=Path)
    annotations.add_argument("--id", dest="identifiers", action="append")
    annotations.add_argument("--resume", action="store_true")

    planner = commands.add_parser("planner")
    planner.add_argument("--annotations", required=True, type=Path)
    planner.add_argument("--base-judgments", type=Path)
    planner.add_argument(
        "--artifacts", required=True, action="append", type=Path
    )
    planner.add_argument("--output-dir", required=True, type=Path)
    planner.add_argument("--compiled-output", required=True, type=Path)
    planner.add_argument("--id", dest="identifiers", action="append")
    planner.add_argument("--resume", action="store_true")
    return parser


def _agent() -> LLMApiStructuredAgent:
    return LLMApiStructuredAgent(
        role="summary",
        reasoning_effort="high",
        config=load_config(),
    )


def _jsonl(path: Path, *, key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        identifier = str(record.get(key, ""))
        if not identifier or identifier in records:
            raise ValueError(f"invalid or duplicate {key} at line {line_number}")
        records[identifier] = record
    return records


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != text:
            raise FileExistsError(f"refusing to overwrite draft artifact: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_once(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def _selected(
    available: set[str], identifiers: list[str] | None
) -> tuple[str, ...]:
    if not identifiers:
        return tuple(sorted(available))
    requested = tuple(dict.fromkeys(identifiers))
    missing = set(requested) - available
    if missing:
        raise ValueError(f"unknown selected IDs: {sorted(missing)}")
    return tuple(requested)


def _sequential(values, prefix: str, field: str = "unit_id") -> bool:
    actual = tuple(str(getattr(item, field)) for item in values)
    expected = tuple(f"{prefix}{index}" for index in range(1, len(values) + 1))
    return actual == expected


def _validate_unit_draft(
    draft: ComponentUnitDraft,
    *,
    expected_route: str,
    expected_fact_count: int | None = None,
) -> None:
    if expected_route == "single_task":
        if len(draft.research_units) != 1 or len(draft.gold_tasks) != 1:
            raise ValueError("single_task draft requires one ResearchUnit and task")
    elif len(draft.research_units) < 2 or len(draft.gold_tasks) < 2:
        raise ValueError("decompose draft requires at least two units and tasks")
    if len(draft.research_units) != len(draft.gold_tasks):
        raise ValueError("draft requires exactly one gold task per ResearchUnit")
    checks = (
        (draft.research_units, "RU", "unit_id"),
        (draft.gold_tasks, "T", "task_id"),
        (draft.evidence_units, "EU", "unit_id"),
        (draft.answer_units, "AU", "unit_id"),
    )
    if not all(_sequential(values, prefix, field) for values, prefix, field in checks):
        raise ValueError("draft IDs must be sequential and use RU/T/EU/AU prefixes")
    for index, task in enumerate(draft.gold_tasks, 1):
        if task.research_unit_ids != (f"RU{index}",):
            raise ValueError("each gold task must map to its same-index ResearchUnit")
    if expected_fact_count is not None:
        if expected_fact_count < 1:
            raise ValueError("annotation input requires expected facts")
        if len(draft.answer_units) != expected_fact_count:
            raise ValueError("draft requires exactly one AnswerUnit per expected fact")
        if not expected_fact_count <= len(draft.evidence_units) <= 2 * expected_fact_count:
            raise ValueError("draft requires one or two EvidenceUnits per expected fact")


def _annotation_from_draft(
    base: ComponentQuestionAnnotation,
    draft: ComponentUnitDraft,
    *,
    expected_fact_count: int | None = None,
) -> ComponentQuestionAnnotation:
    _validate_unit_draft(
        draft,
        expected_route=base.expected_route,
        expected_fact_count=expected_fact_count,
    )
    if (
        base.review_status == "reviewed"
        and base.research_units
        and draft.research_units != base.research_units
    ):
        raise ValueError("draft changed existing reviewed ResearchUnits")
    return base.model_copy(update={
        "review_status": "proposed",
        "reviewed_by": None,
        "reviewed_at": None,
        "research_units": draft.research_units,
        "gold_tasks": draft.gold_tasks,
        "evidence_units": draft.evidence_units,
        "answer_units": draft.answer_units,
    })


def _draft_annotations(args: argparse.Namespace) -> int:
    questions = _jsonl(args.questions, key="id")
    base = load_component_annotations(args.base_annotations)
    if set(base) != set(questions):
        raise ValueError("base annotations must cover exactly the question set")
    proposed = {
        question_id for question_id, item in base.items()
        if item.review_status == "proposed"
    }
    available = set(base) if args.identifiers else proposed
    identifiers = _selected(available, args.identifiers)
    agent = _agent()
    errors: list[dict[str, str]] = []
    generated: dict[str, ComponentQuestionAnnotation] = {}
    for question_id in identifiers:
        output = args.output_dir / f"{question_id}.json"
        if args.resume and output.is_file():
            record = json.loads(output.read_text(encoding="utf-8"))
            generated[question_id] = ComponentQuestionAnnotation.model_validate(
                record["annotation"]
            )
            continue
        try:
            qa = questions[question_id]
            response = agent.parse(
                purpose="draft component semantic units",
                prompt_version=COMPONENT_ANNOTATION_PROMPT_VERSION,
                instructions=COMPONENT_ANNOTATION_INSTRUCTIONS,
                input_payload={
                    "question_id": question_id,
                    "question": qa["question"],
                    "expected_route": base[question_id].expected_route,
                    "expected_answer": qa.get("expected_answer", ""),
                    "expected_facts": qa.get("expected_facts", ()),
                    "evidence_targets": qa.get("evidence_targets", ()),
                    "existing_research_units": [
                        item.model_dump(mode="json")
                        for item in base[question_id].research_units
                    ],
                },
                output_type=ComponentUnitDraft,
            )
            annotation = _annotation_from_draft(
                base[question_id],
                response.output,
                expected_fact_count=len(qa.get("expected_facts", ())),
            )
            generated[question_id] = annotation
            _write_json(output, {
                "schema_version": "specrag.component-annotation-draft/v1",
                "prompt_version": COMPONENT_ANNOTATION_PROMPT_VERSION,
                "receipt": response.receipt.model_dump(mode="json"),
                "annotation": annotation.model_dump(mode="json"),
            })
            print(json.dumps({"id": question_id, "status": "completed"}))
        except Exception as exc:  # noqa: BLE001 - persist bounded batch failure
            errors.append({
                "id": question_id,
                "error": f"{type(exc).__name__}: {str(exc)[:1600]}",
            })
            print(json.dumps({"id": question_id, "status": "error"}))
    if errors:
        _write_json(args.output_dir / "errors.json", errors)
        return 1
    merged = {
        question_id: generated.get(question_id, annotation)
        for question_id, annotation in base.items()
    }
    serialized = "".join(
        item.model_dump_json() + "\n" for item in merged.values()
    )
    _write_once(args.compiled_output, serialized)
    return 0


def _artifact_path(directories: list[Path], question_id: str) -> Path:
    matches = [
        directory / f"{question_id}.json"
        for directory in directories
        if (directory / f"{question_id}.json").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one Planner artifact for {question_id}, found {len(matches)}"
        )
    return matches[0]


def _plan_tasks(path: Path) -> tuple[dict[str, object], ...]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    plan = artifact.get("plan")
    if not isinstance(plan, dict):
        state = artifact.get("state", {})
        plan = state.get("research_plan") if isinstance(state, dict) else None
    if not isinstance(plan, dict) or not plan.get("tasks"):
        raise ValueError(f"Planner artifact contains no tasks: {path}")
    return tuple(plan["tasks"])


def _draft_planner(args: argparse.Namespace) -> int:
    annotations = load_component_annotations(args.annotations)
    base = (
        load_planner_judgments(args.base_judgments)
        if args.base_judgments else {}
    )
    candidates = {
        question_id for question_id, item in annotations.items()
        if item.expected_route == "decompose" and question_id not in base
    }
    identifiers = _selected(candidates, args.identifiers)
    agent = _agent()
    generated: dict[str, PlannerSemanticJudgment] = {}
    errors: list[dict[str, str]] = []
    for question_id in identifiers:
        output = args.output_dir / f"{question_id}.json"
        if args.resume and output.is_file():
            record = json.loads(output.read_text(encoding="utf-8"))
            generated[question_id] = PlannerSemanticJudgment.model_validate(
                record["judgment"]
            )
            continue
        try:
            artifact_path = _artifact_path(args.artifacts, question_id)
            response = agent.parse(
                purpose="judge Planner semantic coverage",
                prompt_version=COMPONENT_PLANNER_JUDGE_PROMPT_VERSION,
                instructions=COMPONENT_PLANNER_JUDGE_INSTRUCTIONS,
                input_payload={
                    "question_id": question_id,
                    "research_units": [
                        item.model_dump(mode="json")
                        for item in annotations[question_id].research_units
                    ],
                    "generated_tasks": _plan_tasks(artifact_path),
                },
                output_type=PlannerJudgmentDraft,
            )
            judgment = PlannerSemanticJudgment(
                question_id=question_id,
                review_status="proposed",
                unit_mappings=response.output.unit_mappings,
                overlapping_task_pairs=response.output.overlapping_task_pairs,
                out_of_scope_task_ids=response.output.out_of_scope_task_ids,
                diagnostic_tags=response.output.diagnostic_tags,
                judge_prompt_version=COMPONENT_PLANNER_JUDGE_PROMPT_VERSION,
                judge_model=response.receipt.model,
            )
            evaluate_planner_artifact(
                artifact_path,
                annotations[question_id],
                judgment,
            )
            generated[question_id] = judgment
            _write_json(output, {
                "schema_version": "specrag.planner-judgment-draft/v1",
                "prompt_version": COMPONENT_PLANNER_JUDGE_PROMPT_VERSION,
                "receipt": response.receipt.model_dump(mode="json"),
                "judgment": judgment.model_dump(mode="json"),
            })
            print(json.dumps({"id": question_id, "status": "completed"}))
        except Exception as exc:  # noqa: BLE001 - persist bounded batch failure
            errors.append({
                "id": question_id,
                "error": f"{type(exc).__name__}: {str(exc)[:1600]}",
            })
            print(json.dumps({"id": question_id, "status": "error"}))
    if errors:
        _write_json(args.output_dir / "errors.json", errors)
        return 1
    merged = {**base, **generated}
    serialized = "".join(
        merged[key].model_dump_json() + "\n" for key in sorted(merged)
    )
    _write_once(args.compiled_output, serialized)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "annotations":
        return _draft_annotations(args)
    if args.command == "planner":
        return _draft_planner(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())

"""Draft evidence-path semantic judgments for manual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from specrag.config import load_config
from specrag.evaluation.component_evidence_io import (
    load_final_evidence_judgments,
    load_researcher_judgments,
    load_synthesizer_judgments,
)
from specrag.evaluation.component_evidence_metrics import (
    _candidate_evidence,
    _deduplicated_final_evidence,
    evaluate_final_evidence_artifact,
    evaluate_researcher_artifact,
    evaluate_synthesizer_artifact,
)
from specrag.evaluation.component_evidence_models import (
    FinalEvidenceJudgment,
    FinalEvidenceJudgmentDraft,
    ResearcherEvidenceJudgment,
    ResearcherQuestionJudgmentDraft,
    SynthesizerJudgment,
    SynthesizerJudgmentDraft,
)
from specrag.evaluation.component_metrics import load_component_annotations
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.prompts import (
    COMPONENT_FINAL_EVIDENCE_JUDGE_INSTRUCTIONS,
    COMPONENT_FINAL_EVIDENCE_JUDGE_PROMPT_VERSION,
    COMPONENT_RESEARCHER_JUDGE_INSTRUCTIONS,
    COMPONENT_RESEARCHER_JUDGE_PROMPT_VERSION,
    COMPONENT_SYNTHESIZER_JUDGE_INSTRUCTIONS,
    COMPONENT_SYNTHESIZER_JUDGE_PROMPT_VERSION,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draft component evidence judgments with versioned prompts."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("researcher", "final", "synthesizer"):
        command = commands.add_parser(name)
        command.add_argument("--annotations", required=True, type=Path)
        command.add_argument("--base-judgments", type=Path)
        command.add_argument("--artifacts", required=True, action="append", type=Path)
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument("--compiled-output", required=True, type=Path)
        command.add_argument("--id", dest="identifiers", action="append")
        command.add_argument("--resume", action="store_true")
    return parser


def _agent() -> LLMApiStructuredAgent:
    return LLMApiStructuredAgent(
        role="summary",
        reasoning_effort="high",
        config=load_config(),
    )


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != text:
            raise FileExistsError(f"refusing to overwrite draft artifact: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_once(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _artifact_path(directories: list[Path], question_id: str) -> Path:
    matches = [
        directory / f"{question_id}.json"
        for directory in directories
        if (directory / f"{question_id}.json").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one component artifact for {question_id}, found {len(matches)}"
        )
    return matches[0]


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"component artifact must be an object: {path}")
    return value


def _state(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    value = artifact.get("state")
    return value if isinstance(value, Mapping) else artifact


def _evidence_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "evidence_id",
        "task_id",
        "specification",
        "version",
        "source_path",
        "document_sha256",
        "heading_path",
        "line_start",
        "line_end",
        "excerpt",
    )
    return {key: item.get(key) for key in fields if item.get(key) is not None}


def _eligible_synthesis_evidence(
    artifact: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    eligible_ids = {
        str(item.get("evidence_id"))
        for item in artifact.get("validated_evidence", ())
        if isinstance(item, Mapping)
        and item.get("source_valid") is True
        and item.get("supports_target") is True
        and item.get("validation_status") == "supported"
    }
    return tuple(
        item
        for item in artifact.get("candidate_evidence", ())
        if isinstance(item, Mapping)
        and str(item.get("evidence_id")) in eligible_ids
    )


def _selected(available: set[str], requested: list[str] | None) -> tuple[str, ...]:
    if not requested:
        return tuple(sorted(available))
    values = tuple(dict.fromkeys(requested))
    missing = set(values) - available
    if missing:
        raise ValueError(f"unknown selected IDs: {sorted(missing)}")
    return values


def _question_text(artifact: Mapping[str, Any]) -> str:
    question = artifact.get("question")
    if isinstance(question, Mapping):
        return str(question.get("question", ""))
    if isinstance(question, str):
        return question
    return str(_state(artifact).get("user_question", ""))


def _researcher(args: argparse.Namespace) -> int:
    annotations = load_component_annotations(args.annotations)
    base = (
        load_researcher_judgments(args.base_judgments)
        if args.base_judgments else {}
    )
    identifiers = _selected(set(annotations), args.identifiers)
    agent = _agent()
    generated: dict[tuple[str, str], ResearcherEvidenceJudgment] = {}
    errors: list[dict[str, str]] = []
    for question_id in identifiers:
        annotation = annotations[question_id]
        missing_tasks = tuple(
            task for task in annotation.gold_tasks
            if (question_id, task.task_id) not in base
        )
        if not missing_tasks:
            continue
        output = args.output_dir / f"{question_id}.json"
        if args.resume and output.is_file():
            record = _read(output)
            for value in record["judgments"]:
                item = ResearcherEvidenceJudgment.model_validate(value)
                generated[(question_id, item.task_id)] = item
            continue
        try:
            artifact_path = _artifact_path(args.artifacts, question_id)
            artifact = _read(artifact_path)
            candidates = _candidate_evidence(artifact)
            task_payloads = []
            for task in missing_tasks:
                research_ids = set(task.research_unit_ids)
                units = [
                    unit.model_dump(mode="json")
                    for unit in annotation.evidence_units
                    if research_ids.intersection(unit.research_unit_ids)
                ]
                task_payloads.append({
                    "task": task.model_dump(mode="json"),
                    "evidence_units": units,
                    "submitted_evidence": [
                        _evidence_payload(item) for item in candidates
                        if str(item.get("task_id", "")) == task.task_id
                    ],
                })
            response = agent.parse(
                purpose="judge Researcher submitted evidence",
                prompt_version=COMPONENT_RESEARCHER_JUDGE_PROMPT_VERSION,
                instructions=COMPONENT_RESEARCHER_JUDGE_INSTRUCTIONS,
                input_payload={
                    "question_id": question_id,
                    "question": _question_text(artifact),
                    "tasks": task_payloads,
                },
                output_type=ResearcherQuestionJudgmentDraft,
            )
            actual_tasks = {task.task_id for task in missing_tasks}
            if {item.task_id for item in response.output.task_judgments} != actual_tasks:
                raise ValueError("draft task judgments do not match requested tasks")
            question_judgments = []
            for draft in response.output.task_judgments:
                judgment = ResearcherEvidenceJudgment(
                    question_id=question_id,
                    task_id=draft.task_id,
                    review_status="proposed",
                    unit_assessments=draft.unit_assessments,
                    item_assessments=draft.item_assessments,
                    diagnostic_tags=draft.diagnostic_tags,
                    judge_prompt_version=COMPONENT_RESEARCHER_JUDGE_PROMPT_VERSION,
                    judge_model=response.receipt.model,
                )
                evaluate_researcher_artifact(
                    artifact_path, annotation, judgment
                )
                generated[(question_id, draft.task_id)] = judgment
                question_judgments.append(judgment.model_dump(mode="json"))
            _write_json(output, {
                "schema_version": "specrag.researcher-judgment-draft/v1",
                "prompt_version": COMPONENT_RESEARCHER_JUDGE_PROMPT_VERSION,
                "receipt": response.receipt.model_dump(mode="json"),
                "judgments": question_judgments,
            })
            print(json.dumps({"id": question_id, "status": "completed"}))
        except Exception as exc:  # noqa: BLE001 - persist bounded batch failure
            errors.append({"id": question_id, "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps({"id": question_id, "status": "error"}))
    return _finish(args, base, generated, errors)


def _final(args: argparse.Namespace) -> int:
    annotations = load_component_annotations(args.annotations)
    base = (
        load_final_evidence_judgments(args.base_judgments)
        if args.base_judgments else {}
    )
    identifiers = _selected(set(annotations) - set(base), args.identifiers)
    agent = _agent()
    generated: dict[str, FinalEvidenceJudgment] = {}
    errors: list[dict[str, str]] = []
    for question_id in identifiers:
        output = args.output_dir / f"{question_id}.json"
        if args.resume and output.is_file():
            record = _read(output)
            generated[question_id] = FinalEvidenceJudgment.model_validate(
                record["judgment"]
            )
            continue
        try:
            artifact_path = _artifact_path(args.artifacts, question_id)
            artifact = _read(artifact_path)
            response = agent.parse(
                purpose="judge whole-question Final Evidence",
                prompt_version=COMPONENT_FINAL_EVIDENCE_JUDGE_PROMPT_VERSION,
                instructions=COMPONENT_FINAL_EVIDENCE_JUDGE_INSTRUCTIONS,
                input_payload={
                    "question_id": question_id,
                    "question": _question_text(artifact),
                    "evidence_units": [
                        item.model_dump(mode="json")
                        for item in annotations[question_id].evidence_units
                    ],
                    "final_evidence": [
                        _evidence_payload(item)
                        for item in _deduplicated_final_evidence(artifact)
                    ],
                },
                output_type=FinalEvidenceJudgmentDraft,
            )
            draft = response.output
            judgment = FinalEvidenceJudgment(
                question_id=question_id,
                review_status="proposed",
                unit_assessments=draft.unit_assessments,
                item_assessments=draft.item_assessments,
                redundant_evidence_ids=draft.redundant_evidence_ids,
                diagnostic_tags=draft.diagnostic_tags,
                judge_prompt_version=COMPONENT_FINAL_EVIDENCE_JUDGE_PROMPT_VERSION,
                judge_model=response.receipt.model,
            )
            evaluate_final_evidence_artifact(
                artifact_path, annotations[question_id], judgment
            )
            generated[question_id] = judgment
            _write_json(output, {
                "schema_version": "specrag.final-evidence-judgment-draft/v1",
                "prompt_version": COMPONENT_FINAL_EVIDENCE_JUDGE_PROMPT_VERSION,
                "receipt": response.receipt.model_dump(mode="json"),
                "judgment": judgment.model_dump(mode="json"),
            })
            print(json.dumps({"id": question_id, "status": "completed"}))
        except Exception as exc:  # noqa: BLE001 - persist bounded batch failure
            errors.append({"id": question_id, "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps({"id": question_id, "status": "error"}))
    return _finish(args, base, generated, errors)


def _synthesizer(args: argparse.Namespace) -> int:
    annotations = load_component_annotations(args.annotations)
    base = (
        load_synthesizer_judgments(args.base_judgments)
        if args.base_judgments else {}
    )
    candidates = (
        set(args.identifiers)
        if args.identifiers
        else set(annotations) - set(base)
    )
    unknown = candidates - set(annotations)
    if unknown:
        raise ValueError(f"unknown selected IDs: {sorted(unknown)}")
    eligible = set()
    for question_id in candidates:
        artifact_path = _artifact_path(args.artifacts, question_id)
        synthesis = _read(artifact_path).get("synthesis")
        if isinstance(synthesis, Mapping) and synthesis.get("answer"):
            eligible.add(question_id)
    identifiers = _selected(eligible, args.identifiers)
    agent = _agent()
    generated: dict[str, SynthesizerJudgment] = {}
    errors: list[dict[str, str]] = []
    for question_id in identifiers:
        output = args.output_dir / f"{question_id}.json"
        if args.resume and output.is_file():
            record = _read(output)
            generated[question_id] = SynthesizerJudgment.model_validate(
                record["judgment"]
            )
            continue
        try:
            artifact_path = _artifact_path(args.artifacts, question_id)
            artifact = _read(artifact_path)
            synthesis = artifact.get("synthesis")
            if not isinstance(synthesis, Mapping) or not synthesis.get("answer"):
                raise ValueError("Synthesizer artifact contains no answer")
            response = agent.parse(
                purpose="judge synthesized answer and citations",
                prompt_version=COMPONENT_SYNTHESIZER_JUDGE_PROMPT_VERSION,
                instructions=COMPONENT_SYNTHESIZER_JUDGE_INSTRUCTIONS,
                input_payload={
                    "question_id": question_id,
                    "question": _question_text(artifact),
                    "answer_units": [
                        item.model_dump(mode="json")
                        for item in annotations[question_id].answer_units
                    ],
                    "synthesis": synthesis,
                    "eligible_final_evidence": [
                        _evidence_payload(item)
                        for item in _eligible_synthesis_evidence(artifact)
                    ],
                },
                output_type=SynthesizerJudgmentDraft,
            )
            draft = response.output
            judgment = SynthesizerJudgment(
                question_id=question_id,
                review_status="proposed",
                answer_unit_assessments=draft.answer_unit_assessments,
                claim_assessments=draft.claim_assessments,
                diagnostic_tags=draft.diagnostic_tags,
                judge_prompt_version=COMPONENT_SYNTHESIZER_JUDGE_PROMPT_VERSION,
                judge_model=response.receipt.model,
            )
            evaluate_synthesizer_artifact(
                artifact_path, annotations[question_id], judgment
            )
            generated[question_id] = judgment
            _write_json(output, {
                "schema_version": "specrag.synthesizer-judgment-draft/v1",
                "prompt_version": COMPONENT_SYNTHESIZER_JUDGE_PROMPT_VERSION,
                "receipt": response.receipt.model_dump(mode="json"),
                "judgment": judgment.model_dump(mode="json"),
            })
            print(json.dumps({"id": question_id, "status": "completed"}))
        except Exception as exc:  # noqa: BLE001 - persist bounded batch failure
            errors.append({"id": question_id, "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps({"id": question_id, "status": "error"}))
    return _finish(args, base, generated, errors)


def _finish(args, base: dict, generated: dict, errors: list[dict[str, str]]) -> int:
    if errors:
        _write_json(
            args.output_dir / (args.compiled_output.stem + ".errors.json"),
            errors,
        )
        return 1
    merged = {**base, **generated}
    serialized = "".join(
        merged[key].model_dump_json() + "\n" for key in sorted(merged)
    )
    _write_once(args.compiled_output, serialized)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "researcher":
        return _researcher(args)
    if args.command == "final":
        return _final(args)
    if args.command == "synthesizer":
        return _synthesizer(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())

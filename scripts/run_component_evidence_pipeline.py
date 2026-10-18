"""Run Judge and Synthesizer from gold-sufficient fixed candidate pools."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from pydantic import BaseModel

from specrag.config import REPO_ROOT, load_config
from specrag.evaluation.component_metrics import load_component_annotations
from specrag.evidence import (
    EvidenceSufficiencyService,
    EvidenceSynthesisService,
    SourceEvidenceValidator,
    SourceValidationService,
)
from specrag.infrastructure.research_assessors import (
    StructuredEvidenceSynthesisGenerator,
    StructuredTaskSufficiencyAssessor,
)
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.models import (
    CandidateEvidence,
    ResearchTask,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)
from specrag.prompts import SUFFICIENCY_PROMPT_VERSION, SYNTHESIS_PROMPT_VERSION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze Judge and Synthesizer from gold-sufficient pools."
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--researcher-artifacts", required=True, type=Path)
    parser.add_argument("--judge-output-dir", required=True, type=Path)
    parser.add_argument("--synthesis-output-dir", required=True, type=Path)
    parser.add_argument("--id", dest="identifiers", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser


def _write_once(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != text:
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            item = json.loads(line)
            values[str(item[key])] = item
    return values


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact must be an object: {path}")
    return value


def _gold_candidates(
    question_id: str,
    tasks: tuple[ResearchTask, ...],
    targets: list[dict[str, Any]],
) -> tuple[CandidateEvidence, ...]:
    return tuple(
        CandidateEvidence(
            evidence_id=f"E:{task.task_id}:GOLD-{index}",
            task_id=task.task_id,
            chunk_id=f"GOLD-{question_id}-{index}",
            specification=str(target["specification"]),
            release=17,
            version=str(target["version"]),
            source_path=str(target["source_path"]),
            document_sha256=str(target["document_sha256"]).upper(),
            heading_path=("Curated exact evidence target",),
            line_start=int(target["line_start"]),
            line_end=int(target["line_end"]),
            excerpt=str(target["excerpt"]),
        )
        for task in tasks
        for index, target in enumerate(targets, 1)
    )


def _candidate_pool(
    artifact: Mapping[str, Any],
    gold: tuple[CandidateEvidence, ...],
) -> tuple[CandidateEvidence, ...]:
    state = artifact["state"]
    active = tuple(
        CandidateEvidence.model_validate(item)
        for item in state.get("candidate_evidence", ())
    )
    known = {item.evidence_id for item in active}
    return (*active, *(item for item in gold if item.evidence_id not in known))


def _oracle_synthesis_inputs(
    tasks: tuple[ResearchTask, ...],
    gold: tuple[CandidateEvidence, ...],
    validation: SourceValidationService,
) -> tuple[
    tuple[ValidatedEvidence, ...],
    tuple[TaskSufficiencyAssessment, ...],
]:
    source_results = validation.validate(tasks, gold).source_validations
    invalid = tuple(item.evidence_id for item in source_results if not item.valid)
    if invalid:
        raise ValueError(
            "curated synthesis evidence failed source validation: "
            + ", ".join(invalid)
        )
    validated = tuple(
        ValidatedEvidence(
            evidence_id=item.evidence_id,
            task_id=item.task_id,
            source_valid=True,
            supports_target=True,
            validation_status="supported",
            rationale="Curated exact evidence target approved for isolated synthesis.",
        )
        for item in gold
    )
    assessments = tuple(
        TaskSufficiencyAssessment(
            task_id=task.task_id,
            status="sufficient",
            supporting_evidence_ids=tuple(
                item.evidence_id for item in gold if item.task_id == task.task_id
            ),
            rationale=(
                "All curated exact evidence targets are available to this gold task; "
                "the fixed input is sufficient by benchmark construction."
            ),
        )
        for task in tasks
    )
    return validated, assessments


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    annotations = load_component_annotations(args.annotations)
    questions = _jsonl(args.questions, "id")
    identifiers = tuple(dict.fromkeys(args.identifiers or sorted(annotations)))
    agent = LLMApiStructuredAgent(
        role="summary", reasoning_effort="high", config=load_config()
    )
    validation = SourceValidationService(SourceEvidenceValidator(REPO_ROOT))
    sufficiency = EvidenceSufficiencyService(
        StructuredTaskSufficiencyAssessor(agent)
    )
    synthesis = EvidenceSynthesisService(
        StructuredEvidenceSynthesisGenerator(agent)
    )
    summaries = []
    for question_id in identifiers:
        judge_output = args.judge_output_dir / f"{question_id}.json"
        synthesis_output = args.synthesis_output_dir / f"{question_id}.json"
        if args.resume and judge_output.is_file() and synthesis_output.is_file():
            summary = {"id": question_id, "status": "resumed"}
            summaries.append(summary)
            print(json.dumps(summary))
            continue
        started = datetime.now(UTC)
        source_path = args.researcher_artifacts / f"{question_id}.json"
        researcher = _read(source_path)
        state = researcher["state"]
        tasks = tuple(
            ResearchTask.model_validate(item) for item in state["research_tasks"]
        )
        gold = _gold_candidates(
            question_id, tasks, questions[question_id]["evidence_targets"]
        )
        candidates = _candidate_pool(researcher, gold)
        source_validation = validation.validate(tasks, candidates)
        result = sufficiency.evaluate(
            tasks,
            candidates,
            source_validation.source_validations,
            original_question=str(questions[question_id]["question"]),
        )
        judge_artifact = {
            "schema_version": "specrag.component-final-evidence-run/v1",
            "source_artifact": source_path.as_posix(),
            "question": {
                "id": question_id,
                "text": questions[question_id]["question"],
            },
            "candidate_pool_contract": {
                "gold_sufficient": True,
                "gold_target_count": len(questions[question_id]["evidence_targets"]),
                "all_targets_replicated_per_gold_task": True,
            },
            "sufficiency_prompt_version": SUFFICIENCY_PROMPT_VERSION,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "state": {
                "research_id": question_id,
                "user_question": questions[question_id]["question"],
                "research_tasks": _jsonable(result.tasks),
                "candidate_evidence": _jsonable(candidates),
                "source_validations": _jsonable(source_validation.source_validations),
                "validated_evidence": _jsonable(result.validated_evidence),
                "sufficiency_assessments": _jsonable(result.assessments),
                "gaps": _jsonable(result.gaps),
                "evidence_sufficient": result.evidence_sufficient,
            },
        }
        _write_once(judge_output, judge_artifact)
        oracle_validated, oracle_assessments = _oracle_synthesis_inputs(
            tasks, gold, validation
        )
        answer = synthesis.synthesize(
            str(questions[question_id]["question"]),
            tasks,
            gold,
            oracle_validated,
            oracle_assessments,
        )
        synthesis_artifact = {
            "schema_version": "specrag.component-synthesis-run/v1",
            "source_artifact": args.annotations.as_posix(),
            "question_id": question_id,
            "status": "completed",
            "input_contract": {
                "judge_independent": True,
                "gold_sufficient": True,
                "all_targets_replicated_per_gold_task": True,
            },
            "synthesis_prompt_version": SYNTHESIS_PROMPT_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "question": questions[question_id]["question"],
            "research_tasks": _jsonable(tasks),
            "candidate_evidence": _jsonable(gold),
            "validated_evidence": _jsonable(oracle_validated),
            "sufficiency_assessments": _jsonable(oracle_assessments),
            "synthesis": answer.model_dump(mode="json"),
            "input_evidence_ids": [item.evidence_id for item in oracle_validated],
        }
        status = "completed"
        _write_once(synthesis_output, synthesis_artifact)
        summary = {
            "id": question_id,
            "status": status,
            "candidate_count": len(candidates),
            "selected_count": len(result.validated_evidence),
            "evidence_sufficient": result.evidence_sufficient,
        }
        summaries.append(summary)
        print(json.dumps(summary))
    if not args.skip_summary:
        summary_value = {
            "schema_version": "specrag.component-evidence-pipeline-summary/v1",
            "question_count": len(identifiers),
            "questions": summaries,
        }
        _write_once(args.judge_output_dir / "summary.json", summary_value)
        _write_once(args.synthesis_output_dir / "summary.json", summary_value)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Draft qrels for one fixed ranked retrieval call per GoldResearchTask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from specrag.config import load_config
from specrag.evaluation.component_evidence_metrics import evaluate_retrieval_artifact
from specrag.evaluation.component_evidence_models import (
    EvidenceLocator,
    RetrievalCallJudgment,
    RetrievalJudgmentDraft,
    RetrievalQrel,
)
from specrag.evaluation.component_metrics import load_component_annotations
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.prompts import (
    COMPONENT_RETRIEVAL_JUDGE_INSTRUCTIONS,
    COMPONENT_RETRIEVAL_JUDGE_PROMPT_VERSION,
)


RANKED_TOOLS = {"hybrid_search", "semantic_search", "keyword_search"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draft fixed-call retrieval qrels.")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--compiled-output", required=True, type=Path)
    parser.add_argument("--id", dest="identifiers", action="append")
    parser.add_argument("--resume", action="store_true")
    return parser


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != text:
            raise FileExistsError(f"refusing to overwrite draft artifact: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _objects(path: Path, key: str) -> dict[str, dict[str, Any]]:
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            value = json.loads(line)
            values[str(value[key])] = value
    return values


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact must be an object: {path}")
    return value


def _locator(value: Mapping[str, Any]) -> EvidenceLocator:
    return EvidenceLocator(
        source_path=str(value["source_path"]),
        document_sha256=str(value["document_sha256"]),
        line_start=int(value["line_start"]),
        line_end=int(value["line_end"]),
    )


def _first_ranked_calls(artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    state = artifact.get("state", artifact)
    calls = state.get("tool_calls", ()) if isinstance(state, Mapping) else ()
    selected = {}
    for call in calls:
        if not isinstance(call, Mapping) or call.get("tool_name") not in RANKED_TOOLS:
            continue
        if not call.get("observed_evidence"):
            continue
        task_id = str(call.get("task_id", ""))
        if task_id and task_id not in selected:
            selected[task_id] = call
    return selected


def _build_judgment(
    *,
    question_id: str,
    call: Mapping[str, Any],
    scope: Mapping[str, Any],
    targets: list[dict[str, Any]],
    results: list[dict[str, Any]],
    draft: RetrievalJudgmentDraft,
    judge_model: str,
) -> RetrievalCallJudgment:
    target_by_id = {item["locator_id"]: item for item in targets}
    result_by_id = {item["candidate_id"]: item for item in results}
    if set(draft.target_locator_ids) - set(target_by_id):
        raise ValueError("draft selected unknown target locator IDs")
    grades = {item.candidate_id: item for item in draft.candidate_grades}
    if set(grades) != set(result_by_id):
        raise ValueError("draft candidate grades do not match ranked results")
    qrels = []
    for index, locator_id in enumerate(draft.target_locator_ids, 1):
        target = target_by_id[locator_id]
        qrels.append(RetrievalQrel(
            qrel_id=f"TARGET-{index}",
            locator=_locator(target),
            relevance=2,
            target=True,
            rationale=f"Reviewed target selected for this GoldResearchTask ({locator_id}).",
        ))
    next_index = 1
    for candidate_id, result in result_by_id.items():
        grade = grades[candidate_id]
        qrels.append(RetrievalQrel(
            qrel_id=f"RESULT-{next_index}",
            locator=_locator(result),
            relevance=grade.relevance,
            rationale=grade.rationale,
        ))
        next_index += 1
    return RetrievalCallJudgment(
        question_id=question_id,
        call_id=str(call["call_id"]),
        task_id=str(call["task_id"]),
        review_status="proposed",
        corpus_scope=dict(scope),
        qrels=tuple(qrels),
        judge_prompt_version=COMPONENT_RETRIEVAL_JUDGE_PROMPT_VERSION,
        judge_model=judge_model,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    annotations = load_component_annotations(args.annotations)
    questions = _objects(args.questions, "id")
    identifiers = tuple(dict.fromkeys(args.identifiers or sorted(annotations)))
    agent = LLMApiStructuredAgent(
        role="summary", reasoning_effort="high", config=load_config()
    )
    judgments = {}
    errors = []
    for question_id in identifiers:
        artifact_path = args.artifacts / f"{question_id}.json"
        artifact = _read(artifact_path)
        calls = _first_ranked_calls(artifact)
        annotation = annotations[question_id]
        for task in annotation.gold_tasks:
            key = (question_id, task.task_id)
            output = args.output_dir / f"{question_id}--{task.task_id}.json"
            if args.resume and output.is_file():
                record = _read(output)
                item = RetrievalCallJudgment.model_validate(record["judgment"])
                judgments[(item.question_id, item.call_id)] = item
                continue
            try:
                call = calls[task.task_id]
                research_ids = set(task.research_unit_ids)
                units = [
                    unit.model_dump(mode="json")
                    for unit in annotation.evidence_units
                    if research_ids.intersection(unit.research_unit_ids)
                ]
                targets = [
                    {"locator_id": f"ET{index}", **item}
                    for index, item in enumerate(
                        questions[question_id]["evidence_targets"], 1
                    )
                ]
                results = [
                    {"candidate_id": f"R{index}", **dict(item)}
                    for index, item in enumerate(call.get("observed_evidence", ()), 1)
                ]
                if not results:
                    raise ValueError("fixed ranked call has no observed results")
                response = agent.parse(
                    purpose="grade fixed retrieval call",
                    prompt_version=COMPONENT_RETRIEVAL_JUDGE_PROMPT_VERSION,
                    instructions=COMPONENT_RETRIEVAL_JUDGE_INSTRUCTIONS,
                    input_payload={
                        "question_id": question_id,
                        "task": task.model_dump(mode="json"),
                        "evidence_units": units,
                        "gold_target_locators": targets,
                        "ranked_request": {
                            field: call.get(field) for field in (
                                "call_id", "tool_name", "lexical_query",
                                "semantic_query", "target_specifications",
                            )
                        },
                        "ranked_candidates": results,
                    },
                    output_type=RetrievalJudgmentDraft,
                )
                judgment = _build_judgment(
                    question_id=question_id,
                    call=call,
                    scope=dict(artifact.get("retrieval_scope", {})),
                    targets=targets,
                    results=results,
                    draft=response.output,
                    judge_model=response.receipt.model,
                )
                evaluate_retrieval_artifact(artifact_path, judgment)
                judgments[(question_id, judgment.call_id)] = judgment
                _write_once(output, json.dumps({
                    "schema_version": "specrag.retrieval-judgment-draft/v1",
                    "prompt_version": COMPONENT_RETRIEVAL_JUDGE_PROMPT_VERSION,
                    "receipt": response.receipt.model_dump(mode="json"),
                    "task_id": task.task_id,
                    "judgment": judgment.model_dump(mode="json"),
                }, ensure_ascii=False, indent=2) + "\n")
                print(json.dumps({"id": f"{question_id}:{task.task_id}", "status": "completed"}))
            except Exception as exc:  # noqa: BLE001 - persist bounded failure
                errors.append({"id": f"{question_id}:{task.task_id}", "error": f"{type(exc).__name__}: {exc}"})
                print(json.dumps({"id": f"{question_id}:{task.task_id}", "status": "error"}))
    if errors:
        error_output = args.output_dir / (
            args.compiled_output.stem + ".errors.json"
        )
        _write_once(error_output, json.dumps(errors, indent=2) + "\n")
        return 1
    serialized = "".join(
        judgments[key].model_dump_json() + "\n" for key in sorted(judgments)
    )
    _write_once(args.compiled_output, serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())

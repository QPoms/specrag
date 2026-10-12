"""Execute frozen Researcher first turns and score source-target recall."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from specrag.agents.models import ResearcherActionPlan
from specrag.agents.researcher import build_retrieval_tool_request
from specrag.agents.state import EvidenceLedger
from specrag.config import AppConfig, REPO_ROOT, load_config
from specrag.retrieval.hybrid import HybridMarkdownRetriever
from specrag.tools import ToolEvidence, build_retrieval_tool_registry


QUESTION_SET = REPO_ROOT / "data" / "QA" / "questions.jsonl"


def _load_question_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        records[str(record["id"])] = record
    return records


def _target_matches(
    target: Mapping[str, Any],
    evidence: ToolEvidence,
) -> bool:
    source_path = str(target["source_path"]).replace("\\", "/").casefold()
    same_path = evidence.source_path.replace("\\", "/").casefold() == source_path
    same_hash = evidence.document_sha256.casefold() == str(
        target["document_sha256"]
    ).casefold()
    overlaps = (
        evidence.line_start <= int(target["line_end"])
        and int(target["line_start"]) <= evidence.line_end
    )
    return same_path and same_hash and overlaps


def run_first_turn_evaluation(
    planning_directory: Path,
    output_directory: Path,
    *,
    release: int = 17,
    question_set: Path = QUESTION_SET,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Execute accepted first-turn actions without making another model call."""

    if release < 1:
        raise ValueError("retrieval Release must be positive")
    artifact_paths = tuple(sorted(planning_directory.glob("r17-qa-*.json")))
    if not artifact_paths:
        raise ValueError("planning directory contains no question artifacts")

    app_config = config or load_config()
    retrieval_config = app_config.retrieval.model_copy(update={
        "version_policy": "highest_version_within_release",
        "target_release": release,
    })
    retriever = HybridMarkdownRetriever(retrieval_config)
    registry = build_retrieval_tool_registry(retriever)
    questions = _load_question_records(question_set)
    output_directory.mkdir(parents=True, exist_ok=True)
    question_results: list[dict[str, Any]] = []

    for artifact_path in artifact_paths:
        raw_artifact = artifact_path.read_bytes()
        artifact = json.loads(raw_artifact)
        question_id = str(artifact["question"]["id"])
        first_turns = artifact["researcher"]["first_turns"]
        if len(first_turns) != 1:
            raise ValueError(
                f"{question_id} must contain exactly one ready simple task"
            )
        plan = ResearcherActionPlan.model_validate(first_turns[0]["proposal"])
        task_ids = tuple(
            str(task["task_id"])
            for task in plan.input_payload["research_tasks"]
        )
        ledger = EvidenceLedger(task_ids, max_per_question=50)
        result_limit = int(
            plan.input_payload["budget"]["hits_per_search_ceiling"]
        )
        action_results: list[dict[str, Any]] = []
        collected_evidence: list[ToolEvidence] = []

        for action, validation in zip(
            plan.decision.actions,
            plan.validations,
            strict=True,
        ):
            action_record: dict[str, Any] = {
                "action": action.model_dump(mode="json"),
                "validation": validation.model_dump(mode="json"),
            }
            if (
                not validation.accepted
                or action.tool == "submit_for_validation"
            ):
                action_record["result"] = None
                action_results.append(action_record)
                continue
            request = build_retrieval_tool_request(
                action,
                round_number=1,
                ledger=ledger,
                result_limit=result_limit,
            )
            result = registry.invoke(action.tool, request)
            collected_evidence.extend(result.evidence)
            action_record["request"] = request.model_dump(mode="json")
            action_record["result"] = result.model_dump(mode="json")
            action_results.append(action_record)

        target_coverage: list[dict[str, Any]] = []
        for target in questions[question_id]["evidence_targets"]:
            matches = tuple(
                evidence.evidence_id
                for evidence in collected_evidence
                if _target_matches(target, evidence)
            )
            target_coverage.append({
                "specification": target["specification"],
                "clause": target["clause"],
                "source_path": target["source_path"],
                "line_start": target["line_start"],
                "line_end": target["line_end"],
                "matched_evidence_ids": matches,
                "hit": bool(matches),
            })

        targets_hit = sum(item["hit"] for item in target_coverage)
        question_result = {
            "schema_version": "specrag.researcher-first-turn-evaluation/v1",
            "question_id": question_id,
            "planning_artifact": artifact_path.relative_to(REPO_ROOT).as_posix(),
            "planning_artifact_sha256": hashlib.sha256(
                raw_artifact
            ).hexdigest().upper(),
            "prompt_version": artifact["researcher"]["prompt_version"],
            "execution_scope": retriever.corpus_scope.model_dump(mode="json"),
            "second_round_model_called": False,
            "action_results": action_results,
            "unique_source_chunks": len({
                evidence.chunk_id for evidence in collected_evidence
            }),
            "target_coverage": target_coverage,
            "targets_hit": targets_hit,
            "target_count": len(target_coverage),
            "all_targets_hit": targets_hit == len(target_coverage),
        }
        output_path = output_directory / f"{question_id}.json"
        output_path.write_text(
            json.dumps(question_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        question_results.append(question_result)

    summary = {
        "schema_version": (
            "specrag.researcher-first-turn-evaluation-summary/v1"
        ),
        "planning_directory": planning_directory.relative_to(REPO_ROOT).as_posix(),
        "prompt_version": question_results[0]["prompt_version"],
        "question_count": len(question_results),
        "action_count": sum(
            len(item["action_results"]) for item in question_results
        ),
        "questions_with_all_targets_hit": sum(
            item["all_targets_hit"] for item in question_results
        ),
        "targets_hit": sum(item["targets_hit"] for item in question_results),
        "target_count": sum(item["target_count"] for item in question_results),
        "second_round_model_called": False,
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["run_first_turn_evaluation"]

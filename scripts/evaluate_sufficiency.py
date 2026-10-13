"""Re-evaluate one task's frozen source-valid evidence set."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

from specrag.config import load_config
from specrag.infrastructure import StructuredTaskSufficiencyAssessor
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.models import CandidateEvidence, ResearchTask
from specrag.prompts import SUFFICIENCY_PROMPT_VERSION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_artifact", type=Path)
    parser.add_argument("task_id")
    parser.add_argument("output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = json.loads(args.run_artifact.read_text(encoding="utf-8"))
    state = artifact["state"]
    tasks = {
        item["task_id"]: ResearchTask.model_validate(item)
        for item in state["research_tasks"]
    }
    if args.task_id not in tasks:
        raise ValueError(f"unknown task ID: {args.task_id}")
    source_valid_ids = {
        item["evidence_id"]
        for item in state["source_validations"]
        if item["valid"]
    }
    evidence = tuple(
        CandidateEvidence.model_validate(item)
        for item in state["candidate_evidence"]
        if item["task_id"] == args.task_id
        and item["evidence_id"] in source_valid_ids
    )
    if not evidence:
        raise ValueError("task has no source-valid evidence to assess")
    agent = LLMApiStructuredAgent(
        role="summary",
        reasoning_effort="high",
        config=load_config(),
    )
    assessment = StructuredTaskSufficiencyAssessor(agent).assess(
        tasks[args.task_id],
        evidence,
    )
    result = {
        "schema_version": "specrag.sufficiency-evaluation/v1",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "source_artifact": args.run_artifact.as_posix(),
        "prompt_version": SUFFICIENCY_PROMPT_VERSION,
        "task": tasks[args.task_id].model_dump(mode="json"),
        "evidence_ids": [item.evidence_id for item in evidence],
        "assessment": assessment.model_dump(mode="json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["assessment"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate final answers from completed Researcher/Judge artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

from specrag.config import load_config
from specrag.evidence import EvidenceSynthesisService
from specrag.infrastructure.research_assessors import (
    StructuredEvidenceSynthesisGenerator,
)
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.models import (
    CandidateEvidence,
    ResearchTask,
    TaskSufficiencyAssessment,
    ValidatedEvidence,
)
from specrag.prompts import SYNTHESIS_PROMPT_VERSION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--answer-language", default="English")
    parser.add_argument("--detail-level", default="standard")
    return parser


def _synthesize(path: Path, output: Path, args: argparse.Namespace) -> dict[str, object]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    state = artifact["state"]
    question = str(state["user_question"])
    if state.get("evidence_sufficient") is not True:
        result = {
            "schema_version": "specrag.synthesis-evaluation/v1",
            "source_artifact": path.as_posix(),
            "question_id": state.get("research_id"),
            "status": "skipped_insufficient_evidence",
            "synthesis_prompt_version": SYNTHESIS_PROMPT_VERSION,
        }
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    tasks = tuple(ResearchTask.model_validate(item) for item in state["research_tasks"])
    candidates = tuple(CandidateEvidence.model_validate(item) for item in state["candidate_evidence"])
    validated = tuple(ValidatedEvidence.model_validate(item) for item in state["validated_evidence"])
    assessments = tuple(
        TaskSufficiencyAssessment.model_validate(item)
        for item in state["sufficiency_assessments"]
    )
    agent = LLMApiStructuredAgent(
        role="summary",
        reasoning_effort="high",
        config=load_config(),
    )
    generator = StructuredEvidenceSynthesisGenerator(
        agent,
        answer_language=args.answer_language,
        detail_level=args.detail_level,
    )
    synthesis = EvidenceSynthesisService(generator).synthesize(
        question,
        tasks,
        candidates,
        validated,
        assessments,
    )
    result = {
        "schema_version": "specrag.synthesis-evaluation/v1",
        "source_artifact": path.as_posix(),
        "question_id": state.get("research_id"),
        "status": "completed",
        "synthesis_prompt_version": SYNTHESIS_PROMPT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "question": question,
        "synthesis": synthesis.model_dump(mode="json"),
        "input_evidence_ids": [
            item.evidence_id
            for item in validated
            if item.validation_status == "supported" and item.source_valid
        ],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for path in sorted(args.input_directory.glob("r17-qa-*.json")):
        output = args.output_directory / path.name
        if args.resume and output.is_file():
            summaries.append({"question_id": path.stem, "status": "resumed"})
            continue
        try:
            result = _synthesize(path, output, args)
        except Exception as exc:  # noqa: BLE001 - persist bounded evaluation failure
            result = {
                "schema_version": "specrag.synthesis-evaluation/v1",
                "source_artifact": path.as_posix(),
                "question_id": path.stem,
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:1600]}",
            }
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(result)
        print(json.dumps({key: result.get(key) for key in ("question_id", "status", "error")}, ensure_ascii=False))

    summary = {
        "schema_version": "specrag.synthesis-evaluation-summary/v1",
        "synthesis_prompt_version": SYNTHESIS_PROMPT_VERSION,
        "input_directory": args.input_directory.as_posix(),
        "output_directory": args.output_directory.as_posix(),
        "results": summaries,
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 1 if any(item.get("status") == "error" for item in summaries) else 0


if __name__ == "__main__":
    sys.exit(main())

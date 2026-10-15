"""Evaluate generated synthesis answers against offline QA annotations."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

from specrag.config import load_config
from specrag.evaluation import SynthesisQualityAssessment
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.prompts import (
    SYNTHESIS_QUALITY_INSTRUCTIONS,
    SYNTHESIS_QUALITY_PROMPT_VERSION,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("synthesis_directory", type=Path)
    parser.add_argument("qa_file", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def _qa_records(path: Path) -> dict[str, dict[str, object]]:
    return {
        item["id"]: item
        for item in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _evaluate(path: Path, qa: dict[str, object], output: Path) -> dict[str, object]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("status") != "completed":
        result = {
            "schema_version": "specrag.synthesis-quality/v1",
            "question_id": path.stem,
            "status": "skipped",
            "reason": artifact.get("status", "not_completed"),
            "prompt_version": SYNTHESIS_QUALITY_PROMPT_VERSION,
        }
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    source_path = artifact.get("source_artifact") or artifact.get(
        "researcher_source_artifact"
    )
    if source_path is None:
        raise ValueError("synthesis artifact has no source Researcher artifact")
    source_artifact = json.loads(Path(source_path).read_text(encoding="utf-8"))
    state = source_artifact["state"]
    selected_ids = set(artifact.get("input_evidence_ids", ()))
    if not selected_ids:
        selected_ids = {
            item["evidence_id"]
            for item in artifact.get("validated_evidence", ())
            if item.get("validation_status") == "supported"
        }
    evidence = [
        item
        for item in state["candidate_evidence"]
        if item["evidence_id"] in selected_ids
    ]
    synthesis = artifact["synthesis"]
    agent = LLMApiStructuredAgent(
        role="summary",
        reasoning_effort="high",
        config=load_config(),
    )
    response = agent.parse(
        purpose="evaluate generated synthesis answer quality",
        prompt_version=SYNTHESIS_QUALITY_PROMPT_VERSION,
        instructions=SYNTHESIS_QUALITY_INSTRUCTIONS,
        input_payload={
            "question_id": qa["id"],
            "original_question": qa["question"],
            "expected_answer": qa["expected_answer"],
            "expected_facts": qa["expected_facts"],
            "generated_answer": synthesis["answer"],
            "generated_claims": synthesis["claims"],
            "judge_supported_evidence": evidence,
        },
        output_type=SynthesisQualityAssessment,
    )
    assessment = response.output.model_copy(update={"question_id": qa["id"]})
    result = {
        "schema_version": "specrag.synthesis-quality/v1",
        "question_id": qa["id"],
        "status": "completed",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "prompt_version": SYNTHESIS_QUALITY_PROMPT_VERSION,
        "assessment": assessment.model_dump(mode="json"),
        "overall": assessment.overall,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    qa = _qa_records(args.qa_file)
    results: list[dict[str, object]] = []
    for path in sorted(args.synthesis_directory.glob("r17-qa-*.json")):
        output = args.output_directory / path.name
        if args.resume and output.is_file():
            result = {"question_id": path.stem, "status": "resumed"}
        else:
            try:
                result = _evaluate(path, qa[path.stem], output)
            except Exception as exc:  # noqa: BLE001 - persist bounded evaluation failure
                result = {
                    "schema_version": "specrag.synthesis-quality/v1",
                    "question_id": path.stem,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {str(exc)[:1600]}",
                }
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        print(json.dumps({key: result.get(key) for key in ("question_id", "status", "overall", "error")}, ensure_ascii=False))

    summary = {
        "schema_version": "specrag.synthesis-quality-summary/v1",
        "prompt_version": SYNTHESIS_QUALITY_PROMPT_VERSION,
        "results": results,
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 1 if any(item.get("status") == "error" for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())

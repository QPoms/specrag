"""Re-run Judge and Synthesizer on frozen Researcher evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from specrag.config import REPO_ROOT, load_config
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
)
from specrag.prompts import (
    SUFFICIENCY_PROMPT_VERSION,
    SYNTHESIS_PROMPT_VERSION,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--id", dest="identifiers", action="append", required=True)
    return parser


def _run(path: Path, output: Path) -> dict[str, object]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    state = artifact["state"]
    tasks = tuple(ResearchTask.model_validate(item) for item in state["research_tasks"])
    candidates = tuple(CandidateEvidence.model_validate(item) for item in state["candidate_evidence"])
    source_validation = SourceValidationService(
        SourceEvidenceValidator(REPO_ROOT)
    ).validate(tasks, candidates)
    agent = LLMApiStructuredAgent(
        role="summary",
        reasoning_effort="high",
        config=load_config(),
    )
    sufficiency = EvidenceSufficiencyService(
        StructuredTaskSufficiencyAssessor(agent)
    ).evaluate(
        tasks,
        candidates,
        source_validation.source_validations,
        original_question=state["user_question"],
    )
    result: dict[str, object] = {
        "schema_version": "specrag.regression-pipeline/v1",
        "question_id": state["research_id"],
        "researcher_source_artifact": path.as_posix(),
        "sufficiency_prompt_version": SUFFICIENCY_PROMPT_VERSION,
        "synthesis_prompt_version": SYNTHESIS_PROMPT_VERSION,
        "research_tasks": [item.model_dump(mode="json") for item in sufficiency.tasks],
        "source_validations": [
            item.model_dump(mode="json")
            for item in source_validation.source_validations
        ],
        "sufficiency_assessments": [
            item.model_dump(mode="json") for item in sufficiency.assessments
        ],
        "validated_evidence": [
            item.model_dump(mode="json") for item in sufficiency.validated_evidence
        ],
        "evidence_sufficient": sufficiency.evidence_sufficient,
        "gaps": [item.model_dump(mode="json") for item in sufficiency.gaps],
    }
    if sufficiency.evidence_sufficient:
        synthesis = EvidenceSynthesisService(
            StructuredEvidenceSynthesisGenerator(agent)
        ).synthesize(
            state["user_question"],
            sufficiency.tasks,
            candidates,
            sufficiency.validated_evidence,
            sufficiency.assessments,
        )
        result["synthesis"] = synthesis.model_dump(mode="json")
        result["status"] = "completed"
    else:
        result["status"] = "insufficient"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    results = []
    for identifier in args.identifiers:
        path = args.artifact_root / f"{identifier}.json"
        output = args.output_directory / path.name
        try:
            result = _run(path, output)
        except Exception as exc:  # noqa: BLE001 - persist bounded regression failure
            result = {
                "schema_version": "specrag.regression-pipeline/v1",
                "question_id": identifier,
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:1600]}",
            }
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        print(json.dumps({
            "question_id": result.get("question_id"),
            "status": result.get("status"),
            "evidence_sufficient": result.get("evidence_sufficient"),
        }, ensure_ascii=False))
    (args.output_directory / "summary.json").write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 1 if any(item.get("status") == "error" for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())

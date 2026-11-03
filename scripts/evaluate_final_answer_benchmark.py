"""Judge saved Workflow answers against the frozen final-answer Benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid
from typing import Any, Mapping

from specrag.config import AppConfig, load_config
from specrag.evaluation import (
    FinalAnswerJudgeExecution,
    FinalAnswerJudgeInput,
    FinalAnswerJudgmentDraft,
    JudgeCallResult,
    execute_final_answer_judge,
    load_end_to_end_benchmark,
)
from specrag.evaluation.benchmark_models import BenchmarkCase, Difficulty
from specrag.infrastructure.structured_agent import (
    CodexStructuredAgent,
    LLMApiStructuredAgent,
)
from specrag.prompts import (
    FINAL_ANSWER_JUDGE_INSTRUCTIONS,
    FINAL_ANSWER_JUDGE_PROMPT_VERSION,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score saved Workflow final answers with the frozen Rubric Judge."
        )
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--difficulty",
        choices=("all", "simple", "complex"),
        default="all",
    )
    parser.add_argument("--id", dest="identifiers", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--backend",
        choices=("api", "codex"),
        default="api",
    )
    return parser


class _StructuredJudge:
    def __init__(
        self,
        agent: LLMApiStructuredAgent | CodexStructuredAgent,
    ) -> None:
        self._agent = agent

    def __call__(
        self,
        judge_input: FinalAnswerJudgeInput,
        *,
        correction: str | None = None,
    ) -> JudgeCallResult:
        instructions = FINAL_ANSWER_JUDGE_INSTRUCTIONS
        if correction is not None:
            instructions = f"{instructions.rstrip()}\n\n{correction}\n"
        response = self._agent.parse(
            purpose="judge final-answer Rubric coverage",
            prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
            instructions=instructions,
            input_payload=judge_input,
            output_type=FinalAnswerJudgmentDraft,
        )
        return JudgeCallResult(
            judgment=response.output,
            judge_model=response.receipt.model,
        )


def _single_request_config() -> AppConfig:
    """Disable adapter retries so the execution policy owns both attempts."""

    base = load_config()
    llm = base.llm.model_copy(update={
        "api_max_retries": 0,
        "api_fallback_models": (),
    })
    return base.model_copy(update={"llm": llm})


def _selected_cases(
    *,
    difficulty: str,
    identifiers: list[str] | None,
    limit: int | None,
) -> tuple[BenchmarkCase, ...]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    benchmark = load_end_to_end_benchmark()
    requested = set(identifiers or ())
    known = {case.task_metadata.task_id for case in benchmark.cases}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown Benchmark question IDs: {sorted(unknown)}")
    cases = tuple(
        case
        for case in benchmark.cases
        if (difficulty == "all" or case.task_metadata.difficulty == difficulty)
        and (not requested or case.task_metadata.task_id in requested)
    )
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("no Benchmark cases matched the selection")
    return cases


def _artifact_payload(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Run artifact must contain an object: {path}")
    return value


def _artifact_state(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    state = artifact.get("state", artifact)
    if not isinstance(state, Mapping):
        raise ValueError("Run artifact state must contain an object")
    return state


def _candidate_answer(artifact: Mapping[str, Any]) -> str | None:
    state = _artifact_state(artifact)
    final_answer = state.get("final_answer")
    if isinstance(final_answer, str):
        return final_answer
    synthesis = state.get("synthesis")
    if isinstance(synthesis, str):
        return synthesis
    if isinstance(synthesis, Mapping) and isinstance(synthesis.get("answer"), str):
        return str(synthesis["answer"])
    return None


def _run_id(artifact: Mapping[str, Any], *, question_id: str) -> str:
    state = _artifact_state(artifact)
    for value in (
        artifact.get("run_id"),
        state.get("run_id"),
        artifact.get("thread_id"),
    ):
        if isinstance(value, str) and value.strip():
            return value
    return f"saved-artifact:{question_id}"


def _validate_artifact_case(
    artifact: Mapping[str, Any],
    case: BenchmarkCase,
) -> None:
    question = artifact.get("question")
    if not isinstance(question, Mapping):
        raise ValueError("Run artifact is missing its question object")
    if question.get("id") != case.task_metadata.task_id:
        raise ValueError("Run artifact question ID does not match Benchmark case")
    if question.get("text") != case.question:
        raise ValueError("Run artifact question text does not match Benchmark case")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    *,
    args: argparse.Namespace,
    cases: tuple[BenchmarkCase, ...],
    model: str,
) -> dict[str, object]:
    artifacts = []
    for case in cases:
        question_id = case.task_metadata.task_id
        path = args.run_directory / f"{question_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing saved Run artifact: {path}")
        artifacts.append({
            "question_id": question_id,
            "difficulty": case.task_metadata.difficulty,
            "path": path.resolve().as_posix(),
            "sha256": _sha256(path),
        })
    benchmark = load_end_to_end_benchmark()
    return {
        "schema_version": "specrag.final-answer-judge-batch/v1",
        "benchmark_id": benchmark.manifest.benchmark_id,
        "benchmark_version": benchmark.manifest.benchmark_version,
        "judge_prompt_version": FINAL_ANSWER_JUDGE_PROMPT_VERSION,
        "backend": args.backend,
        "configured_model": model,
        "difficulty": args.difficulty,
        "question_ids": [case.task_metadata.task_id for case in cases],
        "source_artifacts": artifacts,
    }


def _freeze_manifest(path: Path, expected: Mapping[str, object]) -> None:
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("existing Judge batch manifest does not match this run")
        return
    _write_json(path, expected)


def _summary(
    *,
    cases: tuple[BenchmarkCase, ...],
    executions: tuple[FinalAnswerJudgeExecution, ...],
) -> dict[str, object]:
    difficulty_by_id: dict[str, Difficulty] = {
        case.task_metadata.task_id: case.task_metadata.difficulty for case in cases
    }
    failed = [item for item in executions if item.status == "evaluation_failed"]
    scored = [item for item in executions if item.score is not None]
    results = [
        {
            "question_id": item.question_id,
            "difficulty": difficulty_by_id[item.question_id],
            "status": item.status,
            "score": item.score,
            "attempt_count": item.attempt_count,
        }
        for item in executions
    ]
    average = None
    if not failed and scored:
        average = round(sum(float(item.score) for item in scored) / len(scored), 1)
    score_by_difficulty: dict[str, object] = {}
    for difficulty in ("simple", "complex"):
        selected = [
            item
            for item in executions
            if difficulty_by_id[item.question_id] == difficulty
        ]
        if not selected:
            continue
        selected_failed = [
            item for item in selected if item.status == "evaluation_failed"
        ]
        selected_scored = [item for item in selected if item.score is not None]
        selected_average = None
        if not selected_failed and selected_scored:
            selected_average = round(
                sum(float(item.score) for item in selected_scored)
                / len(selected_scored),
                1,
            )
        score_by_difficulty[difficulty] = {
            "requested_count": len(selected),
            "score_count": len(selected_scored),
            "evaluation_failed_count": len(selected_failed),
            "average_score": selected_average,
        }
    return {
        "schema_version": "specrag.final-answer-judge-summary/v1",
        "requested_count": len(cases),
        "score_count": len(scored),
        "no_answer_count": sum(item.status == "no_answer" for item in executions),
        "evaluation_failed_count": len(failed),
        "average_score": average,
        "score_by_difficulty": score_by_difficulty,
        "results": results,
    }


def _configured_model(config: AppConfig) -> str:
    return (
        config.secrets.openai_summary_model.strip()
        or config.secrets.openai_model.strip()
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = _selected_cases(
        difficulty=args.difficulty,
        identifiers=args.identifiers,
        limit=args.limit,
    )
    config: AppConfig | None = None
    if args.backend == "api":
        config = _single_request_config()
        model = _configured_model(config)
        if not model:
            raise ValueError("no summary or default model is configured")
    else:
        model = "codex_managed_default"
    args.output_directory.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args=args, cases=cases, model=model)
    _freeze_manifest(args.output_directory / "manifest.json", manifest)

    agent: LLMApiStructuredAgent | CodexStructuredAgent | None = None
    executions: list[FinalAnswerJudgeExecution] = []
    for case in cases:
        question_id = case.task_metadata.task_id
        output_path = args.output_directory / f"{question_id}.json"
        if output_path.exists():
            if not args.resume:
                raise FileExistsError(
                    f"Judge output already exists; pass --resume: {output_path}"
                )
            execution = FinalAnswerJudgeExecution.model_validate_json(
                output_path.read_text(encoding="utf-8")
            )
            executions.append(execution)
            print(json.dumps({
                "question_id": question_id,
                "status": "resumed",
                "score": execution.score,
            }, ensure_ascii=False), flush=True)
            continue

        artifact_path = args.run_directory / f"{question_id}.json"
        artifact = _artifact_payload(artifact_path)
        _validate_artifact_case(artifact, case)
        candidate_answer = _candidate_answer(artifact)
        if candidate_answer is not None and candidate_answer.strip() and agent is None:
            if args.backend == "api":
                if config is None:
                    raise RuntimeError("API Judge configuration was not initialized")
                agent = LLMApiStructuredAgent(
                    role="summary",
                    reasoning_effort="high",
                    timeout_seconds=args.timeout_seconds,
                    config=config,
                )
            else:
                agent = CodexStructuredAgent(
                    work_root=args.output_directory / ".codex-judge-work",
                    reasoning_effort="high",
                    timeout_seconds=args.timeout_seconds,
                    sandbox="read-only",
                )
        judge = _StructuredJudge(agent) if agent is not None else _NoCallJudge()
        execution = execute_final_answer_judge(
            case=case,
            run_id=_run_id(artifact, question_id=question_id),
            candidate_answer=candidate_answer,
            judge=judge,
            judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
        )
        _write_json(output_path, execution.model_dump(mode="json"))
        executions.append(execution)
        print(json.dumps({
            "question_id": question_id,
            "status": execution.status,
            "score": execution.score,
            "attempt_count": execution.attempt_count,
        }, ensure_ascii=False), flush=True)

    summary = _summary(cases=cases, executions=tuple(executions))
    _write_json(args.output_directory / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if summary["evaluation_failed_count"] else 0


class _NoCallJudge:
    def __call__(
        self,
        judge_input: FinalAnswerJudgeInput,
        *,
        correction: str | None = None,
    ) -> JudgeCallResult:
        raise AssertionError("blank Candidate Answer must bypass the Judge")


if __name__ == "__main__":
    sys.exit(main())

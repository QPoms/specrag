"""Run the frozen one-trial multi-model final-answer comparison."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

try:
    from scripts import evaluate_final_answer_benchmark as final_answer_runner
except ImportError:  # direct ``python scripts/...`` execution
    import evaluate_final_answer_benchmark as final_answer_runner
from specrag.config import AppConfig, REPO_ROOT, load_config
from specrag.evaluation import (
    ComparisonCase,
    FinalAnswerJudgeExecution,
    ModelCaseResult,
    ModelComparisonExperimentManifest,
    ModelComparisonInput,
    ModelProbeResult,
    ModelTarget,
    load_end_to_end_benchmark,
    summarize_model_comparison,
)
from specrag.infrastructure.model_capabilities import (
    ResearchModelCapability,
    probe_research_model_capability,
)
from specrag.infrastructure.structured_agent import (
    LLMApiStructuredAgent,
    capture_agent_receipts,
)
from specrag.interfaces.model_comparison import (
    benchmark_questions,
    candidate_model_config,
    run_model_cell,
    validate_candidate_receipts,
)
from specrag.models import ResearchBudget
from specrag.prompts import FINAL_ANSWER_JUDGE_PROMPT_VERSION


REQUIRED_MODELS = (
    "gpt-5.6-sol",
    "gpt-6-astra",
    "deepseek-v4.1-flash",
)
OPTIONAL_MODELS = ("claude-opus-5",)
JUDGE_MODEL = "deepseek-v4-pro"


class _JudgeProbeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["ready"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe, run, judge, and summarize the model comparison."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "run", "judge", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("experiment_directory", type=Path)
        if name in {"run", "judge"}:
            command.add_argument("--resume", action="store_true")
        if name in {"probe", "run", "judge"}:
            command.add_argument("--timeout-seconds", type=int, default=600)
    probe = commands.choices["probe"]
    probe.add_argument("--experiment-id")
    probe.add_argument("--model", dest="required_models", action="append")
    probe.add_argument("--optional-model", dest="optional_models", action="append")
    probe.add_argument("--judge-model", default=JUDGE_MODEL)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _freeze_json(path: Path, expected: Mapping[str, Any]) -> None:
    if path.is_file():
        if _load_object(path) != dict(expected):
            raise ValueError(f"frozen experiment input changed: {path}")
        return
    _write_json(path, expected)


def _source_revision() -> dict[str, object]:
    prefix = ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}"]
    try:
        commit = subprocess.run(
            [*prefix, "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            [*prefix, "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unavailable", "dirty": None}
    return {"git_commit": commit, "dirty": dirty}


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_experiment_manifest(
    *,
    experiment_id: str,
    required_models: Sequence[str] = REQUIRED_MODELS,
    optional_models: Sequence[str] = OPTIONAL_MODELS,
    judge_model: str = JUDGE_MODEL,
    budget: ResearchBudget | None = None,
    config: AppConfig | None = None,
) -> dict[str, object]:
    """Build the immutable experiment input without evaluator-only case data."""

    benchmark = load_end_to_end_benchmark()
    required = tuple(_normalized_models(required_models, label="required"))
    optional = tuple(_normalized_models(optional_models, label="optional"))
    if set(required) & set(optional):
        raise ValueError("required and optional model sets must not overlap")
    slugs = tuple(_model_slug(model) for model in (*required, *optional))
    if len(slugs) != len(set(slugs)):
        raise ValueError("model IDs must map to unique artifact directory names")
    normalized_judge = judge_model.strip()
    if not normalized_judge:
        raise ValueError("Judge model must not be blank")
    normalized_experiment = experiment_id.strip()
    if not normalized_experiment:
        raise ValueError("experiment ID must not be blank")
    active_budget = budget or ResearchBudget(
        max_iterations=3,
        max_tool_calls=6,
        max_gap_tool_calls=6,
        max_replans=2,
    )
    settings = config or load_config()
    comparison = ModelComparisonExperimentManifest(
        experiment_id=normalized_experiment,
        benchmark_id=benchmark.manifest.benchmark_id,
        benchmark_version=benchmark.manifest.benchmark_version,
        judge_model=normalized_judge,
        judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
        cases=tuple(
            ComparisonCase(
                question_id=case.task_metadata.task_id,
                difficulty=case.task_metadata.difficulty,
            )
            for case in benchmark.cases
        ),
        model_targets=tuple(
            ModelTarget(model_id=model, required=True) for model in required
        ) + tuple(
            ModelTarget(model_id=model, required=False) for model in optional
        ),
    )
    benchmark_projection = {
        "benchmark_id": benchmark.manifest.benchmark_id,
        "benchmark_version": benchmark.manifest.benchmark_version,
        "question_ids": [case.task_metadata.task_id for case in benchmark.cases],
        "sources": [item.model_dump(mode="json") for item in benchmark.manifest.sources],
        "environment": benchmark.manifest.environment.model_dump(mode="json"),
    }
    return {
        "schema_version": "specrag.model-comparison-operational-manifest/v1",
        "comparison": comparison.model_dump(mode="json"),
        "benchmark": benchmark_projection,
        "benchmark_sha256": _sha256_json(benchmark_projection),
        "execution": {
            "trial_count_per_model": 1,
            "question_order": "benchmark_order",
            "execution_order": "sequential_models_sequential_questions",
            "cache_condition": "warm_cache",
            "fallback_models": [],
            "budget": active_budget.model_dump(mode="json"),
        },
        "source_revision": _source_revision(),
        "langsmith": {
            "enabled": bool(
                settings.secrets.langsmith_tracing
                and settings.secrets.langsmith_api_key.strip()
            ),
            "project": settings.secrets.langsmith_project.strip() or "specrag",
        },
    }


def _normalized_models(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    normalized = tuple(value.strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{label} model IDs must be non-blank")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} model IDs must be unique")
    return normalized


def _comparison_manifest(value: Mapping[str, Any]) -> ModelComparisonExperimentManifest:
    return ModelComparisonExperimentManifest.model_validate(value["comparison"])


def _model_slug(model_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id).strip("-.")
    if not slug or slug in {".", ".."}:
        raise ValueError(f"model ID cannot form an artifact directory: {model_id!r}")
    return slug


def _probe_path(directory: Path, model_id: str) -> Path:
    return directory / "probes" / f"{_model_slug(model_id)}.json"


def _judge_probe_path(directory: Path) -> Path:
    return directory / "probes" / "judge.json"


def _agent_factory(model_id: str, timeout_seconds: int) -> LLMApiStructuredAgent:
    config = candidate_model_config(model_id)
    config = config.model_copy(update={
        "llm": config.llm.model_copy(update={
            "api_max_retries": 0,
            "api_fallback_models": (),
        })
    })
    return LLMApiStructuredAgent(
        role="summary",
        reasoning_effort="high",
        timeout_seconds=timeout_seconds,
        config=config,
    )


def probe_experiment(
    experiment_directory: Path,
    manifest: Mapping[str, Any],
    *,
    timeout_seconds: int,
    agent_factory: Callable[[str, int], Any] = _agent_factory,
) -> int:
    """Probe every candidate and the fixed Judge, persisting secret-free results."""

    comparison = _comparison_manifest(manifest)
    experiment_directory.mkdir(parents=True, exist_ok=True)
    _freeze_json(experiment_directory / "manifest.json", manifest)
    failures = False
    for target in comparison.model_targets:
        try:
            agent = agent_factory(target.model_id, timeout_seconds)
            with capture_agent_receipts() as receipts:
                capability = probe_research_model_capability(agent)
            payloads = [item.model_dump(mode="json") for item in receipts]
        except Exception as exc:  # noqa: BLE001 - safe probe projection below
            result = ModelProbeResult(
                model_id=target.model_id,
                status="unavailable",
                failure_kind=str(
                    getattr(exc, "failure_kind", "configuration")
                )[:200],
                message=(
                    "The required model probe could not be initialized."
                    if target.required
                    else (
                        "The optional model probe could not be initialized "
                        "and was skipped."
                    )
                ),
            )
            _freeze_json(
                _probe_path(experiment_directory, target.model_id),
                result.model_dump(mode="json"),
            )
            failures = failures or target.required
            continue
        status: Literal["ready", "incompatible", "unavailable"]
        observed: str | None = None
        failure_kind: str | None = None
        if capability.status == "ready":
            try:
                if not payloads:
                    raise ValueError("probe produced no model receipts")
                validate_candidate_receipts(payloads, target.model_id)
            except ValueError:
                status = "incompatible"
                failure_kind = "model_contaminated"
            else:
                status = "ready"
                observed = target.model_id
        else:
            failure_kind = _capability_failure_kind(capability)
            status = (
                "incompatible"
                if failure_kind in {"invalid_request", "structured_output", "unknown"}
                else "unavailable"
            )
        result = ModelProbeResult(
            model_id=target.model_id,
            status=status,
            observed_model=observed,
            failure_kind=failure_kind,
            message=(
                "The model passed the structured-output and native-tool replay probes."
                if status == "ready"
                else (
                    "The required model is incompatible with the research contract."
                    if target.required
                    else "The optional model is unavailable or incompatible and was skipped."
                )
            ),
        )
        _freeze_json(_probe_path(experiment_directory, target.model_id), result.model_dump(mode="json"))
        failures = failures or (target.required and result.status != "ready")

    try:
        judge_agent = agent_factory(comparison.judge_model, timeout_seconds)
        judge_payload = _probe_judge(judge_agent, comparison.judge_model)
    except Exception as exc:  # noqa: BLE001 - safe probe projection below
        judge_payload = {
            "schema_version": "specrag.judge-model-probe/v1",
            "model_id": comparison.judge_model,
            "status": "unavailable",
            "failure_kind": str(
                getattr(exc, "failure_kind", "configuration")
            )[:200],
            "message": "The fixed Judge probe could not be initialized.",
        }
    _freeze_json(_judge_probe_path(experiment_directory), judge_payload)
    failures = failures or judge_payload["status"] != "ready"
    return 1 if failures else 0


def _capability_failure_kind(capability: ResearchModelCapability) -> str:
    for check in (capability.structured_output, capability.native_tool_replay):
        if check.status != "ready" and check.failure_kind:
            return check.failure_kind
    return "unknown"


def _probe_judge(agent: Any, model_id: str) -> dict[str, object]:
    try:
        with capture_agent_receipts() as receipts:
            response = agent.parse(
                purpose="final_answer_judge_capability_probe",
                prompt_version="final-answer-judge-capability/v1",
                instructions="Return the strict object with status set to ready.",
                input_payload={"probe": "structured_output"},
                output_type=_JudgeProbeOutput,
            )
        payloads = [item.model_dump(mode="json") for item in receipts]
        if not payloads:
            raise ValueError("Judge probe produced no model receipts")
        validate_candidate_receipts(payloads, model_id)
        observed = response.receipt.model
        if observed != model_id:
            raise ValueError("Judge probe used a different model")
    except Exception as exc:  # noqa: BLE001 - persist only safe taxonomy
        kind = str(getattr(exc, "failure_kind", "unknown"))[:200]
        return {
            "schema_version": "specrag.judge-model-probe/v1",
            "model_id": model_id,
            "status": "unavailable",
            "failure_kind": kind,
            "message": "The fixed Judge failed its strict structured-output probe.",
        }
    return {
        "schema_version": "specrag.judge-model-probe/v1",
        "model_id": model_id,
        "status": "ready",
        "observed_model": observed,
        "message": "The fixed Judge passed its strict structured-output probe.",
    }


def _load_probes(
    directory: Path,
    comparison: ModelComparisonExperimentManifest,
) -> tuple[ModelProbeResult, ...]:
    return tuple(
        ModelProbeResult.model_validate(_load_object(_probe_path(directory, target.model_id)))
        for target in comparison.model_targets
    )


def _ready_targets(
    comparison: ModelComparisonExperimentManifest,
    probes: Sequence[ModelProbeResult],
) -> tuple[ModelTarget, ...]:
    by_id = {item.model_id: item for item in probes}
    blocked = [
        target.model_id for target in comparison.model_targets
        if target.required and by_id[target.model_id].status != "ready"
    ]
    if blocked:
        raise ValueError("required model probes are not ready: " + ", ".join(blocked))
    return tuple(
        target for target in comparison.model_targets
        if by_id[target.model_id].status == "ready"
    )


def run_experiment(directory: Path, *, resume: bool) -> int:
    manifest_path = directory / "manifest.json"
    manifest = _load_object(manifest_path)
    comparison = _comparison_manifest(manifest)
    probes = _load_probes(directory, comparison)
    ready = _ready_targets(comparison, probes)
    benchmark = load_end_to_end_benchmark()
    questions = benchmark_questions(
        benchmark, tuple(case.question_id for case in comparison.cases)
    )
    budget = ResearchBudget.model_validate(manifest["execution"]["budget"])
    result = 0
    for target in ready:
        result = max(result, run_model_cell(
            experiment_id=comparison.experiment_id,
            source_manifest_path=manifest_path,
            requested_model=target.model_id,
            questions=questions,
            output_directory=directory / "runs" / _model_slug(target.model_id),
            budget=budget,
            trial=1,
            release=benchmark.manifest.environment.release,
            resume=resume,
            benchmark=benchmark,
        ))
    return result


def judge_experiment(directory: Path, *, resume: bool, timeout_seconds: int) -> int:
    manifest = _load_object(directory / "manifest.json")
    comparison = _comparison_manifest(manifest)
    probes = _load_probes(directory, comparison)
    ready = _ready_targets(comparison, probes)
    judge_probe = _load_object(_judge_probe_path(directory))
    if (
        judge_probe.get("status") != "ready"
        or judge_probe.get("model_id") != comparison.judge_model
    ):
        raise ValueError("fixed Judge probe is not ready")
    result = 0
    for target in ready:
        run_directory = directory / "runs" / _model_slug(target.model_id)
        completed_ids = tuple(
            case.question_id
            for case in comparison.cases
            if _load_object(run_directory / f"{case.question_id}.json").get(
                "workflow_status"
            ) == "completed"
        )
        if not completed_ids:
            continue
        argv = [
            str(run_directory),
            str(directory / "judgments" / _model_slug(target.model_id)),
            "--backend", "api",
            "--judge-model", comparison.judge_model,
            "--trace-experiment-id", comparison.experiment_id,
            "--trace-candidate-model", target.model_id,
            "--trace-trial", "1",
            "--timeout-seconds", str(timeout_seconds),
        ]
        for question_id in completed_ids:
            argv.extend(("--id", question_id))
        if resume:
            argv.append("--resume")
        result = max(result, final_answer_runner.main(argv))
    return result


def summarize_experiment(directory: Path) -> dict[str, object]:
    manifest = _load_object(directory / "manifest.json")
    comparison = _comparison_manifest(manifest)
    probes = _load_probes(directory, comparison)
    ready = _ready_targets(comparison, probes)
    results: list[ModelCaseResult] = []
    for target in ready:
        slug = _model_slug(target.model_id)
        for case in comparison.cases:
            run = _load_object(directory / "runs" / slug / f"{case.question_id}.json")
            if run.get("workflow_status") == "failed":
                results.append(ModelCaseResult(
                    model_id=target.model_id,
                    question_id=case.question_id,
                    workflow_status="failed",
                    judge_status="not_run",
                    score=None,
                ))
                continue
            judgment_path = directory / "judgments" / slug / f"{case.question_id}.json"
            if not judgment_path.is_file():
                raise ValueError(f"missing Judge result: {judgment_path}")
            execution = FinalAnswerJudgeExecution.model_validate_json(
                judgment_path.read_text(encoding="utf-8")
            )
            results.append(ModelCaseResult(
                model_id=target.model_id,
                question_id=case.question_id,
                workflow_status="completed",
                judge_status=execution.status,
                score=execution.score,
            ))
    summary = summarize_model_comparison(ModelComparisonInput(
        manifest=comparison,
        probes=probes,
        results=tuple(results),
    ))
    payload = summary.model_dump(mode="json")
    _write_json(directory / "comparison.json", payload)
    for model in summary.model_summaries:
        _write_json(
            directory / "model-summaries" / f"{_model_slug(model.model_id)}.json",
            model.model_dump(mode="json"),
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "probe":
            experiment_id = args.experiment_id or (
                "specrag-model-comparison-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            )
            manifest = build_experiment_manifest(
                experiment_id=experiment_id,
                required_models=tuple(args.required_models or REQUIRED_MODELS),
                optional_models=tuple(args.optional_models or OPTIONAL_MODELS),
                judge_model=args.judge_model,
            )
            return probe_experiment(
                args.experiment_directory,
                manifest,
                timeout_seconds=args.timeout_seconds,
            )
        if args.command == "run":
            return run_experiment(args.experiment_directory, resume=args.resume)
        if args.command == "judge":
            return judge_experiment(
                args.experiment_directory,
                resume=args.resume,
                timeout_seconds=args.timeout_seconds,
            )
        payload = summarize_experiment(args.experiment_directory)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"model comparison: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JUDGE_MODEL",
    "OPTIONAL_MODELS",
    "REQUIRED_MODELS",
    "build_experiment_manifest",
    "judge_experiment",
    "main",
    "probe_experiment",
    "run_experiment",
    "summarize_experiment",
]

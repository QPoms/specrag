"""Run the frozen one-trial multi-model final-answer comparison."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
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
from specrag.infrastructure.llm_contracts import LLMApiMode
from specrag.infrastructure.langsmith_tracing import (
    LangSmithPreflightError,
    build_langsmith_tracing,
)
from specrag.infrastructure.model_transport import ModelTransportProfile
from specrag.infrastructure.structured_agent import (
    LLMApiStructuredAgent,
    capture_agent_receipts,
)
from specrag.interfaces.model_comparison import (
    benchmark_questions,
    candidate_model_config,
    run_model_cell,
    validate_candidate_receipts,
    validate_model_cell_artifact,
)
from specrag.models import ResearchBudget
from specrag.prompts import FINAL_ANSWER_JUDGE_PROMPT_VERSION


REQUIRED_MODELS = ("gpt-5.6-sol",)
OPTIONAL_MODELS: tuple[str, ...] = ()
JUDGE_MODEL = "deepseek-v4-pro"


def _require_langsmith_ready(config: AppConfig | None = None) -> None:
    """Block every paid formal phase until trace write/read is verified."""

    result = build_langsmith_tracing(config or load_config()).require_ready()
    if result.status != "ready":
        raise ValueError(
            "LangSmith tracing must be enabled and pass its write/read preflight"
        )


class _JudgeProbeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["ready"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe, run, judge, and summarize the model comparison."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "langsmith-check",
        help="verify LangSmith project access and a synthetic trace write/read",
    )
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
    probe.add_argument("--max-concurrent-models", type=int, default=2)
    probe.add_argument("--provider-retries", type=int, default=3)
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
        status = subprocess.run(
            [*prefix, "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT, check=True, capture_output=True,
        ).stdout
        diff = subprocess.run(
            [*prefix, "diff", "--binary", "HEAD", "--"],
            cwd=REPO_ROOT, check=True, capture_output=True,
        ).stdout
        untracked = subprocess.run(
            [*prefix, "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=REPO_ROOT, check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": "unavailable",
            "dirty": None,
            "worktree_sha256": "unavailable",
        }
    fingerprint = hashlib.sha256()
    fingerprint.update(status)
    fingerprint.update(b"\0DIFF\0")
    fingerprint.update(diff)
    for encoded_path in untracked.split(b"\0"):
        if not encoded_path:
            continue
        fingerprint.update(b"\0UNTRACKED\0")
        fingerprint.update(encoded_path)
        path = REPO_ROOT / encoded_path.decode(sys.getfilesystemencoding())
        if path.is_file():
            fingerprint.update(path.read_bytes())
    return {
        "git_commit": commit,
        "dirty": bool(status.strip()),
        "worktree_sha256": fingerprint.hexdigest(),
    }


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_current_source_revision(manifest: Mapping[str, Any]) -> None:
    frozen = manifest.get("source_revision")
    if not isinstance(frozen, Mapping):
        raise ValueError("experiment manifest is missing its source revision")
    if dict(frozen) != _source_revision():
        raise ValueError("current Git revision differs from frozen experiment revision")


def build_experiment_manifest(
    *,
    experiment_id: str,
    required_models: Sequence[str] = REQUIRED_MODELS,
    optional_models: Sequence[str] = OPTIONAL_MODELS,
    judge_model: str = JUDGE_MODEL,
    budget: ResearchBudget | None = None,
    config: AppConfig | None = None,
    max_concurrent_models: int = 2,
    max_provider_retries: int = 3,
) -> dict[str, object]:
    """Build the immutable experiment input without evaluator-only case data."""

    benchmark = load_end_to_end_benchmark()
    required = tuple(_normalized_models(required_models, label="required"))
    optional = (
        tuple(_normalized_models(optional_models, label="optional"))
        if optional_models
        else ()
    )
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
    if max_concurrent_models < 1:
        raise ValueError("max_concurrent_models must be positive")
    if max_provider_retries < 0:
        raise ValueError("max_provider_retries must be non-negative")
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
    candidate_transports = {
        model: ModelTransportProfile(model_id=model).model_dump(mode="json")
        for model in (*required, *optional)
    }
    judge_transport = ModelTransportProfile(
        model_id=normalized_judge,
        structured_api_mode=LLMApiMode.CHAT_COMPLETIONS,
        reasoning_effort=None,
        supports_native_tools=False,
    )
    return {
        "schema_version": "specrag.model-comparison-operational-manifest/v1",
        "comparison": comparison.model_dump(mode="json"),
        "benchmark": benchmark_projection,
        "benchmark_sha256": _sha256_json(benchmark_projection),
        "transports": {
            "candidates": candidate_transports,
            "judge": judge_transport.model_dump(mode="json"),
        },
        "execution": {
            "trial_count_per_model": 1,
            "question_order": "benchmark_order",
            "execution_order": "concurrent_models_sequential_questions",
            "cache_condition": "warm_cache",
            "max_concurrent_models": max_concurrent_models,
            "max_provider_retries": max_provider_retries,
            "latency_condition": (
                "concurrent_models_not_comparable_to_sequential_baseline"
            ),
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


def _probe_binding(
    *,
    comparison: ModelComparisonExperimentManifest,
    manifest_path: Path,
    transport_profile: ModelTransportProfile,
) -> dict[str, str]:
    return {
        "experiment_id": comparison.experiment_id,
        "manifest_sha256": _sha256(manifest_path),
        "transport_profile_sha256": _sha256_json(
            transport_profile.model_dump(mode="json")
        ),
    }


def _probe_artifact(
    *,
    result: Mapping[str, Any],
    binding: Mapping[str, str],
    attempt_count: int = 1,
) -> dict[str, object]:
    if attempt_count < 1:
        raise ValueError("probe attempt count must be positive")
    return {
        "schema_version": "specrag.model-probe-artifact/v1",
        "binding": dict(binding),
        "attempt_count": attempt_count,
        "result": dict(result),
    }


def _load_probe_artifact(
    path: Path,
    *,
    binding: Mapping[str, str],
) -> dict[str, Any]:
    artifact = _load_object(path)
    if artifact.get("schema_version") != "specrag.model-probe-artifact/v1":
        raise ValueError(f"probe artifact has an unknown schema: {path}")
    if artifact.get("binding") != dict(binding):
        raise ValueError(f"probe artifact binding changed: {path}")
    result = artifact.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"probe artifact result must contain an object: {path}")
    return result


def _agent_factory(
    model_id: str,
    timeout_seconds: int,
    transport_profile: ModelTransportProfile,
) -> LLMApiStructuredAgent:
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
        transport_profile=transport_profile,
    )


def _transport_profiles(
    manifest: Mapping[str, Any],
    comparison: ModelComparisonExperimentManifest,
) -> tuple[dict[str, ModelTransportProfile], ModelTransportProfile]:
    value = manifest.get("transports")
    if not isinstance(value, Mapping):
        raise ValueError("experiment manifest is missing transport profiles")
    candidates_value = value.get("candidates")
    if not isinstance(candidates_value, Mapping):
        raise ValueError("experiment manifest is missing candidate transports")
    expected = tuple(target.model_id for target in comparison.model_targets)
    if tuple(candidates_value) != expected:
        raise ValueError("candidate transport order does not match model targets")
    candidates = {
        model_id: ModelTransportProfile.model_validate(candidates_value[model_id])
        for model_id in expected
    }
    if any(profile.model_id != model_id for model_id, profile in candidates.items()):
        raise ValueError("candidate transport profile model mismatch")
    judge = ModelTransportProfile.model_validate(value.get("judge"))
    if judge.model_id != comparison.judge_model:
        raise ValueError("Judge transport profile model mismatch")
    return candidates, judge


def probe_experiment(
    experiment_directory: Path,
    manifest: Mapping[str, Any],
    *,
    timeout_seconds: int,
    agent_factory: Callable[[str, int, ModelTransportProfile], Any] = _agent_factory,
    retry_sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Probe every candidate and the fixed Judge, persisting secret-free results."""

    _validate_current_source_revision(manifest)
    comparison = _comparison_manifest(manifest)
    candidate_transports, judge_transport = _transport_profiles(manifest, comparison)
    experiment_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment_directory / "manifest.json"
    _freeze_json(manifest_path, manifest)
    concurrency = _manifest_concurrency(manifest)
    max_provider_retries = _manifest_provider_retries(manifest)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = tuple(
            executor.submit(
                _probe_candidate_with_retries,
                target,
                transport_profile=candidate_transports[target.model_id],
                timeout_seconds=timeout_seconds,
                agent_factory=agent_factory,
                max_provider_retries=max_provider_retries,
                retry_sleep=retry_sleep,
            )
            for target in comparison.model_targets
        )
        results = tuple(future.result() for future in futures)
    failures = False
    for target, (result, attempt_count) in zip(
        comparison.model_targets, results, strict=True
    ):
        _freeze_json(
            _probe_path(experiment_directory, target.model_id),
            _probe_artifact(
                result=result.model_dump(mode="json"),
                binding=_probe_binding(
                    comparison=comparison,
                    manifest_path=manifest_path,
                    transport_profile=candidate_transports[target.model_id],
                ),
                attempt_count=attempt_count,
            ),
        )
        failures = failures or (target.required and result.status != "ready")

    judge_payload, judge_attempt_count = _probe_judge_with_retries(
        model_id=comparison.judge_model,
        transport_profile=judge_transport,
        timeout_seconds=timeout_seconds,
        agent_factory=agent_factory,
        max_provider_retries=max_provider_retries,
        retry_sleep=retry_sleep,
    )
    _freeze_json(
        _judge_probe_path(experiment_directory),
        _probe_artifact(
            result=judge_payload,
            binding=_probe_binding(
                comparison=comparison,
                manifest_path=manifest_path,
                transport_profile=judge_transport,
            ),
            attempt_count=judge_attempt_count,
        ),
    )
    failures = failures or judge_payload["status"] != "ready"
    _validate_current_source_revision(manifest)
    return 1 if failures else 0


def _probe_candidate(
    target: ModelTarget,
    *,
    transport_profile: ModelTransportProfile,
    timeout_seconds: int,
    agent_factory: Callable[[str, int, ModelTransportProfile], Any],
) -> ModelProbeResult:
    """Probe one candidate without writing shared files or emitting output."""

    try:
        agent = agent_factory(target.model_id, timeout_seconds, transport_profile)
        with capture_agent_receipts() as receipts:
            capability = probe_research_model_capability(agent)
        payloads = [item.model_dump(mode="json") for item in receipts]
    except Exception as exc:  # noqa: BLE001 - safe probe projection below
        return ModelProbeResult(
            model_id=target.model_id,
            status="unavailable",
            failure_kind=str(
                getattr(exc, "failure_kind", "configuration")
            )[:200],
            message=(
                "The required model probe could not be initialized."
                if target.required
                else (
                    "The optional model probe could not be initialized and was skipped."
                )
            ),
        )
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
    return ModelProbeResult(
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


_RETRYABLE_PROBE_FAILURES = frozenset({
    "provider_unavailable", "rate_limited", "retryable_transient", "timeout",
})


def _probe_candidate_with_retries(
    target: ModelTarget,
    *,
    transport_profile: ModelTransportProfile,
    timeout_seconds: int,
    agent_factory: Callable[[str, int, ModelTransportProfile], Any],
    max_provider_retries: int,
    retry_sleep: Callable[[float], None],
) -> tuple[ModelProbeResult, int]:
    for attempt in range(1, max_provider_retries + 2):
        result = _probe_candidate(
            target,
            transport_profile=transport_profile,
            timeout_seconds=timeout_seconds,
            agent_factory=agent_factory,
        )
        if (
            result.status == "ready"
            or result.failure_kind not in _RETRYABLE_PROBE_FAILURES
            or attempt > max_provider_retries
        ):
            return result, attempt
        retry_sleep(min(60.0, 5.0 * (2 ** (attempt - 1))))
    raise AssertionError("candidate probe retry loop did not terminate")


def _probe_judge_with_retries(
    *,
    model_id: str,
    transport_profile: ModelTransportProfile,
    timeout_seconds: int,
    agent_factory: Callable[[str, int, ModelTransportProfile], Any],
    max_provider_retries: int,
    retry_sleep: Callable[[float], None],
) -> tuple[dict[str, object], int]:
    for attempt in range(1, max_provider_retries + 2):
        try:
            agent = agent_factory(model_id, timeout_seconds, transport_profile)
            payload = _probe_judge(agent, model_id)
        except Exception as exc:  # noqa: BLE001 - safe probe projection below
            payload = {
                "schema_version": "specrag.judge-model-probe/v1",
                "model_id": model_id,
                "status": "unavailable",
                "failure_kind": str(
                    getattr(exc, "failure_kind", "configuration")
                )[:200],
                "message": "The fixed Judge probe could not be initialized.",
            }
        if (
            payload["status"] == "ready"
            or payload.get("failure_kind") not in _RETRYABLE_PROBE_FAILURES
            or attempt > max_provider_retries
        ):
            return payload, attempt
        retry_sleep(min(60.0, 5.0 * (2 ** (attempt - 1))))
    raise AssertionError("Judge probe retry loop did not terminate")


def _manifest_concurrency(manifest: Mapping[str, Any]) -> int:
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        return 2
    value = execution.get("max_concurrent_models", 2)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("manifest max_concurrent_models must be a positive integer")
    return value


def _manifest_provider_retries(manifest: Mapping[str, Any]) -> int:
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("experiment manifest is missing execution settings")
    value = execution.get("max_provider_retries")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("manifest max_provider_retries must be non-negative")
    return value


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
    manifest: Mapping[str, Any],
    comparison: ModelComparisonExperimentManifest,
) -> tuple[ModelProbeResult, ...]:
    candidate_transports, _ = _transport_profiles(manifest, comparison)
    manifest_path = directory / "manifest.json"
    return tuple(
        ModelProbeResult.model_validate(_load_probe_artifact(
            _probe_path(directory, target.model_id),
            binding=_probe_binding(
                comparison=comparison,
                manifest_path=manifest_path,
                transport_profile=candidate_transports[target.model_id],
            ),
        ))
        for target in comparison.model_targets
    )


def _load_judge_probe(
    directory: Path,
    manifest: Mapping[str, Any],
    comparison: ModelComparisonExperimentManifest,
) -> dict[str, Any]:
    _, judge_transport = _transport_profiles(manifest, comparison)
    return _load_probe_artifact(
        _judge_probe_path(directory),
        binding=_probe_binding(
            comparison=comparison,
            manifest_path=directory / "manifest.json",
            transport_profile=judge_transport,
        ),
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


def _validated_workflow_artifact(
    *,
    directory: Path,
    manifest_path: Path,
    comparison: ModelComparisonExperimentManifest,
    target: ModelTarget,
    question: Any,
) -> tuple[Path, dict[str, Any]]:
    path = directory / "runs" / _model_slug(target.model_id) / f"{question.question_id}.json"
    artifact = _load_object(path)
    validate_model_cell_artifact(
        artifact,
        experiment_id=comparison.experiment_id,
        requested_model=target.model_id,
        trial=1,
        source_manifest_sha256=_sha256(manifest_path),
        question=question,
    )
    return path, artifact


def _expected_judge_manifest(
    *,
    comparison: ModelComparisonExperimentManifest,
    target: ModelTarget,
    judge_transport: ModelTransportProfile,
    completed: Sequence[tuple[Any, Path]],
) -> dict[str, object]:
    difficulty_by_id = {
        item.question_id: item.difficulty for item in comparison.cases
    }
    return {
        "schema_version": "specrag.final-answer-judge-batch/v1",
        "benchmark_id": comparison.benchmark_id,
        "benchmark_version": comparison.benchmark_version,
        "judge_prompt_version": comparison.judge_prompt_version,
        "backend": "api",
        "configured_model": comparison.judge_model,
        "transport_profile": judge_transport.model_dump(mode="json"),
        "difficulty": "all",
        "question_ids": [question.question_id for question, _ in completed],
        "source_artifacts": [
            {
                "question_id": question.question_id,
                "difficulty": difficulty_by_id[question.question_id],
                "path": path.resolve().as_posix(),
                "sha256": _sha256(path),
            }
            for question, path in completed
        ],
    }


def _validate_judge_batch(
    *,
    directory: Path,
    comparison: ModelComparisonExperimentManifest,
    target: ModelTarget,
    judge_transport: ModelTransportProfile,
    completed: Sequence[tuple[Any, Path, Mapping[str, Any]]],
) -> tuple[FinalAnswerJudgeExecution, ...]:
    judgment_directory = directory / "judgments" / _model_slug(target.model_id)
    manifest_path = judgment_directory / "manifest.json"
    expected_manifest = _expected_judge_manifest(
        comparison=comparison,
        target=target,
        judge_transport=judge_transport,
        completed=tuple((question, path) for question, path, _ in completed),
    )
    if _load_object(manifest_path) != expected_manifest:
        raise ValueError(f"Judge batch manifest changed: {manifest_path}")
    expected_names = {f"{question.question_id}.json" for question, _, _ in completed}
    actual_names = {path.name for path in judgment_directory.glob("r17-qa-*.json")}
    if actual_names != expected_names:
        raise ValueError("Judge result set does not match completed Workflow artifacts")
    integrity_directory = judgment_directory / "integrity"
    integrity_names = {path.name for path in integrity_directory.glob("*.json")}
    if integrity_names != expected_names:
        raise ValueError("Judge integrity receipt set is incomplete or unexpected")
    return tuple(
        final_answer_runner.validate_judge_artifact(
            execution_path=judgment_directory / f"{question.question_id}.json",
            source_path=source_path,
            batch_manifest_path=manifest_path,
            integrity_path=integrity_directory / f"{question.question_id}.json",
            question_id=question.question_id,
            run_id=final_answer_runner._run_id(
                artifact, question_id=question.question_id
            ),
            judge_model=comparison.judge_model,
        )
        for question, source_path, artifact in completed
    )


def run_experiment(directory: Path, *, resume: bool) -> int:
    manifest_path = directory / "manifest.json"
    manifest = _load_object(manifest_path)
    _validate_current_source_revision(manifest)
    comparison = _comparison_manifest(manifest)
    candidate_transports, _ = _transport_profiles(manifest, comparison)
    probes = _load_probes(directory, manifest, comparison)
    ready = _ready_targets(comparison, probes)
    for target in ready:
        expected = ModelTransportProfile(model_id=target.model_id)
        if candidate_transports[target.model_id] != expected:
            raise ValueError(
                "Workflow candidate transport must use Responses/high/native tools"
            )
    benchmark = load_end_to_end_benchmark()
    questions = benchmark_questions(
        benchmark, tuple(case.question_id for case in comparison.cases)
    )
    budget = ResearchBudget.model_validate(manifest["execution"]["budget"])
    concurrency = _manifest_concurrency(manifest)
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("experiment manifest is missing execution settings")
    max_provider_retries = execution.get("max_provider_retries")
    if (
        not isinstance(max_provider_retries, int)
        or isinstance(max_provider_retries, bool)
        or max_provider_retries < 0
    ):
        raise ValueError("manifest max_provider_retries must be non-negative")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = tuple(
            executor.submit(
                run_model_cell,
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
                emit_progress=False,
                max_provider_retries=max_provider_retries,
            )
            for target in ready
        )
        outcomes: list[int] = []
        failures: list[str] = []
        for target, future in zip(ready, futures, strict=True):
            try:
                outcome = future.result()
            except Exception as exc:  # noqa: BLE001 - safe coordinator projection
                failures.append(f"{target.model_id}:{type(exc).__name__}")
                outcome = 1
            outcomes.append(outcome)
            print(json.dumps({
                "model_id": target.model_id,
                "status": "completed" if outcome == 0 else "failed",
            }, ensure_ascii=False), flush=True)
    if failures:
        raise ValueError("model comparison cells failed: " + ", ".join(failures))
    _validate_current_source_revision(manifest)
    return max(outcomes, default=0)


def judge_experiment(directory: Path, *, resume: bool, timeout_seconds: int) -> int:
    manifest_path = directory / "manifest.json"
    manifest = _load_object(manifest_path)
    _validate_current_source_revision(manifest)
    comparison = _comparison_manifest(manifest)
    _, judge_transport = _transport_profiles(manifest, comparison)
    probes = _load_probes(directory, manifest, comparison)
    ready = _ready_targets(comparison, probes)
    judge_probe = _load_judge_probe(directory, manifest, comparison)
    if (
        judge_probe.get("status") != "ready"
        or judge_probe.get("model_id") != comparison.judge_model
    ):
        raise ValueError("fixed Judge probe is not ready")
    result = 0
    benchmark = load_end_to_end_benchmark()
    questions = benchmark_questions(
        benchmark, tuple(case.question_id for case in comparison.cases)
    )
    for target in ready:
        run_directory = directory / "runs" / _model_slug(target.model_id)
        artifacts = tuple(
            _validated_workflow_artifact(
                directory=directory,
                manifest_path=manifest_path,
                comparison=comparison,
                target=target,
                question=question,
            )
            for question in questions
        )
        completed_ids = tuple(
            question.question_id
            for question, (_, artifact) in zip(questions, artifacts, strict=True)
            if artifact["workflow_status"] == "completed"
        )
        if not completed_ids:
            continue
        argv = [
            str(run_directory),
            str(directory / "judgments" / _model_slug(target.model_id)),
            "--backend", "api",
            "--judge-model", comparison.judge_model,
            "--judge-api-mode", judge_transport.structured_api_mode.value,
            "--judge-reasoning-effort", (
                judge_transport.reasoning_effort or "provider_default"
            ),
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
    _validate_current_source_revision(manifest)
    return result


def summarize_experiment(directory: Path) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = _load_object(manifest_path)
    _validate_current_source_revision(manifest)
    comparison = _comparison_manifest(manifest)
    probes = _load_probes(directory, manifest, comparison)
    ready = _ready_targets(comparison, probes)
    _, judge_transport = _transport_profiles(manifest, comparison)
    benchmark = load_end_to_end_benchmark()
    questions = benchmark_questions(
        benchmark, tuple(case.question_id for case in comparison.cases)
    )
    results: list[ModelCaseResult] = []
    for target in ready:
        validated = tuple(
            (
                question,
                *_validated_workflow_artifact(
                    directory=directory,
                    manifest_path=manifest_path,
                    comparison=comparison,
                    target=target,
                    question=question,
                ),
            )
            for question in questions
        )
        completed = tuple(
            item for item in validated if item[2]["workflow_status"] == "completed"
        )
        executions = _validate_judge_batch(
            directory=directory,
            comparison=comparison,
            target=target,
            judge_transport=judge_transport,
            completed=completed,
        ) if completed else ()
        execution_by_id = {item.question_id: item for item in executions}
        for case, (_, _, run) in zip(comparison.cases, validated, strict=True):
            if run.get("workflow_status") == "failed":
                results.append(ModelCaseResult(
                    model_id=target.model_id,
                    question_id=case.question_id,
                    workflow_status="failed",
                    judge_status="not_run",
                    score=None,
                ))
                continue
            execution = execution_by_id[case.question_id]
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
    _validate_current_source_revision(manifest)
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
        if args.command == "langsmith-check":
            result = build_langsmith_tracing(load_config()).preflight()
            print(json.dumps({
                "status": result.status,
                "failure_kind": result.failure_kind,
                "retryable": result.retryable,
            }, ensure_ascii=False), flush=True)
            return 0 if result.status == "ready" else 1
        if args.command == "probe":
            config = load_config()
            _require_langsmith_ready(config)
            experiment_id = args.experiment_id or (
                "specrag-model-comparison-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            )
            manifest = build_experiment_manifest(
                experiment_id=experiment_id,
                required_models=tuple(args.required_models or REQUIRED_MODELS),
                optional_models=tuple(args.optional_models or OPTIONAL_MODELS),
                judge_model=args.judge_model,
                config=config,
                max_concurrent_models=args.max_concurrent_models,
                max_provider_retries=args.provider_retries,
            )
            return probe_experiment(
                args.experiment_directory,
                manifest,
                timeout_seconds=args.timeout_seconds,
            )
        if args.command == "run":
            _require_langsmith_ready()
            return run_experiment(args.experiment_directory, resume=args.resume)
        if args.command == "judge":
            _require_langsmith_ready()
            return judge_experiment(
                args.experiment_directory,
                resume=args.resume,
                timeout_seconds=args.timeout_seconds,
            )
        payload = summarize_experiment(args.experiment_directory)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (
        LangSmithPreflightError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
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

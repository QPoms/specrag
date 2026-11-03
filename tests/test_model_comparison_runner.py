from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from scripts import evaluate_model_comparison as runner
from specrag.config import AppConfig, Secrets
from specrag.evaluation import (
    ComparisonCase,
    ModelComparisonExperimentManifest,
    ModelProbeResult,
    ModelTarget,
    execute_final_answer_judge,
    load_end_to_end_benchmark,
)
from specrag.infrastructure.model_capabilities import (
    ProviderCapabilityCheck,
    ResearchModelCapability,
)
from specrag.infrastructure.llm_contracts import LLMApiMode
from specrag.infrastructure.model_transport import ModelTransportProfile
from specrag.interfaces.model_comparison import (
    _safe_failure_summary,
    benchmark_questions,
    candidate_model_config,
    validate_candidate_receipts,
)
from specrag.interfaces import model_comparison as comparison_interface
from specrag.models import ResearchBudget
from specrag.prompts import FINAL_ANSWER_JUDGE_PROMPT_VERSION
from specrag.orchestration import new_research_state


ROOT = Path(__file__).resolve().parents[1]


def _config() -> AppConfig:
    return AppConfig(secrets=Secrets(_env_file=None))


def _small_manifest() -> ModelComparisonExperimentManifest:
    return ModelComparisonExperimentManifest(
        experiment_id="comparison-test",
        benchmark_id="specrag-release17-final-answer",
        benchmark_version="1.0.0",
        judge_model="deepseek-v4-pro",
        judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
        cases=(
            ComparisonCase(question_id="r17-qa-001", difficulty="simple"),
            ComparisonCase(question_id="r17-qa-007", difficulty="complex"),
        ),
        model_targets=(ModelTarget(model_id="gpt-5.6-sol", required=True),),
    )


def _operational_manifest(
    comparison: ModelComparisonExperimentManifest | None = None,
) -> dict[str, object]:
    active = comparison or _small_manifest()
    required = tuple(
        target.model_id for target in active.model_targets if target.required
    )
    optional = tuple(
        target.model_id for target in active.model_targets if not target.required
    )
    manifest = runner.build_experiment_manifest(
        experiment_id=active.experiment_id,
        required_models=required,
        optional_models=optional,
        judge_model=active.judge_model,
        config=_config(),
        max_provider_retries=0,
    )
    manifest["comparison"] = active.model_dump(mode="json")
    manifest["transports"] = _transports(active)
    manifest["execution"]["budget"] = {}
    return manifest


def _transports(
    comparison: ModelComparisonExperimentManifest,
) -> dict[str, object]:
    return {
        "candidates": {
            target.model_id: ModelTransportProfile(
                model_id=target.model_id
            ).model_dump(mode="json")
            for target in comparison.model_targets
        },
        "judge": ModelTransportProfile(
            model_id=comparison.judge_model,
            structured_api_mode=LLMApiMode.CHAT_COMPLETIONS,
            reasoning_effort=None,
            supports_native_tools=False,
        ).model_dump(mode="json"),
    }


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_ready_probes(
    directory: Path,
    manifest: dict[str, object],
) -> None:
    manifest_path = directory / "manifest.json"
    comparison = ModelComparisonExperimentManifest.model_validate(
        manifest["comparison"]
    )
    candidates, judge = runner._transport_profiles(manifest, comparison)
    for target in comparison.model_targets:
        result = ModelProbeResult(
            model_id=target.model_id,
            status="ready",
            observed_model=target.model_id,
            message="ready",
        ).model_dump(mode="json")
        _write(
            runner._probe_path(directory, target.model_id),
            runner._probe_artifact(
                result=result,
                binding=runner._probe_binding(
                    comparison=comparison,
                    manifest_path=manifest_path,
                    transport_profile=candidates[target.model_id],
                ),
            ),
        )
    _write(
        runner._judge_probe_path(directory),
        runner._probe_artifact(
            result={
                "schema_version": "specrag.judge-model-probe/v1",
                "model_id": comparison.judge_model,
                "status": "ready",
                "observed_model": comparison.judge_model,
                "message": "ready",
            },
            binding=runner._probe_binding(
                comparison=comparison,
                manifest_path=manifest_path,
                transport_profile=judge,
            ),
        ),
    )


def _ready_capability() -> ResearchModelCapability:
    check = ProviderCapabilityCheck(status="ready", message="ready")
    return ResearchModelCapability(
        status="ready",
        structured_output=check,
        native_tool_replay=check,
    )


def test_candidate_config_binds_all_roles_and_disables_fallback() -> None:
    config = candidate_model_config("deepseek-v4-flash-0731", _config())

    assert config.secrets.openai_model == "deepseek-v4-flash-0731"
    assert config.secrets.openai_planning_model == "deepseek-v4-flash-0731"
    assert config.secrets.openai_summary_model == "deepseek-v4-flash-0731"
    assert config.llm.api_fallback_models == ()


def test_candidate_receipts_reject_cross_model_fallback() -> None:
    with pytest.raises(ValueError, match="cross-model fallback"):
        validate_candidate_receipts(
            ({
                "model": "gpt-5.6-sol",
                "attempted_models": ["gpt-5.6-sol", "gpt-6-astra"],
            },),
            "gpt-5.6-sol",
        )


def test_workflow_failure_summary_does_not_persist_raw_provider_text() -> None:
    error = RuntimeError("raw provider body with sensitive details")
    error.failure_kind = "provider_unavailable"
    error.public_message = "The provider is temporarily unavailable."

    summary = _safe_failure_summary(error)

    assert "raw provider body" not in summary
    assert "provider_unavailable" in summary
    assert "temporarily unavailable" in summary


def test_retryable_provider_failure_uses_bounded_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = load_end_to_end_benchmark()
    question = benchmark_questions(benchmark, ("r17-qa-001",))[0]
    delays: list[float] = []

    class RateLimited(RuntimeError):
        failure_kind = "rate_limited"
        retryable = True
        public_message = "The provider rate-limited the request."

    monkeypatch.setattr(
        comparison_interface,
        "_run_application",
        lambda *args, **kwargs: (_ for _ in ()).throw(RateLimited()),
    )

    artifact = comparison_interface._execute_question_with_retries(
        app=SimpleNamespace(retrieval_index_fingerprint="index"),
        question=question,
        budget=ResearchBudget(),
        experiment_id="retry-test",
        benchmark=benchmark,
        requested_model="gpt-6-astra",
        trial=1,
        source_manifest_sha256="a" * 64,
        source_revision={"commit": "abc", "dirty": False},
        max_provider_retries=3,
        retry_sleep=delays.append,
    )

    assert artifact["workflow_status"] == "failed"
    assert len(artifact["workflow_attempts"]) == 4
    assert delays == [5.0, 10.0, 20.0]


def test_completed_workflow_uses_authoritative_state_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = load_end_to_end_benchmark()
    question = benchmark_questions(benchmark, ("r17-qa-001",))[0]
    state = new_research_state(
        research_id="research-1",
        user_question=question.question,
    )
    state["model_receipts"] = ({
        "model": "gpt-6-astra",
        "attempted_models": ["gpt-6-astra"],
    },)
    monkeypatch.setattr(
        comparison_interface,
        "_run_application",
        lambda *args, **kwargs: state,
    )

    artifact = comparison_interface._execute_question_with_retries(
        app=SimpleNamespace(retrieval_index_fingerprint="index"),
        question=question,
        budget=ResearchBudget(),
        experiment_id="receipt-test",
        benchmark=benchmark,
        requested_model="gpt-6-astra",
        trial=1,
        source_manifest_sha256="a" * 64,
        source_revision={"commit": "abc", "dirty": False},
        max_provider_retries=0,
        retry_sleep=lambda delay: pytest.fail("sleep is not expected"),
    )

    assert artifact["workflow_status"] == "completed"
    assert artifact["model_receipts"] == [
        {"model": "gpt-6-astra", "attempted_models": ["gpt-6-astra"]}
    ]


def test_operational_manifest_has_fixed_models_and_no_evaluator_payload() -> None:
    manifest = runner.build_experiment_manifest(
        experiment_id="test-experiment",
        config=_config(),
    )

    comparison = ModelComparisonExperimentManifest.model_validate(
        manifest["comparison"]
    )
    assert [item.model_id for item in comparison.model_targets] == [
        "gpt-5.6-sol",
    ]
    assert comparison.judge_model == "deepseek-v4-pro"
    assert manifest["execution"]["fallback_models"] == []
    assert manifest["transports"]["judge"] == {
        "model_id": "deepseek-v4-pro",
        "structured_api_mode": "chat_completions",
        "reasoning_effort": None,
        "supports_native_tools": False,
    }
    assert manifest["transports"]["candidates"]["gpt-5.6-sol"] == {
        "model_id": "gpt-5.6-sol",
        "structured_api_mode": "responses",
        "reasoning_effort": "high",
        "supports_native_tools": True,
    }
    assert manifest["execution"]["max_concurrent_models"] == 2
    assert manifest["execution"]["max_provider_retries"] == 3
    assert manifest["execution"]["latency_condition"] == (
        "concurrent_models_not_comparable_to_sequential_baseline"
    )
    serialized = json.dumps(manifest).lower()
    assert "reference_answer" not in serialized
    assert "gold_evidence" not in serialized
    assert '"rubric"' not in serialized


def test_probe_blocks_required_initialization_failure_and_skips_optional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _small_manifest().model_copy(update={
        "model_targets": (
            ModelTarget(model_id="gpt-5.6-sol", required=True),
            ModelTarget(model_id="claude-opus-5", required=False),
        )
    })
    manifest = _operational_manifest(comparison)

    class FakeAgent:
        def __init__(self, model: str) -> None:
            self.model = model

        def parse(self, **kwargs):
            return SimpleNamespace(receipt=SimpleNamespace(model=self.model))

    def factory(model: str, timeout: int, transport_profile):
        if model in {"gpt-5.6-sol", "claude-opus-5"}:
            raise RuntimeError("sensitive upstream text must not be persisted")
        return FakeAgent(model)

    monkeypatch.setattr(runner, "validate_candidate_receipts", lambda *args: None)
    result = runner.probe_experiment(
        tmp_path,
        manifest,
        timeout_seconds=1,
        agent_factory=factory,
    )

    assert result == 1
    required = json.loads((tmp_path / "probes/gpt-5.6-sol.json").read_text())
    optional = json.loads((tmp_path / "probes/claude-opus-5.json").read_text())
    assert required["result"]["status"] == "unavailable"
    assert optional["result"]["status"] == "unavailable"
    assert "sensitive upstream text" not in json.dumps((required, optional))


def test_probe_accepts_ready_candidates_and_fixed_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _operational_manifest()

    class FakeAgent:
        def __init__(self, model: str) -> None:
            self.model = model

        def parse(self, **kwargs):
            return SimpleNamespace(receipt=SimpleNamespace(model=self.model))

    monkeypatch.setattr(
        runner, "probe_research_model_capability", lambda agent: _ready_capability()
    )
    monkeypatch.setattr(runner, "validate_candidate_receipts", lambda *args: None)
    @contextmanager
    def fake_capture():
        yield [SimpleNamespace(model_dump=lambda **kwargs: {"model": "fake"})]

    monkeypatch.setattr(runner, "capture_agent_receipts", fake_capture)

    result = runner.probe_experiment(
        tmp_path,
        manifest,
        timeout_seconds=1,
        agent_factory=lambda model, timeout, profile: FakeAgent(model),
    )

    assert result == 0
    candidate = json.loads((tmp_path / "probes/gpt-5.6-sol.json").read_text())
    judge = json.loads((tmp_path / "probes/judge.json").read_text())
    assert candidate["result"]["status"] == "ready"
    assert judge["result"] == {
        "schema_version": "specrag.judge-model-probe/v1",
        "model_id": "deepseek-v4-pro",
        "status": "ready",
        "observed_model": "deepseek-v4-pro",
        "message": "The fixed Judge passed its strict structured-output probe.",
    }


def test_candidate_probes_run_with_frozen_model_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _small_manifest().model_copy(update={
        "model_targets": (
            ModelTarget(model_id="model-a", required=True),
            ModelTarget(model_id="model-b", required=True),
        )
    })
    manifest = _operational_manifest(comparison)
    manifest["execution"]["max_concurrent_models"] = 2
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    maximum = 0

    class FakeAgent:
        def __init__(self, model: str) -> None:
            self.model = model

        def parse(self, **kwargs):
            return SimpleNamespace(receipt=SimpleNamespace(model=self.model))

    def capability(agent):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        return _ready_capability()

    @contextmanager
    def fake_capture():
        yield [SimpleNamespace(model_dump=lambda **kwargs: {"model": "fake"})]

    monkeypatch.setattr(runner, "probe_research_model_capability", capability)
    monkeypatch.setattr(runner, "validate_candidate_receipts", lambda *args: None)
    monkeypatch.setattr(runner, "capture_agent_receipts", fake_capture)

    assert runner.probe_experiment(
        tmp_path,
        manifest,
        timeout_seconds=1,
        agent_factory=lambda model, timeout, profile: FakeAgent(model),
    ) == 0
    assert maximum == 2


def test_candidate_probe_retries_only_transient_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = iter((
        ModelProbeResult(
            model_id="gpt-6-astra",
            status="unavailable",
            failure_kind="rate_limited",
            message="retry",
        ),
        ModelProbeResult(
            model_id="gpt-6-astra",
            status="unavailable",
            failure_kind="retryable_transient",
            message="retry",
        ),
        ModelProbeResult(
            model_id="gpt-6-astra",
            status="ready",
            observed_model="gpt-6-astra",
            message="ready",
        ),
    ))
    monkeypatch.setattr(runner, "_probe_candidate", lambda *args, **kwargs: next(outcomes))
    delays: list[float] = []

    result, attempts = runner._probe_candidate_with_retries(
        ModelTarget(model_id="gpt-6-astra", required=True),
        transport_profile=ModelTransportProfile(model_id="gpt-6-astra"),
        timeout_seconds=1,
        agent_factory=lambda *args: object(),
        max_provider_retries=3,
        retry_sleep=delays.append,
    )

    assert result.status == "ready"
    assert attempts == 3
    assert delays == [5.0, 10.0]


def test_workflow_models_run_concurrently_but_report_in_manifest_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    comparison = _small_manifest().model_copy(update={
        "model_targets": (
            ModelTarget(model_id="model-a", required=True),
            ModelTarget(model_id="model-b", required=True),
        )
    })
    manifest = _operational_manifest(comparison)
    manifest["execution"]["max_concurrent_models"] = 2
    _write(tmp_path / "manifest.json", manifest)
    _write_ready_probes(tmp_path, manifest)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    maximum = 0

    def fake_cell(**kwargs):
        nonlocal active, maximum
        assert kwargs["emit_progress"] is False
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait(timeout=2)
        if kwargs["requested_model"] == "model-a":
            time.sleep(0.02)
        with lock:
            active -= 1
        return 0

    monkeypatch.setattr(runner, "run_model_cell", fake_cell)

    assert runner.run_experiment(tmp_path, resume=False) == 0
    assert maximum == 2
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["model_id"] for item in lines] == ["model-a", "model-b"]


def test_judge_phase_uses_fixed_model_and_only_completed_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _operational_manifest()
    _write(tmp_path / "manifest.json", manifest)
    _write_ready_probes(tmp_path, manifest)
    run_dir = tmp_path / "runs/gpt-5.6-sol"
    monkeypatch.setattr(
        runner,
        "_validated_workflow_artifact",
        lambda **kwargs: (
            run_dir / f"{kwargs['question'].question_id}.json",
            {"workflow_status": (
                "completed"
                if kwargs["question"].question_id == "r17-qa-007"
                else "failed"
            )},
        ),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        runner.final_answer_runner,
        "main",
        lambda argv: calls.append(argv) or 0,
    )

    assert runner.judge_experiment(tmp_path, resume=True, timeout_seconds=7) == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--judge-model") + 1] == "deepseek-v4-pro"
    assert calls[0][calls[0].index("--judge-api-mode") + 1] == "chat_completions"
    assert calls[0][calls[0].index("--judge-reasoning-effort") + 1] == (
        "provider_default"
    )
    assert calls[0][calls[0].index("--trace-experiment-id") + 1] == (
        "comparison-test"
    )
    assert calls[0][calls[0].index("--trace-candidate-model") + 1] == (
        "gpt-5.6-sol"
    )
    assert "r17-qa-007" in calls[0]
    assert "r17-qa-001" not in calls[0]
    assert "--resume" in calls[0]


def test_summary_keeps_workflow_failure_separate_from_no_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _operational_manifest()
    _write(tmp_path / "manifest.json", manifest)
    _write_ready_probes(tmp_path, manifest)
    run_dir = tmp_path / "runs/gpt-5.6-sol"
    case = next(
        item for item in load_end_to_end_benchmark().cases
        if item.task_metadata.task_id == "r17-qa-007"
    )
    no_answer = execute_final_answer_judge(
        case=case,
        run_id="run-007",
        candidate_answer=None,
        judge=lambda *args, **kwargs: pytest.fail("Judge must not be called"),
        judge_prompt_version=FINAL_ANSWER_JUDGE_PROMPT_VERSION,
    )
    monkeypatch.setattr(
        runner,
        "_validated_workflow_artifact",
        lambda **kwargs: (
            run_dir / f"{kwargs['question'].question_id}.json",
            {"workflow_status": (
                "completed"
                if kwargs["question"].question_id == "r17-qa-007"
                else "failed"
            )},
        ),
    )
    monkeypatch.setattr(runner, "_validate_judge_batch", lambda **kwargs: (no_answer,))

    summary = runner.summarize_experiment(tmp_path)

    model = summary["model_summaries"][0]
    assert model["workflow_failed_count"] == 1
    assert model["no_answer_count"] == 1
    assert model["judge_failed_count"] == 0
    assert model["overall"]["observed_average_score"] == 0.0
    assert "task_success" not in json.dumps(summary).lower()
    assert "pass@" not in json.dumps(summary).lower()


def test_script_help_works_when_executed_by_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_model_comparison.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "{langsmith-check,probe,run,judge,summarize}" in result.stdout

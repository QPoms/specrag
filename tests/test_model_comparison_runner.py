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
from specrag.interfaces.model_comparison import (
    _safe_failure_summary,
    candidate_model_config,
    validate_candidate_receipts,
)
from specrag.prompts import FINAL_ANSWER_JUDGE_PROMPT_VERSION


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


def _operational_manifest() -> dict[str, object]:
    return {
        "schema_version": "specrag.model-comparison-operational-manifest/v1",
        "comparison": _small_manifest().model_dump(mode="json"),
        "execution": {"budget": {}},
    }


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _ready_capability() -> ResearchModelCapability:
    check = ProviderCapabilityCheck(status="ready", message="ready")
    return ResearchModelCapability(
        status="ready",
        structured_output=check,
        native_tool_replay=check,
    )


def test_candidate_config_binds_all_roles_and_disables_fallback() -> None:
    config = candidate_model_config("deepseek-v4.1-flash", _config())

    assert config.secrets.openai_model == "deepseek-v4.1-flash"
    assert config.secrets.openai_planning_model == "deepseek-v4.1-flash"
    assert config.secrets.openai_summary_model == "deepseek-v4.1-flash"
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
        "gpt-6-astra",
        "deepseek-v4.1-flash",
        "claude-opus-5",
    ]
    assert comparison.judge_model == "deepseek-v4-pro"
    assert manifest["execution"]["fallback_models"] == []
    assert manifest["execution"]["max_concurrent_models"] == 2
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
    manifest = {
        "schema_version": "specrag.model-comparison-operational-manifest/v1",
        "comparison": comparison.model_dump(mode="json"),
    }

    class FakeAgent:
        def __init__(self, model: str) -> None:
            self.model = model

        def parse(self, **kwargs):
            return SimpleNamespace(receipt=SimpleNamespace(model=self.model))

    def factory(model: str, timeout: int):
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
    assert required["status"] == "unavailable"
    assert optional["status"] == "unavailable"
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
        agent_factory=lambda model, timeout: FakeAgent(model),
    )

    assert result == 0
    candidate = json.loads((tmp_path / "probes/gpt-5.6-sol.json").read_text())
    judge = json.loads((tmp_path / "probes/judge.json").read_text())
    assert candidate["status"] == "ready"
    assert judge == {
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
    manifest = {
        "schema_version": "specrag.model-comparison-operational-manifest/v1",
        "comparison": comparison.model_dump(mode="json"),
        "execution": {"max_concurrent_models": 2},
    }
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
        agent_factory=lambda model, timeout: FakeAgent(model),
    ) == 0
    assert maximum == 2


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
    _write(tmp_path / "manifest.json", {
        "schema_version": "specrag.model-comparison-operational-manifest/v1",
        "comparison": comparison.model_dump(mode="json"),
        "execution": {"budget": {}, "max_concurrent_models": 2},
    })
    for model in ("model-a", "model-b"):
        _write(
            tmp_path / f"probes/{model}.json",
            ModelProbeResult(
                model_id=model, status="ready", observed_model=model, message="ready"
            ).model_dump(mode="json"),
        )
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
    _write(
        tmp_path / "probes/gpt-5.6-sol.json",
        ModelProbeResult(
            model_id="gpt-5.6-sol", status="ready",
            observed_model="gpt-5.6-sol", message="ready",
        ).model_dump(mode="json"),
    )
    _write(tmp_path / "probes/judge.json", {
        "model_id": "deepseek-v4-pro", "status": "ready",
    })
    run_dir = tmp_path / "runs/gpt-5.6-sol"
    _write(run_dir / "r17-qa-001.json", {"workflow_status": "failed"})
    _write(run_dir / "r17-qa-007.json", {"workflow_status": "completed"})
    calls: list[list[str]] = []
    monkeypatch.setattr(
        runner.final_answer_runner,
        "main",
        lambda argv: calls.append(argv) or 0,
    )

    assert runner.judge_experiment(tmp_path, resume=True, timeout_seconds=7) == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--judge-model") + 1] == "deepseek-v4-pro"
    assert calls[0][calls[0].index("--trace-experiment-id") + 1] == (
        "comparison-test"
    )
    assert calls[0][calls[0].index("--trace-candidate-model") + 1] == (
        "gpt-5.6-sol"
    )
    assert "r17-qa-007" in calls[0]
    assert "r17-qa-001" not in calls[0]
    assert "--resume" in calls[0]


def test_summary_keeps_workflow_failure_separate_from_no_answer(tmp_path: Path) -> None:
    manifest = _operational_manifest()
    _write(tmp_path / "manifest.json", manifest)
    _write(
        tmp_path / "probes/gpt-5.6-sol.json",
        ModelProbeResult(
            model_id="gpt-5.6-sol", status="ready",
            observed_model="gpt-5.6-sol", message="ready",
        ).model_dump(mode="json"),
    )
    run_dir = tmp_path / "runs/gpt-5.6-sol"
    _write(run_dir / "r17-qa-001.json", {"workflow_status": "failed"})
    _write(run_dir / "r17-qa-007.json", {"workflow_status": "completed"})
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
    _write(
        tmp_path / "judgments/gpt-5.6-sol/r17-qa-007.json",
        no_answer.model_dump(mode="json"),
    )

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
    assert "{probe,run,judge,summarize}" in result.stdout

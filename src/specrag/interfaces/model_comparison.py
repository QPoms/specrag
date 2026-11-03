"""Reproducible complete-workflow execution for one comparison model cell."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import inspect
import json
from pathlib import Path
import time
import uuid
from typing import Any

from pydantic import TypeAdapter

from specrag.config import AppConfig, load_config
from specrag.evaluation import EndToEndBenchmark, load_end_to_end_benchmark
from specrag.infrastructure.structured_agent import capture_agent_receipts
from specrag.interfaces.baseline_inputs import BaselineQuestion
from specrag.interfaces.baseline_metadata import benchmark_metadata
from specrag.interfaces.model_comparison_attempts import (
    build_attempt_audit,
    safe_failure_details,
    safe_failure_summary as _safe_failure_summary,
    validate_attempt_history,
    validate_candidate_receipts,
)
from specrag.models import ResearchBudget
from specrag.orchestration import ResearchState
from specrag.orchestration.application import (
    ResearchApplication,
    build_research_application,
)


def candidate_model_config(model_id: str, base: AppConfig | None = None) -> AppConfig:
    """Bind every model-backed Workflow role to one model without fallback."""

    normalized = model_id.strip()
    if not normalized:
        raise ValueError("candidate model ID must not be blank")
    source = base or load_config()
    secrets = source.secrets.model_copy(update={
        "openai_model": normalized,
        "openai_planning_model": normalized,
        "openai_summary_model": normalized,
    })
    llm = source.llm.model_copy(update={"api_fallback_models": ()})
    return source.model_copy(update={"llm": llm, "secrets": secrets})


def benchmark_questions(
    benchmark: EndToEndBenchmark,
    question_ids: Sequence[str],
) -> tuple[BaselineQuestion, ...]:
    """Project Benchmark cases into the exclusive Workflow input boundary."""

    cases = {
        case.task_metadata.task_id: case for case in benchmark.cases
    }
    unknown = tuple(identifier for identifier in question_ids if identifier not in cases)
    if unknown:
        raise ValueError("unknown comparison question IDs: " + ", ".join(unknown))
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("comparison question IDs must be unique")
    return tuple(
        BaselineQuestion(
            question_id=(workflow_input := cases[identifier].workflow_input()).task_id,
            question=workflow_input.question,
        )
        for identifier in question_ids
    )


def run_model_cell(
    *,
    experiment_id: str,
    source_manifest_path: Path,
    requested_model: str,
    questions: tuple[BaselineQuestion, ...],
    output_directory: Path,
    budget: ResearchBudget,
    trial: int = 1,
    release: int = 17,
    resume: bool = False,
    application: ResearchApplication | None = None,
    config: AppConfig | None = None,
    benchmark: EndToEndBenchmark | None = None,
    emit_progress: bool = True,
    max_provider_retries: int = 0,
    retry_sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Execute one model over an ordered case set with fail-closed resume."""

    normalized_experiment = experiment_id.strip()
    normalized_model = requested_model.strip()
    if not normalized_experiment:
        raise ValueError("experiment ID must not be blank")
    if not normalized_model:
        raise ValueError("requested model must not be blank")
    if trial != 1:
        raise ValueError("Model Comparison V1 requires trial=1")
    if max_provider_retries < 0:
        raise ValueError("max_provider_retries must be non-negative")
    if not questions:
        raise ValueError("model comparison cell requires at least one question")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"model comparison manifest does not exist: {source_manifest_path}"
        )

    active_benchmark = benchmark or load_end_to_end_benchmark()
    app = application or build_research_application(
        candidate_model_config(normalized_model, config),
        release=release,
    )
    _validate_environment(app, active_benchmark)
    source_manifest_sha256 = _sha256(source_manifest_path)
    cell_manifest = {
        "schema_version": "specrag.model-comparison-cell/v1",
        "experiment_id": normalized_experiment,
        "source_manifest_sha256": source_manifest_sha256,
        "requested_model": normalized_model,
        "trial": trial,
        "max_provider_retries": max_provider_retries,
        "question_ids": [item.question_id for item in questions],
        "benchmark": benchmark_metadata(app, questions, budget),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    _freeze_json(output_directory / "manifest.json", cell_manifest)

    summaries: list[dict[str, object]] = []
    for question in questions:
        path = output_directory / f"{question.question_id}.json"
        if path.is_file():
            if not resume:
                raise FileExistsError(
                    f"comparison Run already exists; pass --resume: {path}"
                )
            artifact = _load_json_object(path)
            validate_model_cell_artifact(
                artifact,
                experiment_id=normalized_experiment,
                requested_model=normalized_model,
                trial=trial,
                source_manifest_sha256=source_manifest_sha256,
                question=question,
            )
            summary = _artifact_summary(artifact, resumed=True)
            summaries.append(summary)
            if emit_progress:
                print(json.dumps(summary, ensure_ascii=False), flush=True)
            continue

        artifact = _execute_question_with_retries(
            app=app, question=question, budget=budget,
            experiment_id=normalized_experiment, benchmark=active_benchmark,
            requested_model=normalized_model, trial=trial,
            source_manifest_sha256=source_manifest_sha256,
            source_revision=cell_manifest["benchmark"]["source_revision"],
            max_provider_retries=max_provider_retries,
            retry_sleep=retry_sleep,
        )
        _write_json(path, artifact)
        summary = _artifact_summary(artifact, resumed=False)
        summaries.append(summary)
        if emit_progress:
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    _reject_unexpected_artifacts(
        output_directory,
        expected={f"{item.question_id}.json" for item in questions},
    )
    summary_payload = {
        "schema_version": "specrag.model-comparison-cell-summary/v1",
        "experiment_id": normalized_experiment,
        "requested_model": normalized_model,
        "trial": trial,
        "source_manifest_sha256": source_manifest_sha256,
        "questions": summaries,
    }
    _write_json(output_directory / "summary.json", summary_payload)
    return 1 if any(item["workflow_status"] == "failed" for item in summaries) else 0


def _execute_question_with_retries(
    *,
    app: ResearchApplication,
    question: BaselineQuestion,
    budget: ResearchBudget,
    experiment_id: str,
    benchmark: EndToEndBenchmark,
    requested_model: str,
    trial: int,
    source_manifest_sha256: str,
    source_revision: object,
    max_provider_retries: int,
    retry_sleep: Callable[[float], None],
) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    base = {
        "schema_version": "specrag.baseline-run/v1",
        "experiment": _experiment_identity(
            experiment_id=experiment_id, requested_model=requested_model,
            trial=trial, source_manifest_sha256=source_manifest_sha256,
        ),
        "question": _question_identity(question),
        "question_input_sha256": _question_sha256(question),
    }
    for attempt in range(1, max_provider_retries + 2):
        trace = _workflow_trace_metadata(
            experiment_id=experiment_id, benchmark=benchmark,
            requested_model=requested_model, question_id=question.question_id,
            trial=trial, workflow_attempt=attempt, source_revision=source_revision,
            index_fingerprint=app.retrieval_index_fingerprint,
        )
        started = datetime.now(UTC)
        receipts: list[Any] = []
        thread_id = f"{question.question_id}-a{attempt}-{uuid.uuid4().hex}"
        try:
            with capture_agent_receipts() as receipts:
                state = _run_application(
                    app, question=question, thread_id=thread_id,
                    budget=budget, trace=trace,
                )
            receipt_payload = [dict(item) for item in state["model_receipts"]]
            validate_candidate_receipts(receipt_payload, requested_model)
            attempts.append(build_attempt_audit(
                attempt=attempt, status="completed", started=started,
                thread_id=thread_id, trace=trace, receipts=receipt_payload,
                failure=None, previous=attempts,
            ))
            return {
                **base, "workflow_status": "completed",
                "started_at": attempts[0]["started_at"],
                "finished_at": attempts[-1]["finished_at"],
                "thread_id": thread_id, "trace": trace,
                "model_receipts": receipt_payload,
                "workflow_attempts": attempts,
                "state": TypeAdapter(ResearchState).dump_python(state, mode="json"),
            }
        except Exception as exc:  # noqa: BLE001 - bounded provider boundary
            receipt_payload = [item.model_dump(mode="json") for item in receipts]
            failure = safe_failure_details(exc)
            will_retry = bool(
                failure["retryable_provider_failure"]
                and attempt <= max_provider_retries
            )
            retry_delay = min(60.0, 5.0 * (2 ** (attempt - 1)))
            if will_retry:
                failure["retry_delay_seconds"] = retry_delay
            attempts.append(build_attempt_audit(
                attempt=attempt, status="failed", started=started,
                thread_id=thread_id, trace=trace, receipts=receipt_payload,
                failure=failure, previous=attempts,
            ))
            if not will_retry:
                return {
                    **base, "workflow_status": "failed",
                    "started_at": attempts[0]["started_at"],
                    "finished_at": attempts[-1]["finished_at"],
                    "thread_id": thread_id, "trace": trace,
                    "model_receipts": receipt_payload,
                    "workflow_attempts": attempts,
                    "error": _safe_failure_summary(exc),
                }
            retry_sleep(retry_delay)
    raise AssertionError("workflow retry loop did not terminate")


def _run_application(
    app: ResearchApplication,
    *,
    question: BaselineQuestion,
    thread_id: str,
    budget: ResearchBudget,
    trace: Mapping[str, Any],
) -> ResearchState:
    parameters = inspect.signature(app.run).parameters
    extra: dict[str, object] = {}
    if "trace_context" in parameters:
        from specrag.orchestration.parallel import WorkflowTraceContext

        extra["trace_context"] = WorkflowTraceContext(
            run_id=uuid.UUID(str(trace["correlation_id"])),
            run_name=str(trace["run_name"]),
            tags=tuple(str(value) for value in trace["tags"]),
            metadata=dict(trace["metadata"]),
        )
    return app.run(
        research_id=question.question_id,
        question=question.question,
        thread_id=thread_id,
        budget=budget,
        **extra,
    )


def _workflow_trace_metadata(
    *,
    experiment_id: str,
    benchmark: EndToEndBenchmark,
    requested_model: str,
    question_id: str,
    trial: int,
    workflow_attempt: int,
    source_revision: object,
    index_fingerprint: str,
) -> dict[str, object]:
    correlation_id = uuid.uuid4()
    revision = source_revision if isinstance(source_revision, Mapping) else {}
    git_commit = str(revision.get("git_commit") or "unavailable")
    metadata = {
        "experiment_id": experiment_id,
        "benchmark_id": benchmark.manifest.benchmark_id,
        "benchmark_version": benchmark.manifest.benchmark_version,
        "phase": "workflow",
        "candidate_model": requested_model,
        "question_id": question_id,
        "trial": trial,
        "workflow_attempt": workflow_attempt,
        "git_commit": git_commit,
        "index_fingerprint": index_fingerprint,
    }
    return {
        "correlation_id": str(correlation_id),
        "run_name": (
            f"workflow:{requested_model}:{question_id}:trial-{trial}:"
            f"attempt-{workflow_attempt}"
        ),
        "tags": [
            "specrag:model-comparison",
            "phase:workflow",
            f"model:{requested_model}",
            f"question:{question_id}",
            f"attempt:{workflow_attempt}",
        ],
        "metadata": metadata,
    }


def _validate_environment(
    app: ResearchApplication,
    benchmark: EndToEndBenchmark,
) -> None:
    expected = benchmark.manifest.environment
    if app.release != expected.release:
        raise ValueError("live Research application Release differs from Benchmark")
    if app.retrieval_index_fingerprint.lower() != expected.index_fingerprint.lower():
        raise ValueError("live retrieval index fingerprint differs from Benchmark")
    live = {
        (
            item.source_path,
            item.document_sha256.lower(),
            item.specification,
            item.version,
            item.release,
        )
        for item in app.corpus_documents
    }
    frozen = {
        (
            item.source_path,
            item.document_sha256.lower(),
            item.specification,
            item.version,
            item.release,
        )
        for item in expected.documents
    }
    if live != frozen:
        raise ValueError("live corpus document snapshot differs from Benchmark")


def validate_model_cell_artifact(
    artifact: Mapping[str, Any],
    *,
    experiment_id: str,
    requested_model: str,
    trial: int,
    source_manifest_sha256: str,
    question: BaselineQuestion,
) -> None:
    if artifact.get("schema_version") != "specrag.baseline-run/v1":
        raise ValueError("resumed comparison artifact has an unknown schema")
    if artifact.get("experiment") != _experiment_identity(
        experiment_id=experiment_id,
        requested_model=requested_model,
        trial=trial,
        source_manifest_sha256=source_manifest_sha256,
    ):
        raise ValueError("resumed comparison artifact identity changed")
    if artifact.get("question") != _question_identity(question):
        raise ValueError("resumed comparison artifact question changed")
    if artifact.get("question_input_sha256") != _question_sha256(question):
        raise ValueError("resumed comparison artifact input hash changed")
    status = artifact.get("workflow_status")
    if status not in {"completed", "failed"}:
        raise ValueError("resumed comparison artifact has invalid Workflow status")
    receipts = artifact.get("model_receipts")
    if not isinstance(receipts, list) or any(
        not isinstance(item, Mapping) for item in receipts
    ):
        raise ValueError("resumed comparison artifact has invalid model receipts")
    validate_attempt_history(artifact, requested_model)
    validate_candidate_receipts(
        receipts, requested_model, require_nonempty=(status == "completed")
    )
    if status == "completed" and not isinstance(artifact.get("state"), Mapping):
        raise ValueError("completed comparison artifact is missing Workflow state")
    if status == "failed" and not isinstance(artifact.get("error"), str):
        raise ValueError("failed comparison artifact is missing its safe error")


def _artifact_summary(
    artifact: Mapping[str, Any],
    *,
    resumed: bool,
) -> dict[str, object]:
    question = artifact.get("question")
    if not isinstance(question, Mapping):
        raise ValueError("comparison artifact is missing its question identity")
    state = artifact.get("state")
    state_mapping = state if isinstance(state, Mapping) else {}
    result: dict[str, object] = {
        "id": str(question.get("id") or ""),
        "workflow_status": str(artifact.get("workflow_status") or "failed"),
        "resume_status": "resumed" if resumed else "executed",
        "stop_reason": state_mapping.get("stop_reason"),
        "workflow_attempt_count": len(artifact.get("workflow_attempts", ())),
        "workflow_retry_count": max(
            0, len(artifact.get("workflow_attempts", ())) - 1
        ),
    }
    if artifact.get("workflow_status") == "failed":
        error = str(artifact.get("error") or "UnknownError")
        result["error_type"] = error.split(":", 1)[0]
    return result


def _experiment_identity(
    *,
    experiment_id: str,
    requested_model: str,
    trial: int,
    source_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "requested_model": requested_model,
        "trial": trial,
        "source_manifest_sha256": source_manifest_sha256,
    }


def _question_identity(question: BaselineQuestion) -> dict[str, str]:
    return {"id": question.question_id, "text": question.question}


def _question_sha256(question: BaselineQuestion) -> str:
    canonical = json.dumps(
        _question_identity(question),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_json(path: Path, expected: Mapping[str, Any]) -> None:
    if path.is_file():
        actual = _load_json_object(path)
        if actual != expected:
            raise ValueError(f"frozen model comparison file changed: {path}")
        return
    _write_json(path, expected)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reject_unexpected_artifacts(
    directory: Path,
    *,
    expected: set[str],
) -> None:
    actual = {path.name for path in directory.glob("r17-qa-*.json")}
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError("model comparison cell is incomplete (" + "; ".join(details) + ")")


__all__ = [
    "benchmark_questions",
    "candidate_model_config",
    "run_model_cell",
    "validate_model_cell_artifact",
    "validate_candidate_receipts",
]

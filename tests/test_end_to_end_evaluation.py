from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from specrag.evaluation import (
    TaskSuccessEvaluation,
    TaskSuccessReviewDecision,
    TaskSuccessSemanticDraft,
    apply_task_success_review,
    evaluate_task_success,
    extract_efficiency,
    summarize_end_to_end,
)
from specrag.infrastructure.llm import LLMApiMode, LLMCallReceipt
from specrag.infrastructure.structured_agent import (
    CodexAgentError,
    LLMApiStructuredAgent,
    capture_agent_receipts,
)


class _Output(BaseModel):
    value: str


def _artifact(*, input_tokens: int | None = 100) -> dict[str, object]:
    return {
        "schema_version": "specrag.baseline-run/v1",
        "question": {"id": "r17-qa-001", "text": "Question?"},
        "retrieval_scope": {
            "release": 17,
            "documents": [
                {
                    "source_path": "data/3gpp/23.003/v17.11.0/23003-hb0.md",
                    "document_sha256": "A" * 64,
                    "version": "17.11.0",
                    "release": 17,
                },
            ],
        },
        "started_at": "2026-10-18T00:00:00+00:00",
        "finished_at": "2026-10-18T00:00:02+00:00",
        "model_receipts": [
            {
                "model": "test-planner",
                "purpose": "assess independent goals in one 3GPP question",
                "prompt_version": "router/test",
                "latency_ms": 100,
                "input_tokens": input_tokens,
                "output_tokens": 20,
            },
            {
                "model": "test-summary",
                "purpose": "synthesize an evidence-grounded technical answer",
                "prompt_version": "synthesis/test",
                "latency_ms": 200,
                "input_tokens": 300,
                "output_tokens": 40,
            },
        ],
        "state": {
            "stop_reason": "completed",
            "final_answer": "Supported answer [E1]",
            "synthesis": {
                "answer": "Supported answer [E1]",
                "evidence_ids": ["E1"],
            },
            "candidate_evidence": [
                {
                    "evidence_id": "E1",
                    "release": 17,
                    "version": "17.11.0",
                    "source_path": "data/3gpp/23.003/v17.11.0/23003-hb0.md",
                    "document_sha256": "A" * 64,
                },
            ],
            "validated_evidence": [
                {
                    "evidence_id": "E1",
                    "source_valid": True,
                    "validation_status": "supported",
                },
            ],
            "tool_calls": [
                {
                    "task_id": "T1",
                    "tool_name": "hybrid_search",
                    "status": "completed",
                },
                {
                    "task_id": "T1",
                    "tool_name": "expand_context",
                    "status": "completed",
                },
            ],
            "budget": {
                "tool_calls_used_by_task": [["T1", 1]],
                "gap_tool_calls_used_by_task": [["T1", 1]],
            },
        },
    }


def _semantic(**updates: bool) -> TaskSuccessSemanticDraft:
    values = {
        "core_conclusion_answered": True,
        "no_material_error": True,
        "no_material_unsupported_conclusion": True,
        "core_citations_correct": True,
        "rationale": "The core conclusion is correct and supported.",
    }
    values.update(updates)
    return TaskSuccessSemanticDraft.model_validate(values)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_complete_supported_answer_is_proposed_as_success(tmp_path: Path) -> None:
    artifact_path = tmp_path / "r17-qa-001.json"
    _write(artifact_path, _artifact())

    result = evaluate_task_success(
        artifact_path,
        {"id": "r17-qa-001", "release": 17},
        _semantic(),
        judge_prompt_version="judge/test",
        judge_model="test-model",
    )

    assert result.review_status == "proposed"
    assert result.task_success is True
    assert result.failure_reasons == ()
    assert result.efficiency.latency_ms == 2000
    assert result.efficiency.input_tokens == 400
    assert result.efficiency.output_tokens == 60
    assert result.efficiency.total_tokens == 460
    assert result.efficiency.llm_call_count == 2
    assert result.efficiency.tool_call_count == 2
    assert [item.stage for item in result.efficiency.llm_calls] == [
        "router", "synthesizer"
    ]
    assert [item.phase for item in result.efficiency.tool_calls] == [
        "initial", "corrective"
    ]


def test_material_semantic_failure_is_not_task_success(tmp_path: Path) -> None:
    artifact_path = tmp_path / "r17-qa-001.json"
    _write(artifact_path, _artifact())

    result = evaluate_task_success(
        artifact_path,
        {"id": "r17-qa-001", "release": 17},
        _semantic(no_material_unsupported_conclusion=False),
    )

    assert result.task_success is False
    assert result.failure_reasons == ("material_unsupported_claim",)


def test_cited_document_must_belong_to_exact_run_scope(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact["state"]["candidate_evidence"][0]["document_sha256"] = "B" * 64
    artifact_path = tmp_path / "r17-qa-001.json"
    _write(artifact_path, artifact)

    result = evaluate_task_success(
        artifact_path,
        {"id": "r17-qa-001", "release": 17},
        _semantic(),
    )

    assert result.version_scope_compliant is False
    assert result.task_success is False
    assert result.failure_reasons == ("version_scope_violation",)


def test_missing_token_usage_is_explicit_not_zero() -> None:
    result = extract_efficiency(
        _artifact(input_tokens=None),
        question_id="r17-qa-001",
    )

    assert result.receipt_scope == "full_workflow"
    assert result.token_telemetry_complete is False
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.total_tokens is None


def test_retried_call_does_not_claim_complete_token_telemetry() -> None:
    artifact = _artifact()
    artifact["model_receipts"][0]["attempt_count"] = 2
    artifact["model_receipts"][0]["attempted_models"] = [
        "test-planner", "test-planner"
    ]

    result = extract_efficiency(artifact, question_id="r17-qa-001")

    assert result.llm_calls[0].total_tokens == 120
    assert result.llm_calls[0].attempt_count == 2
    assert result.token_telemetry_complete is False
    assert result.total_tokens is None


def test_summary_does_not_report_missing_tokens_as_zero(tmp_path: Path) -> None:
    artifact_path = tmp_path / "r17-qa-001.json"
    _write(artifact_path, _artifact(input_tokens=None))
    proposed = evaluate_task_success(
        artifact_path,
        {"id": "r17-qa-001", "release": 17},
        _semantic(),
    )
    reviewed = TaskSuccessEvaluation.model_validate({
        **proposed.model_dump(mode="json"),
        "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-10-18T08:00:00+08:00",
    })

    summary = summarize_end_to_end((reviewed,))

    assert summary["telemetry_complete_count"] == 0
    assert summary["efficiency"]["total_tokens"] == {
        "count": 0,
        "sum": None,
        "mean": None,
        "p95": None,
    }


def test_one_incomplete_case_nulls_whole_batch_token_aggregate(
    tmp_path: Path,
) -> None:
    evaluations = []
    for question_id, input_tokens in (
        ("r17-qa-001", 100),
        ("r17-qa-002", None),
    ):
        artifact_path = tmp_path / f"{question_id}.json"
        artifact = _artifact(input_tokens=input_tokens)
        artifact["question"]["id"] = question_id
        _write(artifact_path, artifact)
        proposed = evaluate_task_success(
            artifact_path,
            {"id": question_id, "release": 17},
            _semantic(),
        )
        evaluations.append(TaskSuccessEvaluation.model_validate({
            **proposed.model_dump(mode="json"),
            "review_status": "reviewed",
            "reviewed_by": "test-reviewer",
            "reviewed_at": "2026-10-18T08:00:00+08:00",
        }))

    summary = summarize_end_to_end(tuple(evaluations))

    assert summary["telemetry_complete_count"] == 1
    assert summary["efficiency"]["total_tokens"] == {
        "count": 1,
        "sum": None,
        "mean": None,
        "p95": None,
    }


def test_run_artifact_must_match_qa_identity(tmp_path: Path) -> None:
    artifact_path = tmp_path / "r17-qa-002.json"
    _write(artifact_path, _artifact())

    with pytest.raises(ValueError, match="question ID"):
        evaluate_task_success(
            artifact_path,
            {"id": "r17-qa-002", "question": "Different?", "release": 17},
            _semantic(),
        )


def test_summary_requires_review_and_uses_all_reviewed_cases(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "r17-qa-001.json"
    _write(artifact_path, _artifact())
    proposed = evaluate_task_success(
        artifact_path,
        {"id": "r17-qa-001", "release": 17},
        _semantic(),
    )
    with pytest.raises(ValueError, match="requires reviewed judgments"):
        summarize_end_to_end((proposed,))

    reviewed = TaskSuccessEvaluation.model_validate({
        **proposed.model_dump(mode="json"),
        "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-10-18T08:00:00+08:00",
    })
    summary = summarize_end_to_end((reviewed,))

    assert summary["question_count"] == 1
    assert summary["reviewed_count"] == 1
    assert summary["task_success_rate"] == 1.0
    assert summary["efficiency"]["total_tokens"] == {
        "count": 1,
        "sum": 460.0,
        "mean": 460.0,
        "p95": 460.0,
    }


def test_api_agent_capture_records_completed_calls_without_prompt_content() -> None:
    class Client:
        def parse(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                output=_Output(value="ok"),
                receipt=LLMCallReceipt(
                    model="test-model",
                    api_mode=LLMApiMode.RESPONSES,
                    purpose="test purpose",
                    prompt_version="test/v1",
                    request_sha256="a" * 64,
                    response_sha256="b" * 64,
                    input_tokens=11,
                    output_tokens=3,
                    latency_ms=1,
                ),
            )

    agent = object.__new__(LLMApiStructuredAgent)
    agent._model = "test-model"
    agent._reasoning_effort = "high"
    agent._client = Client()

    with capture_agent_receipts() as receipts:
        response = agent.parse(
            purpose="test purpose",
            prompt_version="test/v1",
            instructions="Do not persist this prompt.",
            input_payload={"secret": "not retained"},
            output_type=_Output,
        )

    assert response.output.value == "ok"
    assert len(receipts) == 1
    assert receipts[0].input_tokens == 11
    assert receipts[0].output_tokens == 3
    assert "secret" not in receipts[0].model_dump_json()


def test_human_review_can_change_only_semantic_gates(tmp_path: Path) -> None:
    artifact_path = tmp_path / "r17-qa-001.json"
    _write(artifact_path, _artifact())
    proposed = evaluate_task_success(
        artifact_path,
        {"id": "r17-qa-001", "release": 17},
        _semantic(core_conclusion_answered=False),
        judge_prompt_version="judge/test",
        judge_model="judge-model",
    )
    decision = TaskSuccessReviewDecision(
        question_id="r17-qa-001",
        source_artifact_sha256=proposed.source_artifact_sha256,
        core_conclusion_answered=True,
        no_material_error=True,
        no_material_unsupported_conclusion=True,
        core_citations_correct=True,
        rationale="Manual review confirms a useful, supported answer.",
        reviewed_by="project-owner-delegated:codex-manual-audit",
        reviewed_at="2026-10-18T18:00:00+08:00",
    )

    reviewed = apply_task_success_review(proposed, decision)

    assert reviewed.review_status == "reviewed"
    assert reviewed.task_success is True
    assert reviewed.failure_reasons == ()
    assert reviewed.rationale == decision.rationale
    assert reviewed.judge_rationale == proposed.judge_rationale
    assert reviewed.efficiency == proposed.efficiency
    assert reviewed.citation_integrity == proposed.citation_integrity
    assert reviewed.version_scope_compliant == proposed.version_scope_compliant


def test_human_review_cannot_override_invalid_citation_integrity(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    artifact["state"]["validated_evidence"] = []
    artifact_path = tmp_path / "r17-qa-001.json"
    _write(artifact_path, artifact)
    proposed = evaluate_task_success(
        artifact_path,
        {"id": "r17-qa-001", "release": 17},
        _semantic(),
    )
    decision = TaskSuccessReviewDecision(
        question_id="r17-qa-001",
        source_artifact_sha256=proposed.source_artifact_sha256,
        core_conclusion_answered=True,
        no_material_error=True,
        no_material_unsupported_conclusion=True,
        core_citations_correct=True,
        rationale="The prose is correct, but deterministic citation checks remain binding.",
        reviewed_by="reviewer",
        reviewed_at="2026-10-18T18:00:00+08:00",
    )

    reviewed = apply_task_success_review(proposed, decision)

    assert reviewed.citation_integrity is False
    assert reviewed.core_citations_correct is False
    assert reviewed.task_success is False
    assert reviewed.failure_reasons == ("incorrect_core_citation",)


def test_api_agent_capture_records_failed_logical_call() -> None:
    class ProviderFailure(RuntimeError):
        failure_kind = "rate_limited"
        retryable = True
        status_code = 429
        attempt_count = 2

    class Client:
        def parse(self, **_: object) -> object:
            raise ProviderFailure("provider detail must not be persisted")

    agent = object.__new__(LLMApiStructuredAgent)
    agent._model = "test-model"
    agent._reasoning_effort = "high"
    agent._client = Client()

    with capture_agent_receipts() as receipts:
        with pytest.raises(CodexAgentError):
            agent.parse(
                purpose="test failed purpose",
                prompt_version="test/v1",
                instructions="Test.",
                input_payload={"private": "payload"},
                output_type=_Output,
            )

    assert len(receipts) == 1
    assert receipts[0].outcome == "failed"
    assert receipts[0].failure_kind == "rate_limited"
    assert receipts[0].attempt_count == 2
    assert receipts[0].input_tokens is None
    assert "provider detail" not in receipts[0].model_dump_json()
    assert "payload" not in receipts[0].model_dump_json()

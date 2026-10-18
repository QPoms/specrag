"""Workflow efficiency records extracted from complete Run artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


ToolPhase = Literal["initial", "corrective", "unknown"]
ReceiptScope = Literal["full_workflow", "legacy_state_only", "none"]
CallOutcome = Literal["completed", "failed"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMCallEfficiency(_StrictModel):
    sequence: int = Field(ge=1)
    stage: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    latency_ms: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    attempt_count: int = Field(default=1, ge=1)
    attempted_models: tuple[str, ...] = ()
    outcome: CallOutcome = "completed"
    failure_kind: str | None = None

    @model_validator(mode="after")
    def _coherent_total(self) -> "LLMCallEfficiency":
        expected = (
            self.input_tokens + self.output_tokens
            if self.input_tokens is not None and self.output_tokens is not None
            else None
        )
        if self.total_tokens != expected:
            raise ValueError("LLM call total_tokens must match input plus output")
        if self.outcome == "completed" and self.failure_kind is not None:
            raise ValueError("completed LLM call cannot contain failure kind")
        if self.outcome == "failed" and not self.failure_kind:
            raise ValueError("failed LLM call requires failure kind")
        return self


class ToolCallEfficiency(_StrictModel):
    sequence: int = Field(ge=1)
    task_id: str | None = None
    phase: ToolPhase = "unknown"
    tool_name: str = Field(min_length=1)
    status: str = Field(min_length=1)


class EfficiencyEvaluation(_StrictModel):
    schema_version: Literal["specrag.efficiency-evaluation/v1"] = (
        "specrag.efficiency-evaluation/v1"
    )
    question_id: str = Field(min_length=1)
    latency_ms: float = Field(ge=0)
    receipt_scope: ReceiptScope
    token_telemetry_complete: bool
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    llm_call_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    llm_calls: tuple[LLMCallEfficiency, ...] = ()
    tool_calls: tuple[ToolCallEfficiency, ...] = ()

    @model_validator(mode="after")
    def _coherent_aggregates(self) -> "EfficiencyEvaluation":
        if self.llm_call_count != len(self.llm_calls):
            raise ValueError("LLM call count must match LLM call records")
        if self.tool_call_count != len(self.tool_calls):
            raise ValueError("tool call count must match tool call records")
        complete = bool(self.llm_calls) and all(
            item.total_tokens is not None and item.attempt_count == 1
            for item in self.llm_calls
        )
        if self.token_telemetry_complete != complete:
            raise ValueError("token telemetry completeness is inconsistent")
        expected_input = (
            sum(item.input_tokens or 0 for item in self.llm_calls)
            if complete else None
        )
        expected_output = (
            sum(item.output_tokens or 0 for item in self.llm_calls)
            if complete else None
        )
        expected_total = (
            expected_input + expected_output
            if expected_input is not None and expected_output is not None
            else None
        )
        if (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
        ) != (expected_input, expected_output, expected_total):
            raise ValueError("aggregate token usage must match per-call records")
        return self


def _stage_for_purpose(purpose: str) -> str:
    normalized = purpose.casefold()
    if "independent goals" in normalized:
        return "router"
    if "decompose" in normalized:
        return "planner"
    if "retrieval tool calls" in normalized:
        return "researcher"
    if "sufficiency" in normalized:
        return "sufficiency_judge"
    if "synthesize" in normalized:
        return "synthesizer"
    return "unknown"


def _parse_time(value: object) -> datetime:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.utcoffset() is None:
        raise ValueError("artifact timestamps must include a timezone")
    return parsed


def _phase_by_call(
    calls: tuple[Mapping[str, Any], ...],
    budget: Mapping[str, Any],
) -> tuple[ToolPhase, ...]:
    initial = {
        str(key): int(value)
        for key, value in budget.get("tool_calls_used_by_task", ())
    }
    corrective = {
        str(key): int(value)
        for key, value in budget.get("gap_tool_calls_used_by_task", ())
    }
    seen: dict[str, int] = {}
    phases: list[ToolPhase] = []
    for call in calls:
        task_id = str(call.get("task_id", ""))
        seen[task_id] = seen.get(task_id, 0) + 1
        index = seen[task_id]
        if index <= initial.get(task_id, 0):
            phases.append("initial")
        elif index <= initial.get(task_id, 0) + corrective.get(task_id, 0):
            phases.append("corrective")
        else:
            phases.append("unknown")
    return tuple(phases)


def extract_efficiency(
    artifact: Mapping[str, Any],
    *,
    question_id: str,
) -> EfficiencyEvaluation:
    """Extract complete-workflow efficiency without treating missing usage as zero."""

    started = _parse_time(artifact["started_at"])
    finished = _parse_time(artifact["finished_at"])
    latency_ms = max(0.0, (finished - started).total_seconds() * 1_000)
    state_value = artifact.get("state", {})
    state = state_value if isinstance(state_value, Mapping) else {}

    top_level_receipts = artifact.get("model_receipts")
    if isinstance(top_level_receipts, list):
        raw_receipts = tuple(
            item for item in top_level_receipts if isinstance(item, Mapping)
        )
        receipt_scope: ReceiptScope = "full_workflow"
    else:
        legacy = state.get("model_receipts", ())
        raw_receipts = tuple(
            item for item in legacy if isinstance(item, Mapping)
        )
        receipt_scope = "legacy_state_only" if raw_receipts else "none"

    llm_calls: list[LLMCallEfficiency] = []
    for index, receipt in enumerate(raw_receipts, start=1):
        raw_input = receipt.get("input_tokens")
        raw_output = receipt.get("output_tokens")
        input_tokens = raw_input if isinstance(raw_input, int) else None
        output_tokens = raw_output if isinstance(raw_output, int) else None
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        purpose = str(receipt.get("purpose") or "unknown")
        llm_calls.append(LLMCallEfficiency(
            sequence=index,
            stage=_stage_for_purpose(purpose),
            purpose=purpose,
            model=str(receipt.get("model") or "unknown"),
            prompt_version=str(receipt.get("prompt_version") or "unknown"),
            latency_ms=max(0, int(receipt.get("latency_ms") or 0)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            attempt_count=max(1, int(receipt.get("attempt_count") or 1)),
            attempted_models=tuple(
                str(value) for value in receipt.get("attempted_models", ())
            ),
            outcome=str(receipt.get("outcome") or "completed"),
            failure_kind=(
                str(receipt["failure_kind"])
                if receipt.get("failure_kind") else None
            ),
        ))

    raw_tool_calls = tuple(
        item for item in state.get("tool_calls", ()) if isinstance(item, Mapping)
    )
    budget_value = state.get("budget", {})
    budget = budget_value if isinstance(budget_value, Mapping) else {}
    phases = _phase_by_call(raw_tool_calls, budget)
    tool_calls = tuple(
        ToolCallEfficiency(
            sequence=index,
            task_id=(str(call["task_id"]) if call.get("task_id") else None),
            phase=phases[index - 1],
            tool_name=str(call.get("tool_name") or "unknown"),
            status=str(call.get("status") or "unknown"),
        )
        for index, call in enumerate(raw_tool_calls, start=1)
    )
    complete = bool(llm_calls) and all(
        item.total_tokens is not None and item.attempt_count == 1
        for item in llm_calls
    )
    input_total = sum(item.input_tokens or 0 for item in llm_calls) if complete else None
    output_total = sum(item.output_tokens or 0 for item in llm_calls) if complete else None
    total = (
        input_total + output_total
        if input_total is not None and output_total is not None else None
    )
    return EfficiencyEvaluation(
        question_id=question_id,
        latency_ms=round(latency_ms, 3),
        receipt_scope=receipt_scope,
        token_telemetry_complete=complete,
        input_tokens=input_total,
        output_tokens=output_total,
        total_tokens=total,
        llm_call_count=len(llm_calls),
        tool_call_count=len(tool_calls),
        llm_calls=tuple(llm_calls),
        tool_calls=tool_calls,
    )


__all__ = [
    "EfficiencyEvaluation",
    "LLMCallEfficiency",
    "ToolCallEfficiency",
    "extract_efficiency",
]

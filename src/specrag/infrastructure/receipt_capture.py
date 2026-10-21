"""Provider-neutral model-call receipt capture for bounded executions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
from typing import Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CodexAgentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_runtime: str = "codex_cli"
    codex_version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reasoning_effort: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    latency_ms: int = Field(ge=0)
    thread_id_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    input_ceiling_tokens: int | None = Field(default=None, ge=0)
    attempt_count: int = Field(default=1, ge=1)
    attempted_models: tuple[str, ...] = ()
    outcome: Literal["completed", "failed"] = "completed"
    failure_kind: str | None = None
    sandbox: str = "read-only"
    ephemeral: bool = True

    @model_validator(mode="after")
    def _coherent_outcome(self) -> "CodexAgentReceipt":
        if self.outcome == "completed" and self.failure_kind is not None:
            raise ValueError("completed receipt cannot contain a failure kind")
        if self.outcome == "failed" and not self.failure_kind:
            raise ValueError("failed receipt requires a failure kind")
        return self


_RECEIPT_CAPTURE: ContextVar[list[CodexAgentReceipt] | None] = ContextVar(
    "specrag_agent_receipt_capture",
    default=None,
)


@contextmanager
def capture_agent_receipts() -> Iterator[list[CodexAgentReceipt]]:
    """Capture completed or failed logical agent calls in the current context."""

    receipts: list[CodexAgentReceipt] = []
    token = _RECEIPT_CAPTURE.set(receipts)
    try:
        yield receipts
    finally:
        _RECEIPT_CAPTURE.reset(token)


def record_agent_receipt(receipt: CodexAgentReceipt) -> None:
    capture = _RECEIPT_CAPTURE.get()
    if capture is not None:
        capture.append(receipt)


def record_failed_agent_call(
    *,
    model: str,
    reasoning_effort: str,
    purpose: str,
    prompt_version: str,
    request_sha256: str,
    latency_ms: int,
    error: Exception,
) -> None:
    """Record one failed logical model call without persisting error text."""

    failure_kind = str(getattr(error, "failure_kind", "unknown"))
    attempt_count = max(1, int(getattr(error, "attempt_count", 1) or 1))
    budget = getattr(error, "budget", None)
    record_agent_receipt(CodexAgentReceipt(
        agent_runtime="llm_api",
        codex_version="api",
        model=model,
        reasoning_effort=reasoning_effort,
        purpose=purpose,
        prompt_version=prompt_version,
        request_sha256=request_sha256,
        response_sha256=hashlib.sha256(
            failure_kind.encode("utf-8")
        ).hexdigest(),
        latency_ms=max(0, latency_ms),
        attempt_count=attempt_count,
        attempted_models=tuple(model for _ in range(attempt_count)),
        estimated_input_tokens=getattr(budget, "estimated_input_tokens", None),
        input_ceiling_tokens=getattr(budget, "input_ceiling_tokens", None),
        outcome="failed",
        failure_kind=failure_kind,
        sandbox="api",
    ))


__all__ = [
    "CodexAgentReceipt",
    "capture_agent_receipts",
    "record_agent_receipt",
    "record_failed_agent_call",
]

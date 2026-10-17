"""Sanitized failures crossing the infrastructure agent boundary."""

from __future__ import annotations

from specrag.models import RUN_FAILURE_KINDS


class CodexAgentError(RuntimeError):
    """Sanitized failure raised by the isolated Codex adapter."""

    def __init__(
        self, message: str, *, failure_kind: str = "internal_error",
        retryable: bool = False, status_code: int | None = None,
        attempt_count: int | None = None,
    ) -> None:
        super().__init__(message)
        if failure_kind not in RUN_FAILURE_KINDS:
            failure_kind = "unknown"
        self.failure_kind = failure_kind
        self.retryable = retryable
        self.status_code = status_code
        self.attempt_count = attempt_count


__all__ = ["CodexAgentError"]

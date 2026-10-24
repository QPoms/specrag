"""Sanitized failures crossing the infrastructure agent boundary."""

from __future__ import annotations

from typing import Any

from specrag.models import RUN_FAILURE_KINDS


class CodexAgentError(RuntimeError):
    """Sanitized failure raised by the isolated Codex adapter."""

    def __init__(
        self, message: str, *, failure_kind: str = "internal_error",
        retryable: bool = False, status_code: int | None = None,
        attempt_count: int | None = None,
        public_message: str | None = None,
        capability: str | None = None,
        provider_code: str | None = None,
        provider_parameter: str | None = None,
        api_mode: str | None = None,
    ) -> None:
        super().__init__(message)
        if failure_kind not in RUN_FAILURE_KINDS:
            failure_kind = "unknown"
        self.failure_kind = failure_kind
        self.retryable = retryable
        self.status_code = status_code
        self.attempt_count = attempt_count
        self.public_message = public_message or "The model request failed."
        self.capability = capability
        self.provider_code = provider_code
        self.provider_parameter = provider_parameter
        self.api_mode = api_mode


def public_error_fields(exc: Exception) -> dict[str, Any]:
    """Copy only the sanitized provider fields across the agent boundary."""

    return {
        "public_message": getattr(exc, "public_message", None),
        "capability": getattr(exc, "capability", None),
        "provider_code": getattr(exc, "provider_code", None),
        "provider_parameter": getattr(exc, "provider_parameter", None),
        "api_mode": getattr(exc, "api_mode", None),
    }


__all__ = ["CodexAgentError", "public_error_fields"]

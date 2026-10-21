"""Safe classification for outbound model-provider failures."""

from __future__ import annotations

from dataclasses import dataclass

from specrag.models import RUN_FAILURE_KINDS


class StructuredLLMError(RuntimeError):
    """Sanitized structured API failure with safe retry metadata."""

    def __init__(
        self, message: str, *, failure_kind: str = "structured_output",
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


@dataclass(frozen=True)
class ProviderFailure:
    kind: str
    retryable: bool
    status_code: int | None = None


def classify_provider_error(exc: Exception) -> ProviderFailure:
    """Classify an SDK/network error without retaining provider payloads."""

    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = None
    name = type(exc).__name__
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if code is None and isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = error.get("code")
    if code in {
        "context_length_exceeded",
        "context_window_exceeded",
        "maximum_context_length_exceeded",
        "input_too_long",
    } or name in {"ContextLengthExceededError", "ContextWindowExceededError"}:
        return ProviderFailure("context_overflow", False, status_code)
    if status_code == 429 or name == "RateLimitError":
        return ProviderFailure("rate_limited", True, status_code)
    if status_code == 408 or name in {
        "APITimeoutError", "ConnectTimeout", "ReadTimeout", "TimeoutError",
    }:
        return ProviderFailure("timeout", True, status_code)
    if status_code in {409, 425}:
        return ProviderFailure("retryable_transient", True, status_code)
    if status_code is not None and status_code >= 500:
        return ProviderFailure("provider_unavailable", True, status_code)
    if name in {"APIConnectionError", "ConnectError"}:
        return ProviderFailure("retryable_transient", True, status_code)
    if status_code in {401, 403}:
        return ProviderFailure("authentication_or_configuration", False, status_code)
    if status_code in {400, 404, 422}:
        return ProviderFailure("invalid_request", False, status_code)
    return ProviderFailure("unknown", False, status_code)


def is_retryable_provider_error(exc: Exception) -> bool:
    return classify_provider_error(exc).retryable


__all__ = [
    "ProviderFailure", "StructuredLLMError", "classify_provider_error",
    "is_retryable_provider_error",
]

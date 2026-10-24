"""Safe classification for outbound model-provider failures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from specrag.models import RUN_FAILURE_KINDS


class StructuredLLMError(RuntimeError):
    """Sanitized structured API failure with safe retry metadata."""

    def __init__(
        self, message: str, *, failure_kind: str = "structured_output",
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


@dataclass(frozen=True)
class ProviderFailure:
    kind: str
    retryable: bool
    status_code: int | None = None


_SAFE_PROVIDER_FIELD = re.compile(r"^[A-Za-z0-9_.\[\]-]{1,120}$")


def safe_provider_field(exc: Exception, name: str) -> str | None:
    """Read one allowlisted provider identifier without retaining its body."""

    value = getattr(exc, name, None)
    body = getattr(exc, "body", None)
    if value is None and isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            value = error.get(name)
    if isinstance(value, str) and _SAFE_PROVIDER_FIELD.fullmatch(value):
        return value
    return None


def provider_public_message(
    failure: ProviderFailure,
    *,
    operation: str,
    api_mode: str,
    provider_code: str | None = None,
    provider_parameter: str | None = None,
) -> str:
    """Create an actionable diagnostic containing no provider prose or secrets."""

    status = f"HTTP {failure.status_code}" if failure.status_code else "no HTTP status"
    metadata = [status, f"mode={api_mode}"]
    if provider_code:
        metadata.append(f"code={provider_code}")
    if provider_parameter:
        metadata.append(f"parameter={provider_parameter}")
    suffix = "; ".join(metadata)
    if failure.kind == "invalid_request" and operation == "native_tool_calling":
        return (
            "The model gateway rejected the native tool-calling request "
            f"({suffix}). Verify that the configured model supports this API mode "
            "and strict function tools."
        )
    if failure.kind == "invalid_request":
        return f"The model gateway rejected the structured request ({suffix})."
    if failure.kind == "authentication_or_configuration":
        return f"The model gateway rejected the credentials or endpoint ({suffix})."
    if failure.kind == "rate_limited":
        return f"The model gateway rate-limited the request ({suffix}); retry later."
    if failure.kind in {"retryable_transient", "provider_unavailable", "timeout"}:
        return f"The model gateway is temporarily unavailable ({suffix}); retry later."
    if failure.kind == "context_overflow":
        return f"The model request exceeded the configured context limit ({suffix})."
    return f"The model request failed ({suffix}; kind={failure.kind})."


def safe_structured_llm_error(
    exc: Exception,
    *,
    attempt_count: int | None = None,
    exhausted_models: int | None = None,
    operation: str = "structured_output",
    api_mode: str = "responses",
) -> StructuredLLMError:
    """Project one provider exception into internal and public safe details."""

    failure = classify_provider_error(exc)
    provider_code = safe_provider_field(exc, "code")
    provider_parameter = safe_provider_field(exc, "param")
    details = [type(exc).__name__]
    request_id = getattr(exc, "request_id", None)
    for label, value in (
        ("status", getattr(exc, "status_code", None)),
        ("code", provider_code),
        ("param", provider_parameter),
    ):
        if isinstance(value, (str, int)) and str(value):
            details.append(f"{label}={str(value)[:80]}")
    if isinstance(request_id, str) and request_id:
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        details.append(f"request_id_sha256={digest[:16]}")
    if attempt_count is not None:
        details.append(f"attempts={attempt_count}")
    if exhausted_models is not None:
        details.append(f"models={exhausted_models}")
    return StructuredLLMError(
        "structured LLM request failed (" + ", ".join(details) + ")",
        failure_kind=failure.kind,
        retryable=failure.retryable,
        status_code=failure.status_code,
        attempt_count=attempt_count,
        public_message=provider_public_message(
            failure,
            operation=operation,
            api_mode=api_mode,
            provider_code=provider_code,
            provider_parameter=provider_parameter,
        ),
        capability=operation,
        provider_code=provider_code,
        provider_parameter=provider_parameter,
        api_mode=api_mode,
    )


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
    "is_retryable_provider_error", "provider_public_message",
    "safe_provider_field", "safe_structured_llm_error",
]

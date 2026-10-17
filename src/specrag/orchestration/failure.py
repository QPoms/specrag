"""Safe failure metadata extracted at the Run boundary."""

from __future__ import annotations

from specrag.models import RUN_FAILURE_KINDS


def failure_metadata(exc: Exception) -> dict[str, object]:
    """Return bounded, machine-readable metadata without exception details."""

    kind = getattr(exc, "failure_kind", "internal_error")
    if kind not in RUN_FAILURE_KINDS:
        kind = "unknown"
    attempts = getattr(exc, "attempt_count", None)
    if not isinstance(attempts, int) or attempts < 1:
        attempts = None
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int) or not 100 <= status_code <= 599:
        status_code = None
    return {
        "failure_kind": kind,
        "failure_retryable": bool(getattr(exc, "retryable", False)),
        "failure_attempt_count": attempts,
        "failure_status_code": status_code,
    }


__all__ = ["failure_metadata"]

"""Auditable Workflow-attempt history for model-comparison Runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from typing import Any


_RETRYABLE_PROVIDER_FAILURES = frozenset({
    "provider_unavailable", "rate_limited", "retryable_transient", "timeout",
})


def validate_candidate_receipts(
    receipts: Sequence[Mapping[str, Any]],
    requested_model: str,
    *,
    require_nonempty: bool = True,
) -> None:
    """Require auditable same-model provenance for every captured call."""

    if require_nonempty and not receipts:
        raise ValueError("completed candidate Workflow requires model receipts")
    for index, receipt in enumerate(receipts, start=1):
        observed = str(receipt.get("model") or "")
        attempted_value = receipt.get("attempted_models")
        if not isinstance(attempted_value, (list, tuple)) or not attempted_value:
            raise ValueError(
                f"candidate receipt {index} requires attempted model provenance"
            )
        attempted = tuple(str(value) for value in attempted_value)
        if observed != requested_model:
            raise ValueError(
                f"candidate receipt {index} used {observed or 'unknown'}, "
                f"expected {requested_model}"
            )
        if any(model != requested_model for model in attempted):
            raise ValueError(
                f"candidate receipt {index} contains cross-model fallback attempts"
            )


def build_attempt_audit(
    *, attempt: int, status: str, started: datetime, thread_id: str,
    trace: Mapping[str, Any], receipts: list[dict[str, Any]],
    failure: Mapping[str, object] | None, previous: list[dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "attempt": attempt, "status": status,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "thread_id": thread_id, "trace": dict(trace),
        "model_receipts": receipts,
        "failure": dict(failure) if failure is not None else None,
        "previous_attempt_sha256": (
            str(previous[-1]["attempt_sha256"]) if previous else None
        ),
    }
    payload["attempt_sha256"] = canonical_sha256(payload)
    return payload


def safe_failure_summary(exc: Exception) -> str:
    """Persist only stable provider taxonomy and an allowlisted explanation."""

    failure_kind = str(getattr(exc, "failure_kind", "unknown"))[:200]
    public_message = getattr(exc, "public_message", None)
    message = (
        public_message.strip()[:1000]
        if isinstance(public_message, str) and public_message.strip()
        else "The comparison Workflow failed."
    )
    return f"{type(exc).__name__}:{failure_kind}: {message}"


def safe_failure_details(exc: Exception) -> dict[str, object]:
    failure_kind = str(getattr(exc, "failure_kind", "unknown"))[:200]
    return {
        "error_type": type(exc).__name__[:200],
        "failure_kind": failure_kind,
        "retryable": bool(getattr(exc, "retryable", False)),
        "retryable_provider_failure": bool(
            getattr(exc, "retryable", False)
            and failure_kind in _RETRYABLE_PROVIDER_FAILURES
        ),
        "summary": safe_failure_summary(exc),
    }


def validate_attempt_history(
    artifact: Mapping[str, Any], requested_model: str,
) -> None:
    attempts = artifact.get("workflow_attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("comparison artifact requires Workflow attempt history")
    previous: str | None = None
    for expected, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping) or attempt.get("attempt") != expected:
            raise ValueError("Workflow attempt history is not ordered")
        supplied_hash = attempt.get("attempt_sha256")
        payload = dict(attempt)
        payload.pop("attempt_sha256", None)
        if attempt.get("previous_attempt_sha256") != previous:
            raise ValueError("Workflow attempt hash chain is broken")
        if supplied_hash != canonical_sha256(payload):
            raise ValueError("Workflow attempt hash changed")
        attempt_receipts = attempt.get("model_receipts")
        if not isinstance(attempt_receipts, list) or any(
            not isinstance(item, Mapping) for item in attempt_receipts
        ):
            raise ValueError("Workflow attempt receipts are invalid")
        validate_candidate_receipts(
            attempt_receipts, requested_model,
            require_nonempty=(attempt.get("status") == "completed"),
        )
        if expected < len(attempts):
            failure = attempt.get("failure")
            if (
                attempt.get("status") != "failed"
                or not isinstance(failure, Mapping)
                or failure.get("retryable_provider_failure") is not True
            ):
                raise ValueError("only retryable provider failures may be retried")
        previous = str(supplied_hash)
    if attempts[-1].get("status") != artifact.get("workflow_status"):
        raise ValueError("final Workflow attempt disagrees with artifact status")


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "build_attempt_audit",
    "canonical_sha256",
    "safe_failure_details",
    "safe_failure_summary",
    "validate_attempt_history",
    "validate_candidate_receipts",
]

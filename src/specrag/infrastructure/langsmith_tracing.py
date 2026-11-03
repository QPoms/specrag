"""Optional, secret-safe LangSmith activation for research workflows.

Durable Run events and model receipts remain authoritative.  This adapter
only activates the official LangSmith tracing context and OpenAI wrapper;
disabled or incomplete configuration is a true no-op.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import re
import time
from types import MappingProxyType
from typing import Any, Callable, ContextManager, Iterator, Literal, Mapping, Sequence
from uuid import uuid4

from specrag.config import AppConfig


RuntimeLoader = Callable[
    [], tuple[Callable[..., Any], Callable[..., ContextManager[Any]]]
]
TraceMetadataValue = str | int | float | bool

_SAFE_METADATA_KEYS = frozenset({
    "benchmark_id",
    "benchmark_version",
    "candidate_model",
    "experiment_id",
    "git_commit",
    "index_fingerprint",
    "local_run_id",
    "phase",
    "question_id",
    "resume",
    "thread_id",
    "trial",
})


@dataclass(frozen=True)
class LangSmithPreflightResult:
    """Secret-free readiness of the configured LangSmith workspace."""

    status: Literal["disabled", "ready", "unavailable"]
    failure_kind: Literal[
        "authentication_or_configuration",
        "provider_unavailable",
        "timeout",
        "unknown",
    ] | None = None
    retryable: bool = False


class LangSmithPreflightError(RuntimeError):
    """Sanitized failure raised before any formal model call begins."""

    def __init__(self, result: LangSmithPreflightResult) -> None:
        super().__init__(
            "LangSmith preflight failed "
            f"(failure_kind={result.failure_kind or 'unknown'})"
        )
        self.failure_kind = result.failure_kind or "unknown"
        self.retryable = result.retryable


def freeze_trace_metadata(
    metadata: Mapping[str, TraceMetadataValue] | None,
) -> Mapping[str, TraceMetadataValue]:
    """Validate and freeze the small, explicit trace-correlation projection."""

    values = dict(metadata or {})
    unknown = tuple(sorted(set(values) - _SAFE_METADATA_KEYS))
    if unknown:
        raise ValueError("unsupported trace metadata keys: " + ", ".join(unknown))
    for key, value in values.items():
        if type(value) not in {str, int, float, bool}:  # noqa: E721
            raise ValueError(f"trace metadata {key!r} must be a scalar")
        if isinstance(value, str) and (
            not value.strip()
            or value != value.strip()
            or len(value) > 512
            or not value.isprintable()
        ):
            raise ValueError(f"trace metadata {key!r} has an unsafe string value")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"trace metadata {key!r} must be finite")
    return MappingProxyType(values)


def freeze_trace_tags(tags: Sequence[str]) -> tuple[str, ...]:
    """Validate immutable searchable labels without accepting control text."""

    frozen = tuple(tags)
    if any(
        not tag.strip()
        or tag != tag.strip()
        or len(tag) > 256
        or not tag.isprintable()
        for tag in frozen
    ):
        raise ValueError("trace tags must be printable, trimmed, and non-blank")
    if len(frozen) != len(set(frozen)):
        raise ValueError("trace tags must be unique")
    return frozen


@dataclass(frozen=True)
class LangSmithTracing:
    """Resolved tracing activation without credential-bearing diagnostics."""

    enabled: bool
    project_name: str
    _api_key: str = field(default="", repr=False)
    _endpoint: str = field(default="", repr=False)
    _workspace_id: str = field(default="", repr=False)
    _runtime_loader: RuntimeLoader = field(
        default=lambda: _load_runtime(), repr=False, compare=False
    )

    @property
    def status(self) -> str:
        return "enabled" if self.enabled else "disabled"

    def _client(self) -> Any:
        client_factory, _ = self._runtime_loader()
        client_kwargs: dict[str, str] = {"api_key": self._api_key}
        if self._endpoint:
            client_kwargs["api_url"] = self._endpoint
        if self._workspace_id:
            client_kwargs["workspace_id"] = self._workspace_id
        return client_factory(**client_kwargs)

    def preflight(self) -> LangSmithPreflightResult:
        """Verify workspace access plus a synthetic trace write/read round trip."""

        if not self.enabled:
            return LangSmithPreflightResult(status="disabled")
        try:
            client = self._client()
            run_id = uuid4()
            client.create_run(
                "specrag.langsmith.preflight",
                {"probe": "trace-write-read"},
                "chain",
                id=run_id,
                project_name=self.project_name,
                outputs={"status": "ready"},
                metadata={"phase": "langsmith_preflight"},
            )
            client.flush(timeout=10.0)
            observed = None
            for poll in range(1, 6):
                try:
                    observed = client.read_run(run_id)
                    break
                except Exception as exc:  # noqa: BLE001 - bounded visibility poll
                    if type(exc).__name__ != "LangSmithNotFoundError" or poll == 5:
                        raise
                    time.sleep(1.0)
            if str(getattr(observed, "id", "")) != str(run_id):
                raise RuntimeError("LangSmith preflight trace identity changed")
        except Exception as exc:  # noqa: BLE001 - classified without provider prose
            return _preflight_failure(exc)
        return LangSmithPreflightResult(status="ready")

    def require_ready(self) -> LangSmithPreflightResult:
        """Fail closed on an enabled but unavailable tracing destination."""

        result = self.preflight()
        if result.status == "unavailable":
            raise LangSmithPreflightError(result)
        return result

    @contextmanager
    def activate(
        self,
        *,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[None]:
        """Activate the official tracing context, or yield without side effects."""

        if not self.enabled:
            yield
            return

        _, tracing_context = self._runtime_loader()
        client = self._client()
        safe_tags = freeze_trace_tags(tags)
        safe_metadata = freeze_trace_metadata(metadata)
        with tracing_context(
            enabled=True,
            project_name=self.project_name,
            client=client,
            tags=list(safe_tags),
            metadata=dict(safe_metadata),
        ):
            yield


def build_langsmith_tracing(config: AppConfig) -> LangSmithTracing:
    """Resolve optional tracing without leaking any configured secret."""

    secrets = config.secrets
    api_key = secrets.langsmith_api_key.strip()
    enabled = bool(secrets.langsmith_tracing and api_key)
    return LangSmithTracing(
        enabled=enabled,
        project_name=secrets.langsmith_project.strip() or "specrag",
        _api_key=api_key if enabled else "",
        _endpoint=secrets.langsmith_endpoint.strip() if enabled else "",
        _workspace_id=secrets.langsmith_workspace_id.strip() if enabled else "",
    )


def _preflight_failure(exc: Exception) -> LangSmithPreflightResult:
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int):
        match = re.search(r"\b(401|403|404|408|429|5\d\d)\b", str(exc))
        status_code = int(match.group(1)) if match else None
    name = type(exc).__name__
    if status_code in {401, 403, 404}:
        kind = "authentication_or_configuration"
        retryable = False
    elif status_code == 408 or name in {
        "ConnectTimeout", "ReadTimeout", "Timeout", "TimeoutError",
    }:
        kind = "timeout"
        retryable = True
    elif (isinstance(status_code, int) and status_code >= 500) or name in {
        "ConnectionError", "ConnectError", "HTTPError",
    }:
        kind = "provider_unavailable"
        retryable = True
    else:
        kind = "unknown"
        retryable = False
    return LangSmithPreflightResult(
        status="unavailable", failure_kind=kind, retryable=retryable,
    )


def wrap_openai_client(client: Any, *, enabled: bool) -> Any:
    """Use LangSmith's official OpenAI wrapper only when explicitly enabled."""

    if not enabled:
        return client
    try:
        from langsmith.wrappers import wrap_openai
    except ImportError as exc:  # pragma: no cover - installation contract
        raise RuntimeError(
            "LangSmith tracing is enabled but the langsmith package is unavailable"
        ) from exc
    return wrap_openai(client)


def _load_runtime() -> tuple[Callable[..., Any], Callable[..., ContextManager[Any]]]:
    try:
        from langsmith import Client, tracing_context
    except ImportError as exc:  # pragma: no cover - installation contract
        raise RuntimeError(
            "LangSmith tracing is enabled but the langsmith package is unavailable"
        ) from exc
    return Client, tracing_context


__all__ = [
    "LangSmithPreflightError",
    "LangSmithPreflightResult",
    "LangSmithTracing",
    "TraceMetadataValue",
    "build_langsmith_tracing",
    "freeze_trace_metadata",
    "freeze_trace_tags",
    "wrap_openai_client",
]

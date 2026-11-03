"""Optional, secret-safe LangSmith activation for research workflows.

Durable Run events and model receipts remain authoritative.  This adapter
only activates the official LangSmith tracing context and OpenAI wrapper;
disabled or incomplete configuration is a true no-op.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Iterator, Mapping, Sequence

from specrag.config import AppConfig


RuntimeLoader = Callable[
    [], tuple[Callable[..., Any], Callable[..., ContextManager[Any]]]
]


@dataclass(frozen=True)
class LangSmithTracing:
    """Resolved tracing activation without credential-bearing diagnostics."""

    enabled: bool
    project_name: str
    _api_key: str = field(default="", repr=False)
    _endpoint: str = field(default="", repr=False)
    _runtime_loader: RuntimeLoader = field(
        default=lambda: _load_runtime(), repr=False, compare=False
    )

    @property
    def status(self) -> str:
        return "enabled" if self.enabled else "disabled"

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

        client_factory, tracing_context = self._runtime_loader()
        client_kwargs: dict[str, str] = {"api_key": self._api_key}
        if self._endpoint:
            client_kwargs["api_url"] = self._endpoint
        client = client_factory(**client_kwargs)
        with tracing_context(
            enabled=True,
            project_name=self.project_name,
            client=client,
            tags=list(tags),
            metadata=dict(metadata or {}),
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
    "LangSmithTracing",
    "build_langsmith_tracing",
    "wrap_openai_client",
]

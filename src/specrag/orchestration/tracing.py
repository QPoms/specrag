"""LangGraph root-run correlation without evaluator or provider data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import UUID

from specrag.infrastructure.langsmith_tracing import (
    TraceMetadataValue,
    freeze_trace_metadata,
    freeze_trace_tags,
)


@dataclass(frozen=True)
class WorkflowTraceContext:
    """Safe tags and metadata forwarded to one LangGraph root run."""

    run_id: UUID
    run_name: str
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, TraceMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.run_name.strip()
            or self.run_name != self.run_name.strip()
            or len(self.run_name) > 512
            or not self.run_name.isprintable()
        ):
            raise ValueError("trace run_name must be printable, trimmed, and non-blank")
        object.__setattr__(self, "tags", freeze_trace_tags(self.tags))
        object.__setattr__(self, "metadata", freeze_trace_metadata(self.metadata))


def workflow_config(
    thread_id: str,
    trace_context: WorkflowTraceContext | None = None,
) -> dict[str, Any]:
    """Build one LangGraph invocation config with optional safe correlation."""

    if not thread_id.strip():
        raise ValueError("thread_id must not be blank")
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if trace_context is not None:
        config.update({
            "run_id": trace_context.run_id,
            "run_name": trace_context.run_name,
            "tags": list(trace_context.tags),
            "metadata": dict(trace_context.metadata),
        })
    return config


__all__ = ["TraceMetadataValue", "WorkflowTraceContext", "workflow_config"]

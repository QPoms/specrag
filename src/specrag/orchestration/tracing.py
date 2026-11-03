"""LangGraph root-run correlation without evaluator or provider data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import UUID


TraceMetadataValue = str | int | float | bool


@dataclass(frozen=True)
class WorkflowTraceContext:
    """Safe tags and metadata forwarded to one LangGraph root run."""

    run_id: UUID
    run_name: str
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, TraceMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise ValueError("trace run_name must not be blank")
        if any(not tag.strip() or tag != tag.strip() for tag in self.tags):
            raise ValueError("trace tags must be non-blank and trimmed")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("trace tags must be unique")
        metadata = dict(self.metadata)
        if any(not key.strip() or key != key.strip() for key in metadata):
            raise ValueError("trace metadata keys must be non-blank and trimmed")
        if any(value is None for value in metadata.values()):
            raise ValueError("trace metadata values must not be null")


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

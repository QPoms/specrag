"""Bounded agent-callable capabilities backed by trusted project services."""

from .models import (
    RetrievalTool,
    RetrievalToolName,
    RetrievalToolRequest,
    RetrievalToolResult,
    ScoreKind,
    ToolEvidence,
    ToolCapability,
    ToolStatus,
)
from .registry import ToolRegistry
from .retrieval import (
    HyDESearchTool,
    build_retrieval_tool_registry,
    retrieval_tool_capabilities,
)

__all__ = [
    "RetrievalTool",
    "RetrievalToolName",
    "RetrievalToolRequest",
    "RetrievalToolResult",
    "ScoreKind",
    "ToolEvidence",
    "ToolCapability",
    "ToolRegistry",
    "ToolStatus",
    "HyDESearchTool",
    "build_retrieval_tool_registry",
    "retrieval_tool_capabilities",
]

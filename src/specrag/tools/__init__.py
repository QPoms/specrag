"""Bounded agent-callable capabilities backed by trusted project services."""

from .models import (
    ClauseLookupArguments,
    ExpandSourceContextArguments,
    HybridSearchArguments,
    HyDESearchArguments,
    KeywordSearchArguments,
    RerankArguments,
    RetrievalToolArguments,
    RetrievalTool,
    RetrievalToolName,
    RetrievalToolRequest,
    RetrievalToolResult,
    ScoreKind,
    SemanticSearchArguments,
    parse_retrieval_tool_arguments,
    ToolEvidence,
    ToolCapability,
    ToolStatus,
)
from .registry import ToolRegistry
from .retrieval import (
    HyDESearchTool,
    build_retrieval_tool_registry,
    retrieval_tool_capabilities,
    retrieval_tool_definitions,
)
from .research import conversation_research_tool_definitions

__all__ = [
    "RetrievalTool",
    "RetrievalToolArguments",
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
    "retrieval_tool_definitions",
    "ClauseLookupArguments",
    "ExpandSourceContextArguments",
    "HybridSearchArguments",
    "HyDESearchArguments",
    "KeywordSearchArguments",
    "RerankArguments",
    "SemanticSearchArguments",
    "parse_retrieval_tool_arguments",
    "conversation_research_tool_definitions",
]

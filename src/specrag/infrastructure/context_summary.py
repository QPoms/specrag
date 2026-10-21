"""Structured LLM adapter for Context Harness summary generation."""

from __future__ import annotations

from typing import Any, Protocol

from specrag.context import SummaryRequest
from specrag.models import CompactionSummary
from specrag.prompts import (
    CONTEXT_SUMMARY_INSTRUCTIONS,
    CONTEXT_SUMMARY_PROMPT_VERSION,
)


class _SummaryAgent(Protocol):
    def parse(self, **kwargs: Any) -> Any: ...


class StructuredContextSummarizer:
    def __init__(self, agent: _SummaryAgent) -> None:
        self.agent = agent

    def summarize(self, request: SummaryRequest) -> CompactionSummary:
        response = self.agent.parse(
            purpose="context_compaction",
            prompt_version=CONTEXT_SUMMARY_PROMPT_VERSION,
            instructions=CONTEXT_SUMMARY_INSTRUCTIONS,
            input_payload=request,
            output_type=CompactionSummary,
        )
        return CompactionSummary.model_validate(response.output)


__all__ = ["StructuredContextSummarizer"]

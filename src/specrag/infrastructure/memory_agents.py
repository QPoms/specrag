"""Structured-model adapters for Memory extraction and consolidation."""

from __future__ import annotations

from dataclasses import dataclass

from specrag.models import (
    MemoryConsolidationResult,
    MemoryEntry,
    MemoryExtractionResult,
    MemoryExtractionSource,
    MemoryMutationDraft,
    MemoryTopic,
)
from specrag.prompts import (
    MEMORY_CONSOLIDATION_INSTRUCTIONS,
    MEMORY_CONSOLIDATION_PROMPT_VERSION,
    MEMORY_EXTRACTION_INSTRUCTIONS,
    MEMORY_EXTRACTION_PROMPT_VERSION,
)

from .structured_agent import CodexStructuredAgent


def _entry(item: MemoryEntry) -> dict[str, object]:
    return {
        "memory_id": item.memory_id,
        "scope": item.scope,
        "layer": item.layer,
        "topic_id": item.topic_id,
        "kind": item.kind,
        "key": item.key,
        "content": item.content,
        "source_refs": list(item.source_refs),
    }


@dataclass(frozen=True)
class StructuredMemoryExtractor:
    agent: CodexStructuredAgent

    def extract(
        self,
        source: MemoryExtractionSource,
        *,
        user_core: tuple[MemoryEntry, ...],
        project_core: tuple[MemoryEntry, ...],
        topics: tuple[MemoryTopic, ...],
        topic_entries: tuple[MemoryEntry, ...],
    ) -> tuple[MemoryMutationDraft, ...]:
        response = self.agent.parse(
            purpose="extract reviewable long-term Memory candidates",
            prompt_version=MEMORY_EXTRACTION_PROMPT_VERSION,
            instructions=MEMORY_EXTRACTION_INSTRUCTIONS,
            input_payload={
                "source": {
                    "source_ref": source.source_ref,
                    "user_text": source.user_text,
                    "assistant_text": source.assistant_text,
                },
                "user_core": [_entry(item) for item in user_core],
                "project_core": [_entry(item) for item in project_core],
                "topics": [topic.model_dump(mode="json") for topic in topics],
                "topic_entries": [_entry(item) for item in topic_entries],
            },
            output_type=MemoryExtractionResult,
        )
        return response.output.drafts


@dataclass(frozen=True)
class StructuredMemoryConsolidator:
    agent: CodexStructuredAgent

    def consolidate(
        self, entries: tuple[MemoryEntry, ...]
    ) -> tuple[MemoryMutationDraft, ...]:
        response = self.agent.parse(
            purpose="propose conservative Memory merges",
            prompt_version=MEMORY_CONSOLIDATION_PROMPT_VERSION,
            instructions=MEMORY_CONSOLIDATION_INSTRUCTIONS,
            input_payload={"entries": [_entry(item) for item in entries]},
            output_type=MemoryConsolidationResult,
        )
        return response.output.drafts


__all__ = ["StructuredMemoryConsolidator", "StructuredMemoryExtractor"]

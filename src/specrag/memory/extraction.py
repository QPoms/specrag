"""Reviewable Memory extraction from completed Turn or Run text."""

from __future__ import annotations

from typing import Protocol

from specrag.models import (
    ActorContext,
    MemoryEntry,
    MemoryExtractionSource,
    MemoryMutationDraft,
    MemoryProposal,
    MemoryTopic,
)

from .service import MemoryService, _key
from .proposals import MemoryProposalService

class MemoryDraftExtractor(Protocol):
    def extract(
        self,
        source: MemoryExtractionSource,
        *,
        user_core: tuple[MemoryEntry, ...],
        project_core: tuple[MemoryEntry, ...],
        topics: tuple[MemoryTopic, ...],
        topic_entries: tuple[MemoryEntry, ...],
    ) -> tuple[MemoryMutationDraft, ...]: ...


class MemoryExtractionService:
    def __init__(
        self, memory: MemoryService, proposals: MemoryProposalService
    ) -> None:
        self.memory = memory
        self.proposals = proposals

    def propose_from_source(
        self,
        context: ActorContext,
        source: MemoryExtractionSource,
        extractor: MemoryDraftExtractor,
    ) -> tuple[MemoryProposal, ...]:
        user_core = self.memory.list(context)
        project_core = self.memory.list_project_core(context)
        topics = self.memory.list_topics(context)
        topic_entries = tuple(
            entry
            for topic in topics
            for entry in self.memory.list_topic_entries(context, topic.topic_id)
        )
        drafts = extractor.extract(
            source,
            user_core=user_core,
            project_core=project_core,
            topics=topics,
            topic_entries=topic_entries,
        )
        proposals: list[MemoryProposal] = []
        for index, draft in enumerate(drafts):
            if draft.operation in {"merge", "archive", "restore"}:
                raise ValueError("Extractor cannot propose consolidation or pruning operations")
            enriched = draft.model_copy(
                update={
                    "source_refs": (source.source_ref,)
                }
            )
            proposals.append(
                self.proposals.propose(
                    context,
                    enriched,
                    operation_key=_key(
                        "memory:extract:",
                        source.source_ref,
                        str(index),
                        enriched.model_dump_json(),
                    ),
                )
            )
        return tuple(proposals)


__all__ = [
    "MemoryDraftExtractor", "MemoryExtractionService", "MemoryExtractionSource",
]

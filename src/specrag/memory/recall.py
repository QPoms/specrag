"""Bounded Core and Topic Memory recall."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from specrag.models import ActorContext, MemoryEntry, MemoryTopic

from .service import MemoryService


_TOKEN = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*|[\u4e00-\u9fff]+", re.I)
_CONFIGURATION_KINDS = {"default_release", "answer_language", "detail_level"}


def _tokenize(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.casefold()):
        token = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) == 1:
                tokens.append(token)
            else:
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        elif len(token) > 1:
            tokens.append(token)
    return tuple(tokens)


@dataclass(frozen=True)
class MemoryRecallResult:
    entries: tuple[MemoryEntry, ...]
    pinned_topic_ids: tuple[str, ...]
    recalled_topic_ids: tuple[str, ...]
    omitted_entry_count: int
    characters_used: int


class MemoryRecallService:
    def __init__(
        self,
        memory: MemoryService,
        *,
        topic_limit: int = 3,
        entry_limit: int = 40,
        character_limit: int = 12000,
    ) -> None:
        if topic_limit < 0 or entry_limit < 1 or character_limit < 1:
            raise ValueError("Memory recall budgets must be positive")
        self.memory = memory
        self.topic_limit = topic_limit
        self.entry_limit = entry_limit
        self.character_limit = character_limit

    def recall(self, context: ActorContext, query: str) -> MemoryRecallResult:
        if not query.strip():
            raise ValueError("Memory recall query cannot be blank")
        user_core = tuple(
            item
            for item in self.memory.list(context)
            if item.kind not in _CONFIGURATION_KINDS
        )
        project_core = self.memory.list_project_core(context)
        topics = self.memory.list_topics(context)
        pinned = tuple(topic for topic in topics if topic.pinned)
        normal = tuple(topic for topic in topics if not topic.pinned)
        recalled = self._rank_topics(context, query, normal)
        ordered = [*user_core, *project_core]
        for topic in (*pinned, *recalled):
            ordered.extend(self.memory.list_topic_entries(context, topic.topic_id))
        unique: list[MemoryEntry] = []
        seen: set[str] = set()
        for item in ordered:
            if item.memory_id not in seen:
                seen.add(item.memory_id)
                unique.append(item)
        selected: list[MemoryEntry] = []
        characters = 0
        for item in unique:
            size = len(item.key) + len(item.content)
            if len(selected) >= self.entry_limit or characters + size > self.character_limit:
                continue
            selected.append(item)
            characters += size
        return MemoryRecallResult(
            entries=tuple(selected),
            pinned_topic_ids=tuple(topic.topic_id for topic in pinned),
            recalled_topic_ids=tuple(topic.topic_id for topic in recalled),
            omitted_entry_count=len(unique) - len(selected),
            characters_used=characters,
        )

    def _rank_topics(
        self,
        context: ActorContext,
        query: str,
        topics: tuple[MemoryTopic, ...],
    ) -> tuple[MemoryTopic, ...]:
        if not topics or self.topic_limit == 0:
            return ()
        documents: list[tuple[MemoryTopic, Counter[str]]] = []
        for topic in topics:
            entries = self.memory.list_topic_entries(context, topic.topic_id)
            text = " ".join(
                (topic.slug, topic.description, *(item.content for item in entries))
            )
            documents.append((topic, Counter(_tokenize(text))))
        document_count = len(documents)
        frequencies: Counter[str] = Counter()
        for _, counts in documents:
            frequencies.update(counts.keys())
        inverse = {
            term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in frequencies.items()
        }
        lengths = [sum(counts.values()) for _, counts in documents]
        average = sum(lengths) / max(1, len(lengths))
        query_terms = set(_tokenize(query))
        scored: list[tuple[float, MemoryTopic]] = []
        for (topic, counts), length in zip(documents, lengths, strict=True):
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * length / max(1.0, average)
                )
                score += inverse.get(term, 0.0) * frequency * 2.5 / denominator
            if score > 0:
                scored.append((score, topic))
        scored.sort(key=lambda item: (-item[0], item[1].topic_id))
        return tuple(topic for _, topic in scored[: self.topic_limit])


__all__ = ["MemoryRecallResult", "MemoryRecallService"]

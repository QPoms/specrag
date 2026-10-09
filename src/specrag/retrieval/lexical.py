"""Deterministic BM25 retrieval over immutable Markdown chunks."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

from .index import searchable_text
from .models import MarkdownChunk
from .normalization import protocol_token_variants


_TOKEN = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*|[\u4e00-\u9fff]+", re.I)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does",
    "for", "from", "how", "if", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "under", "what", "when", "whether",
    "which", "with", "without",
})


def tokenize(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(text):
        raw_token = match.group(0)
        token_variants = protocol_token_variants(raw_token)
        if not token_variants:
            continue
        token = token_variants[0]
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) == 1:
                tokens.append(token)
            else:
                tokens.extend(
                    token[index : index + 2]
                    for index in range(len(token) - 1)
                )
            continue
        for variant in token_variants:
            if variant not in _STOPWORDS and len(variant) > 1:
                tokens.append(variant)
    return tuple(tokens)


class BM25Index:
    """Small in-memory BM25 index with deterministic tie-breaking."""

    def __init__(self, chunks: tuple[MarkdownChunk, ...]):
        self.chunks = chunks
        self.frequencies = tuple(
            Counter(tokenize(searchable_text(chunk))) for chunk in chunks
        )
        self.lengths = tuple(sum(freq.values()) for freq in self.frequencies)
        lengths_by_collection: dict[str, list[int]] = defaultdict(list)
        frequency_by_collection: dict[str, Counter[str]] = defaultdict(Counter)
        for chunk, frequencies, length in zip(
            chunks, self.frequencies, self.lengths, strict=True
        ):
            lengths_by_collection[chunk.source_collection].append(length)
            frequency_by_collection[chunk.source_collection].update(
                frequencies.keys()
            )
        self.average_length = {
            collection: sum(lengths) / max(1, len(lengths))
            for collection, lengths in lengths_by_collection.items()
        }
        self.idf = {
            collection: {
                term: math.log(
                    1 + (len(lengths_by_collection[collection]) - count + 0.5)
                    / (count + 0.5)
                )
                for term, count in frequencies.items()
            }
            for collection, frequencies in frequency_by_collection.items()
        }

    def search(
        self,
        text: str,
        *,
        source_collection: str,
        limit: int,
        allowed_specifications: frozenset[str] | None = None,
    ) -> list[tuple[str, float]]:
        query_terms = set(tokenize(text))
        average_length = self.average_length.get(source_collection, 1.0)
        inverse_frequency = self.idf.get(source_collection, {})
        scored: list[tuple[str, float]] = []
        for chunk, frequencies, length in zip(
            self.chunks, self.frequencies, self.lengths, strict=True
        ):
            if chunk.source_collection != source_collection:
                continue
            if (
                allowed_specifications is not None
                and chunk.specification not in allowed_specifications
            ):
                continue
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * length / max(1.0, average_length)
                )
                score += (
                    inverse_frequency.get(term, 0.0)
                    * frequency
                    * 2.5
                    / denominator
                )
            if score > 0:
                scored.append((chunk.chunk_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]


__all__ = ["BM25Index", "tokenize"]

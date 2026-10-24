"""Process-local reuse of immutable, read-only specification retrievers."""

from __future__ import annotations

import json
from threading import Lock
from typing import Callable

from specrag.config import RetrievalCfg

from .hybrid import HybridMarkdownRetriever
from .models import CorpusScope


RetrieverBuilder = Callable[[RetrievalCfg, CorpusScope], HybridMarkdownRetriever]


class RetrieverPool:
    """Construct at most one read-only retriever for each immutable scope."""

    def __init__(
        self,
        config: RetrievalCfg,
        builder: RetrieverBuilder | None = None,
    ) -> None:
        self.config = config
        self.builder = builder or self._build
        self._items: dict[str, HybridMarkdownRetriever] = {}
        self._lock = Lock()

    def get(self, scope: CorpusScope) -> HybridMarkdownRetriever:
        key = json.dumps(
            scope.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        with self._lock:
            current = self._items.get(key)
            if current is None:
                current = self.builder(self.config, scope)
                self._items[key] = current
            return current

    @staticmethod
    def _build(
        config: RetrievalCfg, scope: CorpusScope,
    ) -> HybridMarkdownRetriever:
        return HybridMarkdownRetriever(
            config,
            corpus_scope=scope,
            allow_index_build=False,
        )


__all__ = ["RetrieverPool"]

from __future__ import annotations

from types import SimpleNamespace

import pytest

from specrag.config import RetrievalCfg
from specrag.retrieval import hybrid
from specrag.retrieval.hybrid import HybridMarkdownRetriever
from specrag.retrieval.models import CorpusScope, SourceDocument
from specrag.retrieval.pool import RetrieverPool


def test_retriever_pool_reuses_one_read_only_retriever_per_scope() -> None:
    built: list[CorpusScope] = []

    def builder(_config, scope):
        built.append(scope)
        return SimpleNamespace(corpus_scope=scope)

    pool = RetrieverPool(RetrievalCfg(), builder=builder)
    release17 = CorpusScope(
        version_policy="highest_version_within_release", release=17
    )
    release18 = CorpusScope(
        version_policy="highest_version_within_release", release=18
    )

    first = pool.get(release17)
    replay = pool.get(release17.model_copy())
    other = pool.get(release18)

    assert first is replay
    assert other is not first
    assert built == [release17, release18]


def test_read_only_retriever_fails_without_creating_a_chunk_snapshot(
    tmp_path, monkeypatch,
) -> None:
    document = SourceDocument(
        source_collection="3gpp_markdown",
        specification="24.501",
        version="17.10.0",
        release=17,
        source_path="data/3gpp/24.501/17.10.0.md",
        document_sha256="A" * 64,
    )
    index_root = tmp_path / "indexes"
    monkeypatch.setattr(
        hybrid,
        "discover_source_documents",
        lambda _config, _scope: (document,),
    )
    monkeypatch.setattr(
        hybrid,
        "repository_path",
        lambda _value, _label: index_root,
    )

    with pytest.raises(RuntimeError, match="chunk snapshot is missing"):
        HybridMarkdownRetriever(
            RetrievalCfg(),
            corpus_scope=CorpusScope(
                version_policy="highest_version_within_release",
                release=17,
            ),
            allow_index_build=False,
        )

    assert not index_root.exists()

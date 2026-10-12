from __future__ import annotations

from specrag.evaluation.first_turn import _target_matches
from specrag.tools import ToolEvidence


def _evidence(*, line_start: int = 10, document_hash: str = "A" * 64):
    return ToolEvidence(
        evidence_id="E1",
        chunk_id="C1",
        rank=1,
        match_kind="hybrid_search",
        score=0.1,
        score_kind="rrf",
        specification="33.501",
        release=17,
        version="17.1.0",
        source_path="data/3gpp/33.501/v17.1.0/source.md",
        document_sha256=document_hash,
        line_start=line_start,
        line_end=line_start + 4,
        excerpt="Normative source text.",
    )


def test_target_match_requires_identity_and_line_overlap() -> None:
    target = {
        "source_path": "data/3gpp/33.501/v17.1.0/source.md",
        "document_sha256": "a" * 64,
        "line_start": 12,
        "line_end": 16,
    }

    assert _target_matches(target, _evidence()) is True
    assert _target_matches(target, _evidence(line_start=20)) is False
    assert _target_matches(target, _evidence(document_hash="B" * 64)) is False

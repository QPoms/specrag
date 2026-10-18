"""JSONL loaders for evidence-path semantic judgments."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .component_evidence_models import (
    FinalEvidenceJudgment,
    ResearcherEvidenceJudgment,
    RetrievalCallJudgment,
    SynthesizerJudgment,
)


def load_retrieval_judgments(
    path: Path,
) -> dict[tuple[str, str], RetrievalCallJudgment]:
    return _load_judgments(
        path,
        RetrievalCallJudgment,
        lambda item: (item.question_id, item.call_id),
    )


def load_researcher_judgments(
    path: Path,
) -> dict[tuple[str, str], ResearcherEvidenceJudgment]:
    return _load_judgments(
        path,
        ResearcherEvidenceJudgment,
        lambda item: (item.question_id, item.task_id),
    )


def load_final_evidence_judgments(path: Path) -> dict[str, FinalEvidenceJudgment]:
    return _load_judgments(
        path,
        FinalEvidenceJudgment,
        lambda item: item.question_id,
    )


def load_synthesizer_judgments(path: Path) -> dict[str, SynthesizerJudgment]:
    return _load_judgments(
        path,
        SynthesizerJudgment,
        lambda item: item.question_id,
    )


def _load_judgments(path: Path, model: type[BaseModel], key):
    results = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not raw_line.strip():
            continue
        item = model.model_validate_json(raw_line)
        item_key = key(item)
        if item_key in results:
            raise ValueError(f"duplicate judgment {item_key!r} at line {line_number}")
        results[item_key] = item
    if not results:
        raise ValueError("judgment file is empty")
    return results


__all__ = [
    "load_final_evidence_judgments",
    "load_researcher_judgments",
    "load_retrieval_judgments",
    "load_synthesizer_judgments",
]

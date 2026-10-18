"""Reproducibility metadata for complete-workflow baseline runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Protocol

from specrag.config import REPO_ROOT
from specrag.interfaces.baseline_inputs import BaselineQuestion, QUESTION_SET
from specrag.models import ResearchBudget


class BaselineApplication(Protocol):
    release: int


def retrieval_scope_metadata(app: BaselineApplication) -> dict[str, object]:
    scope = getattr(app, "corpus_scope", None)
    metadata: dict[str, object] = (
        scope.model_dump(mode="json")
        if scope is not None else {"release": app.release}
    )
    fingerprint = getattr(app, "retrieval_index_fingerprint", None)
    if fingerprint:
        metadata["index_fingerprint"] = fingerprint
    documents = tuple(getattr(app, "corpus_documents", ()))
    metadata["documents"] = [
        item.model_dump(mode="json") for item in documents
    ]
    return metadata


def _source_revision(repository_root: Path) -> dict[str, object]:
    git_prefix = [
        "git", "-c", f"safe.directory={repository_root.as_posix()}",
    ]
    try:
        commit = subprocess.run(
            [*git_prefix, "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            [*git_prefix, "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unavailable", "dirty": None}
    return {"git_commit": commit, "dirty": dirty}


def benchmark_metadata(
    app: BaselineApplication,
    questions: tuple[BaselineQuestion, ...],
    budget: ResearchBudget,
) -> dict[str, object]:
    return {
        "question_set": QUESTION_SET.relative_to(REPO_ROOT).as_posix(),
        "question_set_sha256": hashlib.sha256(QUESTION_SET.read_bytes()).hexdigest(),
        "selected_question_ids": [item.question_id for item in questions],
        "source_revision": _source_revision(REPO_ROOT),
        "budget": budget.model_dump(mode="json"),
        "retrieval_scope": retrieval_scope_metadata(app),
        "cache_condition": "warm_cache",
        "execution_order": "sequential_questions",
    }


__all__ = ["benchmark_metadata", "retrieval_scope_metadata"]

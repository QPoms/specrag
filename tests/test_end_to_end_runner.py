from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_end_to_end import _evaluation_paths


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_batch_manifest_prevents_missing_case_from_leaving_denominator(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "manifest.json", {
        "schema_version": "specrag.end-to-end-evaluation-batch/v1",
        "question_ids": ["r17-qa-001", "r17-qa-002"],
    })
    _write_json(tmp_path / "r17-qa-001.json", {})

    with pytest.raises(ValueError, match="missing: r17-qa-002.json"):
        _evaluation_paths(tmp_path)


def test_batch_manifest_rejects_stale_extra_case(tmp_path: Path) -> None:
    _write_json(tmp_path / "manifest.json", {
        "schema_version": "specrag.end-to-end-evaluation-batch/v1",
        "question_ids": ["r17-qa-001"],
    })
    _write_json(tmp_path / "r17-qa-001.json", {})
    _write_json(tmp_path / "r17-qa-002.json", {})

    with pytest.raises(ValueError, match="unexpected: r17-qa-002.json"):
        _evaluation_paths(tmp_path)

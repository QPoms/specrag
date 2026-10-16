from __future__ import annotations

import json

from specrag.evaluation.answer_quality import (
    evaluate_answer_artifact,
    summarize_answer_evaluations,
    target_matches,
)


def _target(line_start: int = 10) -> dict[str, object]:
    return {
        "specification": "23.501",
        "version": "17.15.0",
        "clause": "5.31.7.3",
        "source_path": "data/3gpp/23.501/v17.15.0/23501-hf0.md",
        "document_sha256": "A" * 64,
        "line_start": line_start,
        "line_end": line_start + 2,
    }


def _evidence(evidence_id: str = "E1", line_start: int = 10) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "source_path": "data/3gpp/23.501/v17.15.0/23501-hf0.md",
        "document_sha256": "A" * 64,
        "line_start": line_start,
        "line_end": line_start + 2,
    }


def _write_artifacts(tmp_path, *, claims, input_ids=("E1",), status="completed"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "research.json"
    source.write_text(json.dumps({
        "state": {
            "candidate_evidence": [_evidence()],
            "validated_evidence": [{
                "evidence_id": "E1",
                "source_valid": True,
                "supports_target": True,
                "validation_status": "supported",
            }],
        },
    }), encoding="utf-8")
    artifact = tmp_path / "r17-qa-001.json"
    artifact.write_text(json.dumps({
        "status": status,
        "question_id": "r17-qa-001",
        "source_artifact": str(source),
        "input_evidence_ids": list(input_ids),
        "synthesis": {"answer": "Grounded answer.", "claims": claims},
    }), encoding="utf-8")
    return artifact


def _qa(targets=None):
    return {
        "id": "r17-qa-001",
        "difficulty": "simple",
        "reasoning_scope": "single_clause",
        "evidence_targets": targets or [_target()],
    }


def test_target_matches_document_hash_path_and_line_overlap() -> None:
    assert target_matches(_target(), _evidence())
    assert not target_matches(_target(), _evidence(line_start=20))


def test_answer_evaluation_reports_supported_target_coverage(tmp_path) -> None:
    artifact = _write_artifacts(tmp_path, claims=[{
        "claim_id": "C1",
        "text": "Grounded claim.",
        "evidence_ids": ["E1"],
    }])

    result = evaluate_answer_artifact(artifact, _qa())

    assert result.status == "completed"
    assert result.answer_present is True
    assert result.target_recall == 1
    assert result.citation_validity == 1
    assert result.claim_support_rate == 1
    assert result.all_targets_covered is True


def test_answer_evaluation_keeps_unknown_citations_auditable(tmp_path) -> None:
    artifact = _write_artifacts(tmp_path, claims=[{
        "claim_id": "C1",
        "text": "Unsupported claim.",
        "evidence_ids": ["MISSING"],
    }], input_ids=("E1", "MISSING"))

    result = evaluate_answer_artifact(artifact, _qa())

    assert result.known_citation_count == 0
    assert result.supported_citation_count == 0
    assert result.citation_validity == 0
    assert result.target_recall == 0


def test_answer_evaluation_deduplicates_reordered_target_matches(tmp_path) -> None:
    artifact = _write_artifacts(tmp_path, claims=[
        {
            "claim_id": "C2",
            "text": "Second claim.",
            "evidence_ids": ["E1"],
        },
        {
            "claim_id": "C1",
            "text": "Repeated claim.",
            "evidence_ids": ["E1"],
        },
    ])

    result = evaluate_answer_artifact(artifact, _qa())

    assert result.target_recall == 1
    assert result.covered_target_count == 1
    assert result.relevant_citation_count == 2


def test_answer_summary_groups_difficulty_and_scope(tmp_path) -> None:
    complete = evaluate_answer_artifact(
        _write_artifacts(tmp_path / "complete", claims=[{
            "claim_id": "C1", "text": "Answer", "evidence_ids": ["E1"],
        }]),
        _qa(),
    )
    skipped = evaluate_answer_artifact(
        _write_artifacts(tmp_path / "skipped", claims=[], status="skipped"),
        _qa(),
    )

    summary = summarize_answer_evaluations((complete, skipped))

    assert summary["question_count"] == 2
    assert summary["completed_count"] == 1
    assert summary["skipped_count"] == 1
    assert summary["by_difficulty"]["simple"]["completed_count"] == 1

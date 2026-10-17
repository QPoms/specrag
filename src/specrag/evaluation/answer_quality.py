"""Deterministic alignment and aggregation for generated research answers."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from specrag.config import REPO_ROOT


EvaluationStatus = Literal["completed", "skipped", "error"]


class AnswerEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["specrag.answer-evaluation/v1"] = (
        "specrag.answer-evaluation/v1"
    )
    question_id: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    reasoning_scope: str = Field(min_length=1)
    status: EvaluationStatus
    reason: str | None = None
    answer_present: bool = False
    claim_count: int = Field(default=0, ge=0)
    cited_evidence_count: int = Field(default=0, ge=0)
    known_citation_count: int = Field(default=0, ge=0)
    supported_citation_count: int = Field(default=0, ge=0)
    relevant_citation_count: int = Field(default=0, ge=0)
    supported_claim_count: int = Field(default=0, ge=0)
    gold_target_count: int = Field(default=0, ge=0)
    covered_target_count: int = Field(default=0, ge=0)
    citation_validity: float = Field(default=0.0, ge=0, le=1)
    citation_relevance: float = Field(default=0.0, ge=0, le=1)
    claim_support_rate: float = Field(default=0.0, ge=0, le=1)
    target_recall: float = Field(default=0.0, ge=0, le=1)
    all_targets_covered: bool = False
    target_coverage: tuple[dict[str, Any], ...] = ()


def target_matches(target: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    """Match a citation to an annotated target without comparing answer text."""

    source_path = str(target["source_path"]).replace("\\", "/").casefold()
    evidence_path = str(evidence.get("source_path", "")).replace("\\", "/")
    if evidence_path.casefold() != source_path:
        return False
    if str(evidence.get("document_sha256", "")).casefold() != str(
        target["document_sha256"]
    ).casefold():
        return False
    try:
        evidence_start = int(evidence["line_start"])
        evidence_end = int(evidence["line_end"])
        target_start = int(target["line_start"])
        target_end = int(target["line_end"])
    except (KeyError, TypeError, ValueError):
        return False
    return evidence_start <= target_end and target_start <= evidence_end


def _artifact_path(value: object, *, base_directory: Path | None = None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = (
        base_directory / path if base_directory is not None else path,
        path,
        REPO_ROOT / path,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), path)


def _source_evidence(artifact: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    state = artifact.get("state", {})
    candidates = state.get("candidate_evidence", ()) if isinstance(state, Mapping) else ()
    return {
        str(item["evidence_id"]): item
        for item in candidates
        if isinstance(item, Mapping) and item.get("evidence_id")
    }


def _supported_ids(artifact: Mapping[str, Any]) -> set[str]:
    state = artifact.get("state", {})
    validations = state.get("validated_evidence", ()) if isinstance(state, Mapping) else ()
    return {
        str(item["evidence_id"])
        for item in validations
        if isinstance(item, Mapping)
        and item.get("evidence_id")
        and item.get("source_valid") is True
        and item.get("supports_target") is True
        and item.get("validation_status") == "supported"
    }


def evaluate_answer_artifact(
    artifact_path: Path,
    qa_record: Mapping[str, Any],
) -> AnswerEvaluation:
    """Evaluate one synthesis artifact using only persisted evidence metadata."""

    difficulty = str(qa_record.get("difficulty", "unknown"))
    reasoning_scope = str(qa_record.get("reasoning_scope", "unknown"))
    raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    question_id = str(raw.get("question_id") or qa_record["id"])
    # Current synthesis artifacts are flat; the retained baseline runner used
    # a durable `state` envelope.  Normalize both formats at this boundary so
    # historical runs can be evaluated without rewriting generated artifacts.
    state = raw.get("state", {})
    if not isinstance(state, Mapping):
        state = {}
    status = raw.get("status")
    if status is None:
        status = "completed" if state.get("stop_reason") == "completed" else state.get(
            "stop_reason", "not_completed"
        )
    if status != "completed":
        return AnswerEvaluation(
            question_id=question_id,
            difficulty=difficulty,
            reasoning_scope=reasoning_scope,
            status="skipped",
            reason=str(status),
        )

    synthesis = raw.get("synthesis") or state.get("synthesis") or {}
    if not synthesis and state.get("final_answer"):
        synthesis = {"answer": state["final_answer"], "evidence_ids": ()}
    answer = str(synthesis.get("answer", "")).strip()
    claims = tuple(item for item in synthesis.get("claims", ()) if isinstance(item, Mapping))
    source_value = raw.get("source_artifact")
    source = {}
    if source_value:
        source_path = _artifact_path(source_value, base_directory=artifact_path.parent)
        if source_path.is_file():
            source = json.loads(source_path.read_text(encoding="utf-8"))
    elif state:
        source = raw
    evidence_by_id = _source_evidence(source)
    selected_ids = {str(value) for value in raw.get("input_evidence_ids", ())}
    if not selected_ids:
        selected_ids = set(evidence_by_id)
    evidence_by_id = {
        key: value for key, value in evidence_by_id.items() if key in selected_ids
    }
    supported_ids = _supported_ids(source) & set(evidence_by_id)
    cited_ids = tuple(str(value) for value in synthesis.get("evidence_ids", ()))
    if not cited_ids:
        cited_ids = tuple(
            str(evidence_id)
            for claim in claims
            for evidence_id in claim.get("evidence_ids", ())
        )
    known_citations = sum(item in evidence_by_id for item in cited_ids)
    supported_citations = sum(item in supported_ids for item in cited_ids)
    targets = tuple(qa_record.get("evidence_targets", ()))
    coverage: list[dict[str, Any]] = []
    covered_count = 0
    relevant_ids: set[str] = set()
    for index, target in enumerate(targets):
        matches = tuple(
            evidence_id
            for evidence_id in dict.fromkeys(cited_ids)
            if evidence_id in evidence_by_id
            and target_matches(target, evidence_by_id[evidence_id])
        )
        if matches:
            covered_count += 1
            relevant_ids.update(matches)
        coverage.append({
            "target_index": index,
            "specification": target.get("specification"),
            "clause": target.get("clause"),
            "matched_evidence_ids": matches,
            "covered": bool(matches),
        })
    supported_claims = (
        sum(
            bool(claim.get("evidence_ids"))
            and all(str(item) in supported_ids for item in claim.get("evidence_ids", ()))
            for claim in claims
        )
        if claims
        else int(bool(answer) and bool(cited_ids) and all(
            item in supported_ids for item in cited_ids
        ))
    )
    citation_count = len(cited_ids)
    claim_count = len(claims) if claims else int(bool(answer))
    target_count = len(targets)
    return AnswerEvaluation(
        question_id=question_id,
        difficulty=difficulty,
        reasoning_scope=reasoning_scope,
        status="completed",
        answer_present=bool(answer),
        claim_count=claim_count,
        cited_evidence_count=citation_count,
        known_citation_count=known_citations,
        supported_citation_count=supported_citations,
        relevant_citation_count=sum(item in relevant_ids for item in cited_ids),
        supported_claim_count=supported_claims,
        gold_target_count=target_count,
        covered_target_count=covered_count,
        citation_validity=(supported_citations / citation_count if citation_count else 0.0),
        citation_relevance=(
            sum(item in relevant_ids for item in cited_ids) / citation_count
            if citation_count else 0.0
        ),
        claim_support_rate=(supported_claims / claim_count if claim_count else 0.0),
        target_recall=(covered_count / target_count if target_count else 0.0),
        all_targets_covered=bool(target_count) and covered_count == target_count,
        target_coverage=tuple(coverage),
    )


def summarize_answer_evaluations(
    evaluations: tuple[AnswerEvaluation, ...],
) -> dict[str, Any]:
    """Aggregate transparent metrics overall and by QA difficulty/scope."""

    def group(items: tuple[AnswerEvaluation, ...]) -> dict[str, Any]:
        completed = tuple(item for item in items if item.status == "completed")
        def mean(field: str) -> float:
            return round(
                sum(float(getattr(item, field)) for item in completed) / len(completed),
                4,
            ) if completed else 0.0
        return {
            "question_count": len(items),
            "completed_count": len(completed),
            "answer_present_rate": round(
                sum(item.answer_present for item in completed) / len(completed), 4
            ) if completed else 0.0,
            "all_targets_covered_rate": round(
                sum(item.all_targets_covered for item in completed) / len(completed), 4
            ) if completed else 0.0,
            "target_recall": mean("target_recall"),
            "citation_validity": mean("citation_validity"),
            "citation_relevance": mean("citation_relevance"),
            "claim_support_rate": mean("claim_support_rate"),
        }

    by_difficulty: dict[str, list[AnswerEvaluation]] = defaultdict(list)
    by_scope: dict[str, list[AnswerEvaluation]] = defaultdict(list)
    for item in evaluations:
        by_difficulty[item.difficulty].append(item)
        by_scope[item.reasoning_scope].append(item)
    return {
        "schema_version": "specrag.answer-evaluation-summary/v1",
        "question_count": len(evaluations),
        "completed_count": sum(item.status == "completed" for item in evaluations),
        "skipped_count": sum(item.status == "skipped" for item in evaluations),
        "error_count": sum(item.status == "error" for item in evaluations),
        "overall": group(evaluations),
        "by_difficulty": {
            key: group(tuple(value)) for key, value in sorted(by_difficulty.items())
        },
        "by_reasoning_scope": {
            key: group(tuple(value)) for key, value in sorted(by_scope.items())
        },
    }


__all__ = [
    "AnswerEvaluation",
    "evaluate_answer_artifact",
    "summarize_answer_evaluations",
    "target_matches",
]

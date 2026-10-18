"""Internal artifact and metric helpers for component evidence evaluation."""

from __future__ import annotations

import json
from math import log2
from pathlib import Path
from typing import Any, Mapping

from .component_models import ComponentQuestionAnnotation


def read_artifact(path: Path) -> tuple[bytes, Mapping[str, Any]]:
    raw = path.read_bytes()
    artifact = json.loads(raw)
    if not isinstance(artifact, Mapping):
        raise ValueError("component artifact must be a JSON object")
    return raw, artifact


def state(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    value = artifact.get("state")
    return value if isinstance(value, Mapping) else artifact


def require_question(artifact: Mapping[str, Any], expected: str) -> None:
    question = artifact.get("question")
    question_id = question.get("id") if isinstance(question, Mapping) else None
    question_id = artifact.get("question_id") or question_id or state(artifact).get(
        "research_id"
    )
    if str(question_id) != expected:
        raise ValueError(f"artifact question {question_id!r} does not match {expected}")


def locator_matches(locator: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    same_path = str(locator["source_path"]).replace("\\", "/").casefold() == str(
        evidence.get("source_path", "")
    ).replace("\\", "/").casefold()
    same_hash = str(locator["document_sha256"]).casefold() == str(
        evidence.get("document_sha256", "")
    ).casefold()
    try:
        overlaps = int(evidence["line_start"]) <= int(locator["line_end"]) and int(
            locator["line_start"]
        ) <= int(evidence["line_end"])
    except (KeyError, TypeError, ValueError):
        return False
    return same_path and same_hash and overlaps


def locator_equals(locator: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    return (
        str(locator["source_path"]).replace("\\", "/").casefold()
        == str(evidence.get("source_path", "")).replace("\\", "/").casefold()
        and str(locator["document_sha256"]).casefold()
        == str(evidence.get("document_sha256", "")).casefold()
        and int(locator["line_start"]) == int(evidence.get("line_start", -1))
        and int(locator["line_end"]) == int(evidence.get("line_end", -1))
    )


def target_recall(results, targets, k: int) -> float:
    hits = sum(
        any(locator_matches(target.locator.model_dump(), result) for result in results[:k])
        for target in targets
    )
    return hits / len(targets)


def ndcg(relevances: tuple[int, ...], qrel_grades: tuple[int, ...], k: int) -> float:
    def dcg(values) -> float:
        return sum(
            (2**grade - 1) / log2(index + 2)
            for index, grade in enumerate(values[:k])
        )

    ideal = dcg(sorted(qrel_grades, reverse=True))
    return dcg(relevances) / ideal if ideal else 0.0


def candidate_evidence(
    artifact: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        item
        for item in state(artifact).get("candidate_evidence", ())
        if isinstance(item, Mapping) and item.get("evidence_id")
    )


def unique_evidence_ids(items) -> tuple[str, ...]:
    ids = tuple(str(item["evidence_id"]) for item in items)
    if len(ids) != len(set(ids)):
        raise ValueError("artifact evidence IDs must be unique")
    return ids


def task_evidence_units(
    annotation: ComponentQuestionAnnotation, task_id: str
) -> set[str]:
    tasks = tuple(item for item in annotation.gold_tasks if item.task_id == task_id)
    if len(tasks) != 1:
        raise ValueError(f"annotation requires exactly one gold task {task_id}")
    research_units = set(tasks[0].research_unit_ids)
    unit_ids = {
        item.unit_id
        for item in annotation.evidence_units
        if research_units.intersection(item.research_unit_ids)
    }
    if not unit_ids:
        raise ValueError(f"gold task {task_id} has no EvidenceUnits")
    return unit_ids


def require_fixed_gold_task(
    artifact: Mapping[str, Any],
    annotation: ComponentQuestionAnnotation,
    task_id: str,
) -> None:
    if artifact.get("schema_version") != "specrag.component-researcher-run/v1":
        return
    expected = next(
        (item for item in annotation.gold_tasks if item.task_id == task_id), None
    )
    actual = next(
        (
            item
            for item in state(artifact).get("research_tasks", ())
            if isinstance(item, Mapping) and item.get("task_id") == task_id
        ),
        None,
    )
    if expected is None or actual is None:
        raise ValueError(f"fixed GoldResearchTask {task_id} is absent")
    expected_fields = {
        "task_id": expected.task_id,
        "question": expected.question,
        "objective": expected.objective,
        "dependencies": list(expected.dependencies),
    }
    actual_fields = {
        key: actual.get(key, [] if key == "dependencies" else "")
        for key in expected_fields
    }
    if actual_fields != expected_fields:
        raise ValueError(f"artifact task {task_id} differs from reviewed GoldResearchTask")


def require_assessed_units(expected: set[str], assessments) -> None:
    actual = {item.unit_id for item in assessments}
    if actual != expected:
        raise ValueError(set_difference("EvidenceUnits", expected, actual))


def require_evidence_unit_references(assessments, unit_ids: set[str]) -> None:
    unknown = {
        unit_id
        for item in assessments
        for unit_id in item.evidence_unit_ids
        if unit_id not in unit_ids
    }
    if unknown:
        raise ValueError(f"evidence assessments reference unknown units: {sorted(unknown)}")


def deduplicated_final_evidence(
    artifact: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    artifact_state = state(artifact)
    supported_ids = {
        str(item.get("evidence_id"))
        for item in artifact_state.get("validated_evidence", ())
        if isinstance(item, Mapping)
        and item.get("source_valid") is True
        and item.get("supports_target") is True
        and item.get("validation_status") == "supported"
    }
    seen: set[tuple[object, ...]] = set()
    selected: list[Mapping[str, Any]] = []
    for item in candidate_evidence(artifact):
        if str(item["evidence_id"]) not in supported_ids:
            continue
        key = (
            str(item.get("source_path", "")).replace("\\", "/").casefold(),
            str(item.get("document_sha256", "")).casefold(),
            item.get("line_start"),
            item.get("line_end"),
        )
        if key not in seen:
            seen.add(key)
            selected.append(item)
    return tuple(selected)


def synthesis_citations(synthesis: Mapping[str, Any]) -> set[str]:
    direct = {str(item) for item in synthesis.get("evidence_ids", ())}
    claims = synthesis.get("claims", ())
    return direct or {
        str(evidence_id)
        for claim in claims
        if isinstance(claim, Mapping)
        for evidence_id in claim.get("evidence_ids", ())
    }


def set_difference(label: str, expected: set[str], actual: set[str]) -> str:
    return (
        f"{label} mismatch; missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}"
    )


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def optional_text(value: object) -> str | None:
    return str(value) if value not in (None, "") else None


def artifact_metadata(artifact: Mapping[str, Any], key: str) -> str | None:
    if artifact.get(key):
        return str(artifact[key])
    receipt = artifact.get("receipt")
    if isinstance(receipt, Mapping) and receipt.get(key):
        return str(receipt[key])
    return None

"""Validate the curated Release 17 development question set."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUESTION_SET = ROOT / "data" / "QA" / "questions.jsonl"
CORPUS_ROOT = (ROOT / "data" / "3gpp").resolve()
EXPECTED_IDENTIFIERS = {f"r17-qa-{index:03d}" for index in range(1, 51)}
EXPECTED_DIFFICULTIES = Counter({"simple": 34, "complex": 16})
EXPECTED_SCOPES = Counter(
    {"single_clause": 30, "cross_clause": 4, "cross_specification": 16}
)
EXPECTED_SPLITS = Counter({"development": 50})
ALLOWED_TRANSFORMATIONS = {
    "verbatim",
    "light_edit",
    "focused_rewrite",
    "scope_expansion",
    "cross_spec_extension",
}

REQUIRED_RECORD_FIELDS = {
    "schema_version",
    "id",
    "question",
    "release",
    "difficulty",
    "reasoning_scope",
    "split",
    "source",
    "expected_answer",
    "expected_facts",
    "evidence_targets",
}
REQUIRED_SOURCE_FIELDS = {
    "dataset",
    "record_kind",
    "record_id",
    "license",
    "original_question",
    "transformation",
}
REQUIRED_EVIDENCE_FIELDS = {
    "specification",
    "version",
    "clause",
    "source_path",
    "document_sha256",
    "line_start",
    "line_end",
    "excerpt",
}


def _require_fields(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")


def _load_records() -> list[dict[str, Any]]:
    if not QUESTION_SET.is_file():
        raise ValueError(f"question set does not exist: {QUESTION_SET}")
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        QUESTION_SET.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        records.append(record)
    if not records:
        raise ValueError("question set is empty")
    return records


def _validate_evidence(evidence: dict[str, Any], label: str) -> None:
    _require_fields(evidence, REQUIRED_EVIDENCE_FIELDS, label)
    if not str(evidence["version"]).startswith("17."):
        raise ValueError(f"{label} is not bound to Release 17")

    relative_path = Path(str(evidence["source_path"]))
    source_path = (ROOT / relative_path).resolve()
    try:
        source_path.relative_to(CORPUS_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} source is outside data/3gpp") from exc
    if not source_path.is_file():
        raise ValueError(f"{label} source does not exist: {relative_path}")

    expected_path_parts = {
        str(evidence["specification"]),
        f"v{evidence['version']}",
    }
    if not expected_path_parts.issubset(set(relative_path.parts)):
        raise ValueError(f"{label} source path disagrees with spec/version fields")

    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest.lower() != str(evidence["document_sha256"]).lower():
        raise ValueError(f"{label} document hash does not match local source")

    lines = source_path.read_text(encoding="utf-8").splitlines()
    start = evidence["line_start"]
    end = evidence["line_end"]
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"{label} line locators must be integers")
    if start < 1 or end < start or end > len(lines):
        raise ValueError(f"{label} has an invalid line range {start}-{end}")
    actual_excerpt = "\n".join(lines[start - 1 : end])
    if actual_excerpt != evidence["excerpt"]:
        raise ValueError(f"{label} excerpt does not match lines {start}-{end}")


def validate_question_set() -> list[dict[str, Any]]:
    records = _load_records()
    identifiers: set[str] = set()
    questions: set[str] = set()
    original_questions: set[str] = set()
    source_records: set[tuple[str, str, str]] = set()
    for index, record in enumerate(records, start=1):
        label = f"record {index}"
        _require_fields(record, REQUIRED_RECORD_FIELDS, label)
        identifier = str(record["id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate question id: {identifier}")
        identifiers.add(identifier)

        if record["schema_version"] != "specrag.qa/v2":
            raise ValueError(f"{identifier} has an unsupported schema version")
        if record["release"] != 17:
            raise ValueError(f"{identifier} must target Release 17")
        if record["difficulty"] not in {"simple", "complex"}:
            raise ValueError(f"{identifier} has an invalid difficulty")
        if record["reasoning_scope"] not in {
            "single_clause",
            "cross_clause",
            "cross_specification",
        }:
            raise ValueError(f"{identifier} has an invalid reasoning scope")
        if record["split"] != "development":
            raise ValueError(f"{identifier} must use the development split")
        if not isinstance(record["question"], str) or not record["question"].strip():
            raise ValueError(f"{identifier} must define a question")
        normalized_question = " ".join(record["question"].casefold().split())
        if normalized_question in questions:
            raise ValueError(f"{identifier} duplicates another curated question")
        questions.add(normalized_question)
        if not isinstance(record["expected_answer"], str) or not record[
            "expected_answer"
        ].strip():
            raise ValueError(f"{identifier} must define an expected answer")
        if not isinstance(record["expected_facts"], list) or not record["expected_facts"]:
            raise ValueError(f"{identifier} must define expected facts")
        if any(not isinstance(fact, str) or not fact.strip() for fact in record["expected_facts"]):
            raise ValueError(f"{identifier} expected facts must be non-empty strings")

        source = record["source"]
        if not isinstance(source, dict):
            raise ValueError(f"{identifier} source must be an object")
        _require_fields(source, REQUIRED_SOURCE_FIELDS, f"{identifier} source")
        if source["dataset"] != "MM-TelcoBench" or source["record_kind"] != "IR":
            raise ValueError(f"{identifier} must reference an MM-TelcoBench IR record")
        if source["license"] != "CC-BY-4.0":
            raise ValueError(f"{identifier} has an unexpected source license")
        if source["transformation"] not in ALLOWED_TRANSFORMATIONS:
            raise ValueError(f"{identifier} has an unsupported transformation")
        if not isinstance(source["original_question"], str) or not source[
            "original_question"
        ].strip():
            raise ValueError(f"{identifier} must retain the original question")
        if (
            source["transformation"] == "verbatim"
            and record["question"] != source["original_question"]
        ):
            raise ValueError(f"{identifier} is not verbatim despite its transformation")
        normalized_original = " ".join(source["original_question"].casefold().split())
        if normalized_original in original_questions:
            raise ValueError(f"{identifier} duplicates another upstream question")
        original_questions.add(normalized_original)
        source_identity = (
            str(source["dataset"]),
            str(source["record_kind"]),
            str(source["record_id"]),
        )
        if source_identity in source_records:
            raise ValueError(
                f"{identifier} reuses upstream source record "
                f"{source_identity[0]}/{source_identity[1]}/{source_identity[2]}"
            )
        source_records.add(source_identity)

        targets = record["evidence_targets"]
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"{identifier} must define evidence targets")
        if record["reasoning_scope"] == "single_clause" and len(targets) != 1:
            raise ValueError(
                f"{identifier} single-clause evidence needs exactly one target"
            )
        if record["reasoning_scope"] != "single_clause" and len(targets) < 2:
            raise ValueError(
                f"{identifier} multi-source evidence needs at least two targets"
            )
        for evidence_index, evidence in enumerate(targets, start=1):
            if not isinstance(evidence, dict):
                raise ValueError(f"{identifier} evidence target must be an object")
            _validate_evidence(evidence, f"{identifier} evidence {evidence_index}")

        specifications = {str(target["specification"]) for target in targets}
        clauses = {
            (str(target["specification"]), str(target["clause"]))
            for target in targets
        }
        if record["reasoning_scope"] == "cross_specification" and len(specifications) < 2:
            raise ValueError(
                f"{identifier} cross-specification question needs distinct specifications"
            )
        if record["reasoning_scope"] == "cross_clause" and len(clauses) < 2:
            raise ValueError(f"{identifier} cross-clause question needs distinct clauses")

    if identifiers != EXPECTED_IDENTIFIERS:
        missing = sorted(EXPECTED_IDENTIFIERS - identifiers)
        unexpected = sorted(identifiers - EXPECTED_IDENTIFIERS)
        raise ValueError(
            f"question IDs do not form r17-qa-001..050; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if Counter(record["difficulty"] for record in records) != EXPECTED_DIFFICULTIES:
        raise ValueError("question set must contain 34 simple and 16 complex cases")
    if Counter(record["reasoning_scope"] for record in records) != EXPECTED_SCOPES:
        raise ValueError(
            "question set must contain 30 single-clause, 4 cross-clause, and "
            "16 cross-specification cases"
        )
    if Counter(record["split"] for record in records) != EXPECTED_SPLITS:
        raise ValueError("all 50 questions must use the development split")
    return records


def main() -> int:
    try:
        records = validate_question_set()
    except ValueError as exc:
        print(f"QA dataset check failed: {exc}")
        return 1
    print(f"QA dataset check passed: {len(records)} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

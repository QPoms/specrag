"""Load and validate the composed final-answer Benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .benchmark_models import (
    BenchmarkCase,
    BenchmarkCaseEnvironment,
    BenchmarkManifest,
    BenchmarkReviewCase,
    BenchmarkReviewFile,
    BenchmarkReview,
    BenchmarkRubric,
    EndToEndBenchmark,
    GoldEvidence,
    RubricAnnotation,
    RubricItem,
    ScoreBand,
    SourceMetadata,
    TaskMetadata,
)
from .component_models import ComponentQuestionAnnotation


MANIFEST_PATH = Path("data/QA/end_to_end_benchmark.json")
QUESTION_PATH = Path("data/QA/questions.jsonl")
ANNOTATION_PATH = Path("data/QA/component_evaluation_release17.jsonl")
RUBRIC_PATH = Path("data/QA/end_to_end_rubrics.jsonl")
REVIEW_PATH = Path("data/QA/end_to_end_benchmark_review.json")
EXPECTED_SOURCE_PATHS = (QUESTION_PATH, ANNOTATION_PATH, RUBRIC_PATH)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_inside(root: Path, relative: Path) -> Path:
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Benchmark path escapes repository root: {relative}") from exc
    return resolved


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number} must contain an object")
        values.append(value)
    if not values:
        raise ValueError(f"Benchmark source is empty: {path}")
    return values


def _unique_by(
    values: list[Any], *, field: str, label: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        key = str(value[field] if isinstance(value, dict) else getattr(value, field))
        if key in result:
            raise ValueError(f"Duplicate {label}: {key}")
        result[key] = value
    return result


def _validate_frozen_sources(root: Path, manifest: BenchmarkManifest) -> None:
    expected = {path.as_posix() for path in EXPECTED_SOURCE_PATHS}
    actual = {source.path for source in manifest.sources}
    if actual != expected:
        raise ValueError(
            f"Benchmark source set mismatch: expected={sorted(expected)}, "
            f"actual={sorted(actual)}"
        )
    for source in manifest.sources:
        path = _resolve_inside(root, Path(source.path))
        if not path.is_file():
            raise ValueError(f"Frozen Benchmark source does not exist: {source.path}")
        if _sha256(path) != source.sha256:
            raise ValueError(f"Frozen Benchmark source hash changed: {source.path}")


def _validate_environment(root: Path, manifest: BenchmarkManifest) -> None:
    for document in manifest.environment.documents:
        path = _resolve_inside(root, Path(document.source_path))
        if not path.is_file():
            raise ValueError(f"Environment document does not exist: {document.source_path}")
        if _sha256(path).lower() != document.document_sha256.lower():
            raise ValueError(
                f"Environment document hash changed: {document.source_path}"
            )


def _rubric(
    annotation: ComponentQuestionAnnotation,
    rubric_source: RubricAnnotation,
) -> BenchmarkRubric:
    answer_units = {unit.unit_id: unit for unit in annotation.answer_units}
    source_ids = [item.answer_unit_id for item in rubric_source.items]
    if source_ids != list(answer_units):
        raise ValueError(
            f"{rubric_source.question_id} Rubric items must map in order to every "
            "reviewed Answer Unit"
        )
    items: list[RubricItem] = []
    for index, source in enumerate(rubric_source.items, start=1):
        criterion = answer_units[source.answer_unit_id].statement
        items.append(RubricItem(
            rubric_id=f"R{index}",
            criterion=criterion,
            weight=source.weight,
            full=ScoreBand(
                ratio=1.0,
                description=(
                    "The answer correctly and completely expresses every actor, "
                    f"condition, action, direction, value, and outcome required by: {criterion}"
                ),
            ),
            partial=ScoreBand(
                ratio=0.5,
                description=(
                    source.partial_credit_guidance
                    or (
                        "The answer states the correct central direction of this "
                        "criterion but omits at least one actor, condition, action, "
                        "direction, value, or outcome explicitly required by it, "
                        "without stating a contradiction."
                    )
                ),
            ),
            none=ScoreBand(
                ratio=0.0,
                description=(
                    "The answer omits this point, merely mentions related terminology, "
                    "states it incorrectly, or directly contradicts it."
                ),
            ),
        ))
    return BenchmarkRubric(items=tuple(items))


def _gold_evidence(record: dict[str, Any]) -> tuple[GoldEvidence, ...]:
    return tuple(
        GoldEvidence(
            evidence_id=f"GE{index}",
            release=record["release"],
            **target,
        )
        for index, target in enumerate(record["evidence_targets"], start=1)
    )


def load_end_to_end_benchmark(root: Path | None = None) -> EndToEndBenchmark:
    """Compose the 50-case Benchmark and fail closed on source drift."""

    repository = (
        root.resolve()
        if root is not None
        else Path(__file__).resolve().parents[3]
    )
    manifest_file = _resolve_inside(repository, MANIFEST_PATH)
    if not manifest_file.is_file():
        raise ValueError(f"Benchmark manifest does not exist: {MANIFEST_PATH}")
    manifest = BenchmarkManifest.model_validate_json(
        manifest_file.read_text(encoding="utf-8")
    )
    _validate_frozen_sources(repository, manifest)
    _validate_environment(repository, manifest)

    question_records = _load_jsonl(_resolve_inside(repository, QUESTION_PATH))
    annotations = [
        ComponentQuestionAnnotation.model_validate(value)
        for value in _load_jsonl(_resolve_inside(repository, ANNOTATION_PATH))
    ]
    rubrics = [
        RubricAnnotation.model_validate(value)
        for value in _load_jsonl(_resolve_inside(repository, RUBRIC_PATH))
    ]
    questions_by_id = _unique_by(
        question_records, field="id", label="Benchmark question ID"
    )
    annotations_by_id = _unique_by(
        annotations, field="question_id", label="component annotation ID"
    )
    rubrics_by_id = _unique_by(
        rubrics, field="question_id", label="Rubric annotation ID"
    )
    expected_ids = {f"r17-qa-{index:03d}" for index in range(1, 51)}
    for label, identifiers in (
        ("questions", set(questions_by_id)),
        ("annotations", set(annotations_by_id)),
        ("rubrics", set(rubrics_by_id)),
    ):
        if identifiers != expected_ids:
            raise ValueError(
                f"Benchmark {label} IDs differ from r17-qa-001..050: "
                f"missing={sorted(expected_ids - identifiers)}, "
                f"unexpected={sorted(identifiers - expected_ids)}"
            )

    environment_documents = {
        (
            item.source_path,
            item.document_sha256.lower(),
            item.specification,
            item.version,
        )
        for item in manifest.environment.documents
    }
    cases: list[BenchmarkCase] = []
    for question_id in sorted(expected_ids):
        record = questions_by_id[question_id]
        annotation = annotations_by_id[question_id]
        rubric_source = rubrics_by_id[question_id]
        if annotation.review_status != "reviewed":
            raise ValueError(f"{question_id} component annotation is not reviewed")
        evidence = _gold_evidence(record)
        for item in evidence:
            identity = (
                item.source_path,
                item.document_sha256.lower(),
                item.specification,
                item.version,
            )
            if identity not in environment_documents:
                raise ValueError(
                    f"{question_id} Gold Evidence is outside frozen Environment: "
                    f"{item.source_path}"
                )
        cases.append(BenchmarkCase(
            question=record["question"],
            reference_answer=record["expected_answer"],
            rubric=_rubric(annotation, rubric_source),
            gold_evidence=evidence,
            environment=manifest.environment,
            task_metadata=TaskMetadata(
                task_id=question_id,
                difficulty=record["difficulty"],
                source=SourceMetadata.model_validate(record["source"]),
                review=BenchmarkReview(
                    reviewed_by=rubric_source.curated_by,
                    reviewed_at=rubric_source.curated_at,
                ),
            ),
        ))
    difficulties = Counter(case.task_metadata.difficulty for case in cases)
    if dict(difficulties) != manifest.difficulty_counts:
        raise ValueError("Composed difficulty counts disagree with manifest")
    return EndToEndBenchmark(manifest=manifest, cases=tuple(cases))


def benchmark_review_file(benchmark: EndToEndBenchmark) -> BenchmarkReviewFile:
    """Build the normalized single-file review projection."""

    manifest = benchmark.manifest
    documents_by_path = {
        document.source_path: document for document in manifest.environment.documents
    }
    review_cases: list[BenchmarkReviewCase] = []
    for case in benchmark.cases:
        relevant_paths = tuple(dict.fromkeys(
            evidence.source_path for evidence in case.gold_evidence
        ))
        review_cases.append(BenchmarkReviewCase(
            task_metadata=case.task_metadata,
            question=case.question,
            reference_answer=case.reference_answer,
            rubric=case.rubric,
            gold_evidence=case.gold_evidence,
            environment=BenchmarkCaseEnvironment(
                environment_id=manifest.environment.environment_id,
                release=manifest.environment.release,
                relevant_documents=tuple(
                    documents_by_path[path] for path in relevant_paths
                ),
            ),
        ))
    return BenchmarkReviewFile(
        benchmark_id=manifest.benchmark_id,
        benchmark_version=manifest.benchmark_version,
        case_count=manifest.case_count,
        trial_count=manifest.trial_count,
        scoring_scope=manifest.scoring_scope,
        difficulty_counts=manifest.difficulty_counts,
        cases=tuple(review_cases),
        frozen_execution_environment=manifest.environment,
    )


def render_benchmark_review(benchmark: EndToEndBenchmark) -> str:
    """Render the deterministic checked-in review file."""

    return benchmark_review_file(benchmark).model_dump_json(indent=2) + "\n"


def validate_benchmark_review(root: Path | None = None) -> BenchmarkReviewFile:
    """Require the materialized review file to match the canonical composition."""

    repository = root.resolve() if root is not None else Path(__file__).resolve().parents[3]
    benchmark = load_end_to_end_benchmark(repository)
    review_path = _resolve_inside(repository, REVIEW_PATH)
    if not review_path.is_file():
        raise ValueError(f"Materialized Benchmark review file does not exist: {REVIEW_PATH}")
    actual = review_path.read_text(encoding="utf-8")
    expected = render_benchmark_review(benchmark)
    if actual != expected:
        raise ValueError(
            "Materialized Benchmark review file is stale; run "
            "uv run python scripts/export_end_to_end_benchmark.py"
        )
    return BenchmarkReviewFile.model_validate_json(actual)


__all__ = [
    "ANNOTATION_PATH",
    "MANIFEST_PATH",
    "QUESTION_PATH",
    "REVIEW_PATH",
    "RUBRIC_PATH",
    "benchmark_review_file",
    "load_end_to_end_benchmark",
    "render_benchmark_review",
    "validate_benchmark_review",
]

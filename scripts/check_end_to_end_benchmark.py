"""Validate the frozen final-answer Benchmark."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from specrag.evaluation import (  # noqa: E402
    load_end_to_end_benchmark,
    validate_benchmark_review,
)


def main() -> int:
    try:
        benchmark = load_end_to_end_benchmark(ROOT)
        validate_benchmark_review(ROOT)
    except (OSError, ValueError) as exc:
        print(f"End-to-end Benchmark check failed: {exc}")
        return 1
    rubric_count = sum(len(case.rubric.items) for case in benchmark.cases)
    evidence_count = sum(len(case.gold_evidence) for case in benchmark.cases)
    print(
        "End-to-end Benchmark check passed: "
        f"{len(benchmark.cases)} cases, {rubric_count} Rubric items, "
        f"{evidence_count} Gold Evidence passages"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

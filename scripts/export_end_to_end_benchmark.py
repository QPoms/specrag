"""Materialize the composed Benchmark as one human-reviewable JSON file."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from specrag.evaluation.benchmark_loader import (  # noqa: E402
    REVIEW_PATH,
    load_end_to_end_benchmark,
    render_benchmark_review,
)


def main() -> int:
    benchmark = load_end_to_end_benchmark(ROOT)
    output = ROOT / REVIEW_PATH
    output.write_text(render_benchmark_review(benchmark), encoding="utf-8")
    print(f"Wrote {len(benchmark.cases)} cases to {REVIEW_PATH.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

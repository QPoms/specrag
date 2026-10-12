"""Execute and score frozen first-turn Researcher planning artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from specrag.evaluation import run_first_turn_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("planning_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--release", type=int, default=17)
    args = parser.parse_args()
    summary = run_first_turn_evaluation(
        args.planning_directory.resolve(),
        args.output_directory.resolve(),
        release=args.release,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

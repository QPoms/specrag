"""Stable console entry point."""

from specrag.interfaces.cli import main
from specrag.interfaces.diagnostics import (
    BaselineQuestion,
    CuratedRouteLabel,
    run_baseline,
    run_planning,
    run_researcher_planning,
    run_routing,
    select_baseline_questions,
)

__all__ = [
    "BaselineQuestion",
    "CuratedRouteLabel",
    "main",
    "run_baseline",
    "run_planning",
    "run_researcher_planning",
    "run_routing",
    "select_baseline_questions",
]


if __name__ == "__main__":
    raise SystemExit(main())

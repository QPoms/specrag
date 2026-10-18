"""Diagnostic and baseline command handlers retained by the product CLI."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import uuid

from pydantic import TypeAdapter

from specrag.orchestration.application import (
    PlanningApplication,
    ResearchApplication,
    ResearcherPlanningApplication,
    RoutingApplication,
    build_planning_application,
    build_research_application,
    build_researcher_planning_application,
    build_routing_application,
)
from specrag.config import REPO_ROOT
from specrag.interfaces.baseline_inputs import (
    BaselineQuestion,
    CuratedRouteLabel,
    QUESTION_SET,
    load_component_route_labels,
    load_route_labels,
    select_baseline_questions,
)
from specrag.models import ResearchBudget, ResearchRouteDecision
from specrag.orchestration import ResearchState
from specrag.prompts import (
    PLANNER_PROMPT_VERSION,
    RESEARCHER_CONTROLLER_PROMPT_VERSION,
)


DEFAULT_RUN_ROOT = REPO_ROOT / "artifacts" / "runs"


def _fixed_route_source(labels: dict[str, CuratedRouteLabel]) -> str:
    sources = {label.source for label in labels.values()}
    return next(iter(sources)) if len(sources) == 1 else "mixed_fixed_annotations"


def _json_state(state: ResearchState) -> dict[str, object]:
    return TypeAdapter(ResearchState).dump_python(state, mode="json")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _planner_metadata(response) -> dict[str, object]:
    receipt = response.receipt
    return {
        "mode": response.mode,
        "contract_version": response.contract_version,
        "prompt_version": response.prompt_version,
        "receipt": receipt.model_dump(mode="json") if receipt else None,
    }


def run_baseline(
    questions: tuple[BaselineQuestion, ...],
    *,
    output_directory: Path,
    application: ResearchApplication | None = None,
    budget: ResearchBudget | None = None,
    release: int = 17,
) -> int:
    app = application or build_research_application(release=release)
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for question in questions:
        thread_id = f"{question.question_id}-{uuid.uuid4().hex}"
        started = datetime.now(UTC)
        try:
            state = app.run(
                research_id=question.question_id,
                question=question.question,
                thread_id=thread_id,
                budget=budget,
            )
            artifact = {
                "schema_version": "specrag.baseline-run/v1",
                "question": {
                    "id": question.question_id,
                    "text": question.question,
                },
                "retrieval_scope": {"release": app.release},
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "thread_id": thread_id,
                "state": _json_state(state),
            }
            summary = {
                "id": question.question_id,
                "status": "completed",
                "stop_reason": state["stop_reason"],
                "evidence_sufficient": state["evidence_sufficient"],
                "tool_calls_used": state["budget"].tool_calls_used,
                "replans_used": state["budget"].replans_used,
            }
        except Exception as exc:  # noqa: BLE001 - persist bounded run failure
            artifact = {
                "schema_version": "specrag.baseline-run/v1",
                "question": {
                    "id": question.question_id,
                    "text": question.question,
                },
                "retrieval_scope": {"release": app.release},
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "thread_id": thread_id,
                "error": f"{type(exc).__name__}: {str(exc)[:1600]}",
            }
            summary = {
                "id": question.question_id,
                "status": "error",
                "error_type": type(exc).__name__,
            }
        _write_json(output_directory / f"{question.question_id}.json", artifact)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    _write_json(
        output_directory / "summary.json",
        {
            "schema_version": "specrag.baseline-summary/v1",
            "questions": summaries,
        },
    )
    return 1 if any(item["status"] == "error" for item in summaries) else 0


def run_planning(
    questions: tuple[BaselineQuestion, ...],
    *,
    output_directory: Path,
    application: PlanningApplication | None = None,
    route_labels: dict[str, CuratedRouteLabel] | None = None,
) -> int:
    """Run Planner from fixed annotations, or Router when none are supplied."""

    app = application or build_planning_application()
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for question in questions:
        started = datetime.now(UTC)
        try:
            if route_labels is None:
                route, planning_response = app.run_with_receipt(
                    research_id=question.question_id,
                    question=question.question,
                )
                route_source = "live_router"
            else:
                try:
                    label = route_labels[question.question_id]
                except KeyError as exc:
                    raise ValueError(
                        "missing fixed route annotation for " + question.question_id
                    ) from exc
                route = ResearchRouteDecision(
                    route=label.route,
                    complexity=label.difficulty,
                    rationale=(
                        "Reviewed development label for Planner evaluation."
                        if label.review_status == "reviewed"
                        else "Proposed component annotation for Planner calibration."
                    ),
                )
                route, planning_response = app.run_with_route_receipt(
                    research_id=question.question_id,
                    question=question.question,
                    route=route,
                )
                route_source = label.source
            plan = planning_response.output
            artifact = {
                "schema_version": "specrag.planning-run/v3",
                "question": {
                    "id": question.question_id,
                    "text": question.question,
                },
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "route_source": route_source,
                "route_annotation": (
                    {
                        "review_status": label.review_status,
                        "reviewed_by": label.reviewed_by,
                        "reviewed_at": label.reviewed_at,
                    }
                    if route_labels is not None
                    else None
                ),
                "route": route.model_dump(mode="json"),
                "planner": _planner_metadata(planning_response),
                "plan": plan.model_dump(mode="json"),
            }
            summary = {
                "id": question.question_id,
                "status": "completed",
                "route": route.route,
                "complexity": route.complexity,
                "task_count": len(plan.tasks),
            }
        except Exception as exc:  # noqa: BLE001 - persist bounded run failure
            artifact = {
                "schema_version": "specrag.planning-run/v1",
                "question": {
                    "id": question.question_id,
                    "text": question.question,
                },
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {str(exc)[:1600]}",
            }
            summary = {
                "id": question.question_id,
                "status": "error",
                "error_type": type(exc).__name__,
            }
        _write_json(output_directory / f"{question.question_id}.json", artifact)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    _write_json(
        output_directory / "summary.json",
        {
            "schema_version": "specrag.planning-summary/v3",
            "route_source": (
                _fixed_route_source(route_labels)
                if route_labels is not None
                else "live_router"
            ),
            "planner_contract_version": PLANNER_PROMPT_VERSION,
            "questions": summaries,
        },
    )
    return 1 if any(item["status"] == "error" for item in summaries) else 0


def run_routing(
    questions: tuple[BaselineQuestion, ...],
    *,
    output_directory: Path,
    application: RoutingApplication | None = None,
    labels: dict[str, CuratedRouteLabel] | None = None,
) -> int:
    """Run Router only, then compare with labels after every model call ends."""

    app = application or build_routing_application()
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for question in questions:
        started = datetime.now(UTC)
        try:
            response = app.run(question=question.question)
            decision = response.output
            artifact = {
                "schema_version": "specrag.routing-run/v2",
                "question": {
                    "id": question.question_id,
                    "text": question.question,
                },
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "route": decision.model_dump(mode="json"),
                "routing_assessment": response.assessment.model_dump(mode="json"),
                "receipt": response.receipt.model_dump(mode="json"),
            }
            summary = {
                "id": question.question_id,
                "status": "completed",
                "route": decision.route,
                "complexity": decision.complexity,
            }
            if labels is not None and question.question_id in labels:
                expected = labels[question.question_id]
                comparisons.append({
                    "id": question.question_id,
                    "expected_route": expected.route,
                    "predicted_route": decision.route,
                    "route_match": expected.route == decision.route,
                    "expected_difficulty": expected.difficulty,
                    "predicted_difficulty": decision.complexity,
                    "difficulty_match": (
                        expected.difficulty == decision.complexity
                    ),
                })
        except Exception as exc:  # noqa: BLE001 - persist bounded run failure
            artifact = {
                "schema_version": "specrag.routing-run/v2",
                "question": {
                    "id": question.question_id,
                    "text": question.question,
                },
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {str(exc)[:1600]}",
            }
            summary = {
                "id": question.question_id,
                "status": "error",
                "error_type": type(exc).__name__,
            }
        _write_json(output_directory / f"{question.question_id}.json", artifact)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    _write_json(
        output_directory / "summary.json",
        {
            "schema_version": "specrag.routing-summary/v1",
            "questions": summaries,
        },
    )
    if labels is not None:
        _write_json(
            output_directory / "comparison.json",
            {
                "schema_version": "specrag.routing-comparison/v2",
                "questions": comparisons,
            },
        )
    return 1 if any(item["status"] == "error" for item in summaries) else 0


def run_researcher_planning(
    questions: tuple[BaselineQuestion, ...],
    *,
    output_directory: Path,
    route_labels: dict[str, CuratedRouteLabel],
    application: ResearcherPlanningApplication | None = None,
    release: int = 17,
) -> int:
    """Generate and validate first-turn tool calls without executing them."""

    app = application or build_researcher_planning_application(release=release)
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for question in questions:
        started = datetime.now(UTC)
        try:
            label = route_labels[question.question_id]
            route = ResearchRouteDecision(
                route=label.route,
                complexity=label.difficulty,
                rationale=(
                    "Reviewed development label for Researcher inspection."
                    if label.review_status == "reviewed"
                    else "Proposed component annotation for Researcher calibration."
                ),
            )
            response = app.run_with_route(
                research_id=question.question_id,
                question=question.question,
                route=route,
            )
            planning = response.planning
            first_turns = [
                {
                    "task_id": task_id,
                    "proposal": proposal.model_dump(mode="json"),
                }
                for task_id, proposal in response.first_turns
            ]
            artifact = {
                "schema_version": "specrag.researcher-planning-run/v1",
                "question": {
                    "id": question.question_id,
                    "text": question.question,
                },
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "route_source": label.source,
                "route_annotation": {
                    "review_status": label.review_status,
                    "reviewed_by": label.reviewed_by,
                    "reviewed_at": label.reviewed_at,
                },
                "route": route.model_dump(mode="json"),
                "retrieval_scope": {"release": app.release},
                "planner": _planner_metadata(planning),
                "plan": planning.output.model_dump(mode="json"),
                "researcher": {
                    "prompt_version": RESEARCHER_CONTROLLER_PROMPT_VERSION,
                    "execution_performed": False,
                    "first_turns": first_turns,
                    "deferred_task_ids": list(response.deferred_task_ids),
                },
            }
            accepted = sum(p.accepted for _, p in response.first_turns)
            summary = {
                "id": question.question_id,
                "status": "completed",
                "complexity": route.complexity,
                "task_count": len(planning.output.tasks),
                "planned_ready_tasks": len(response.first_turns),
                "accepted_first_turns": accepted,
                "deferred_tasks": len(response.deferred_task_ids),
            }
        except Exception as exc:  # noqa: BLE001 - persist bounded run failure
            artifact = {
                "schema_version": "specrag.researcher-planning-run/v1",
                "question": {
                    "id": question.question_id,
                    "text": question.question,
                },
                "retrieval_scope": {"release": app.release},
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {str(exc)[:1600]}",
            }
            summary = {
                "id": question.question_id,
                "status": "error",
                "error_type": type(exc).__name__,
            }
        _write_json(output_directory / f"{question.question_id}.json", artifact)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    _write_json(
        output_directory / "summary.json",
        {
            "schema_version": "specrag.researcher-planning-summary/v1",
            "researcher_prompt_version": RESEARCHER_CONTROLLER_PROMPT_VERSION,
            "execution_performed": False,
            "questions": summaries,
        },
    )
    return 1 if any(item["status"] == "error" for item in summaries) else 0


def _add_question_selection(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--id", dest="identifiers", action="append")
    selection.add_argument("--all", dest="select_all", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="specrag")
    commands = parser.add_subparsers(dest="command", required=True)
    routing = commands.add_parser(
        "route",
        help="classify question difficulty without planning or retrieval",
    )
    _add_question_selection(routing)
    planning = commands.add_parser(
        "plan",
        help="plan reviewed questions without running retrieval",
    )
    _add_question_selection(planning)
    planning.add_argument(
        "--route-annotations",
        type=Path,
        help="component annotation JSONL used instead of QA difficulty labels",
    )
    planning.add_argument(
        "--allow-proposed-route-annotations",
        action="store_true",
        help="allow calibration planning from annotations not yet human-reviewed",
    )
    researcher_planning = commands.add_parser(
        "research-plan", help="inspect Researcher calls without executing them"
    )
    _add_question_selection(researcher_planning)
    researcher_planning.add_argument("--release", type=int, default=17)
    baseline = commands.add_parser(
        "baseline",
        help="run curated development questions through the real workflow",
    )
    _add_question_selection(baseline)
    baseline.add_argument("--max-iterations", type=int, default=3)
    baseline.add_argument("--max-tool-calls", type=int, default=6)
    baseline.add_argument("--max-gap-tool-calls", type=int, default=6)
    baseline.add_argument("--max-replans", type=int, default=2)
    baseline.add_argument("--release", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        questions = select_baseline_questions(
            identifiers=tuple(args.identifiers or ()),
            select_all=args.select_all,
            limit=args.limit,
        )
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        if args.command == "route":
            output_directory = args.output_dir or (
                DEFAULT_RUN_ROOT / "routing" / timestamp
            )
            return run_routing(
                questions,
                output_directory=output_directory,
                labels=load_route_labels(),
            )
        if args.command == "plan":
            output_directory = args.output_dir or (
                DEFAULT_RUN_ROOT / "planning" / timestamp
            )
            route_labels = (
                load_component_route_labels(
                    args.route_annotations,
                    allow_proposed=args.allow_proposed_route_annotations,
                )
                if args.route_annotations is not None
                else load_route_labels()
            )
            return run_planning(
                questions,
                output_directory=output_directory,
                route_labels=route_labels,
            )
        if args.command == "research-plan":
            output_directory = args.output_dir or (
                DEFAULT_RUN_ROOT / "researcher-planning"
                / f"release-{args.release}" / timestamp
            )
            return run_researcher_planning(
                questions,
                output_directory=output_directory,
                route_labels=load_route_labels(),
                release=args.release,
            )
        if args.command != "baseline":
            raise AssertionError(f"unhandled command: {args.command}")
        output_directory = args.output_dir or (
            DEFAULT_RUN_ROOT / "baseline" / f"release-{args.release}" / timestamp
        )
        budget = ResearchBudget(
            max_iterations=args.max_iterations,
            max_tool_calls=args.max_tool_calls,
            max_gap_tool_calls=args.max_gap_tool_calls,
            max_replans=args.max_replans,
        )
        return run_baseline(
            questions,
            output_directory=output_directory,
            budget=budget,
            release=args.release,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"specrag: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


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

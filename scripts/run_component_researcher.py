"""Run Researcher against reviewed GoldResearchTasks for component evaluation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

from pydantic import BaseModel

from specrag.agents.researcher import RegistryResearcherService
from specrag.config import load_config
from specrag.evaluation.component_metrics import load_component_annotations
from specrag.infrastructure.researcher_controller import StructuredResearcherController
from specrag.infrastructure.structured_agent import LLMApiStructuredAgent
from specrag.models import ResearchBudget, ResearchPlan, ResearchRouteDecision, ResearchTask
from specrag.orchestration.adapters import researcher_execution_update
from specrag.orchestration.state import new_research_state
from specrag.retrieval.hybrid import HybridMarkdownRetriever, effective_corpus_scope
from specrag.tools import build_retrieval_tool_registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze Researcher output from reviewed GoldResearchTasks."
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--id", dest="identifiers", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--release", type=int, default=17)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    return parser


def _write_once(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != text:
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _questions(path: Path) -> dict[str, dict[str, Any]]:
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            value = json.loads(line)
            values[str(value["id"])] = value
    return values


def _artifact_summary(path: Path, *, status: str) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8-sig"))
    state = artifact["state"]
    return {
        "id": str(artifact["question"]["id"]),
        "status": status,
        "task_count": len(state.get("research_tasks", ())),
        "evidence_count": len(state.get("candidate_evidence", ())),
        "tool_call_count": len(state.get("tool_calls", ())),
        "task_error_count": sum(
            len(item.get("errors", ()))
            for item in artifact.get("component_task_runs", ())
        ),
    }


def _service(release: int) -> tuple[RegistryResearcherService, object]:
    config = load_config()
    retrieval = config.retrieval.model_copy(update={
        "version_policy": "highest_version_within_release",
        "target_release": release,
    })
    config = config.model_copy(update={"retrieval": retrieval})
    retriever = HybridMarkdownRetriever(retrieval)
    registry = build_retrieval_tool_registry(retriever)
    agent = LLMApiStructuredAgent(
        role="planning",
        reasoning_effort="high",
        config=config,
    )
    return RegistryResearcherService(
        controller_agent=StructuredResearcherController(agent),
        retriever=retriever,
        app_config=config,
        tool_registry=registry,
        max_rounds=3,
        max_total_actions=6,
        hits_per_search=5,
        max_evidence_per_question=12,
    ), effective_corpus_scope(retrieval, retriever)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    annotations = load_component_annotations(args.annotations)
    questions = _questions(args.questions)
    available = set(annotations) & set(questions)
    identifiers = tuple(dict.fromkeys(args.identifiers or sorted(available)))
    missing = set(identifiers) - available
    if missing:
        raise ValueError(f"unknown question IDs: {sorted(missing)}")
    service, scope = _service(args.release)
    summaries = []
    for question_id in identifiers:
        output = args.output_dir / f"{question_id}.json"
        if args.resume and output.is_file():
            summaries.append(_artifact_summary(output, status="resumed"))
            print(json.dumps(summaries[-1]))
            continue
        started = datetime.now(UTC)
        annotation = annotations[question_id]
        tasks = tuple(
            ResearchTask(
                task_id=item.task_id,
                question=item.question,
                objective=item.objective,
                dependencies=item.dependencies,
            )
            for item in annotation.gold_tasks
        )
        state = new_research_state(
            research_id=question_id,
            user_question=str(questions[question_id]["question"]),
            budget=ResearchBudget(max_tool_calls=args.max_tool_calls),
        )
        state["route_decision"] = ResearchRouteDecision(
            route=annotation.expected_route,
            complexity=(
                "simple" if annotation.expected_route == "single_task" else "complex"
            ),
            rationale=annotation.route_rationale,
        )
        state["research_plan"] = ResearchPlan(
            plan_id=f"PLAN-{question_id}", tasks=tasks
        )
        state["research_tasks"] = tasks
        task_runs = []
        for task in tasks:
            execution, mapping = service.run(
                (task,),
                shared_context={
                    "research_id": question_id,
                    "original_question": state["user_question"],
                    # The original question is trusted fixed upstream input.
                    # It may name specifications intentionally omitted from a
                    # minimal GoldResearchTask, so expose it to tool-policy
                    # grounding through the dependency-context channel.
                    "dependency_context": ({
                        "kind": "original_question",
                        "text": state["user_question"],
                    },),
                    "component_evaluation": True,
                },
                max_tool_calls=args.max_tool_calls,
            )
            update = researcher_execution_update(
                state,
                execution,
                query_to_task=mapping,
                task_id=task.task_id,
            )
            state.update(update)
            state["iteration"] += 1
            task_runs.append({
                "task_id": task.task_id,
                "status": execution.status,
                "stop_reason": execution.stop_reason,
                "rounds_used": execution.rounds_used,
                "search_actions_used": execution.search_actions_used,
                "expansion_actions_used": execution.expansion_actions_used,
                "errors": list(execution.errors),
            })
        artifact = {
            "schema_version": "specrag.component-researcher-run/v1",
            "question": {
                "id": question_id,
                "text": state["user_question"],
            },
            "annotation_path": args.annotations.as_posix(),
            "annotation_review_status": annotation.review_status,
            "retrieval_scope": _jsonable(scope),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "component_task_runs": task_runs,
            "state": _jsonable(state),
        }
        _write_once(output, artifact)
        summary = _artifact_summary(output, status="completed")
        summaries.append(summary)
        print(json.dumps(summary))
    if not args.skip_summary:
        _write_once(args.output_dir / "summary.json", {
            "schema_version": "specrag.component-researcher-summary/v1",
            "question_count": len(identifiers),
            "questions": summaries,
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())

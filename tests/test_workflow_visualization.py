from collections.abc import Mapping

from specrag.models import (
    ResearchTask,
    ResearchTaskExecution,
    ResearchWorkerResult,
)
from specrag.orchestration.parallel import build_parallel_research_workflow
from specrag.orchestration.state import ResearchState
from specrag.orchestration.workflow import (
    ResearchNodeHandlers,
    build_single_research_workflow,
)


def _test_node_handler(_state: ResearchState) -> Mapping[str, object]:
    """Compile-only handler: topology tests never execute graph nodes."""

    raise AssertionError("topology inspection must not execute node handlers")


def _test_parallel_researcher(
    _state: ResearchState,
    _task: ResearchTask,
    _execution: ResearchTaskExecution,
    *,
    max_tool_calls: int,
) -> ResearchWorkerResult:
    del max_tool_calls
    raise AssertionError("topology inspection must not execute researcher workers")


def _handlers() -> ResearchNodeHandlers:
    return ResearchNodeHandlers(
        router=_test_node_handler,
        planner=_test_node_handler,
        researcher=_test_node_handler,
        source_validator=_test_node_handler,
        sufficiency_judge=_test_node_handler,
        replanner=_test_node_handler,
        synthesizer=_test_node_handler,
    )


def _topology(graph: object) -> tuple[set[str], set[tuple[str, str, bool]]]:
    drawable = graph.get_graph()  # type: ignore[attr-defined]
    nodes = set(drawable.nodes)
    edges = {
        (edge.source, edge.target, edge.conditional)
        for edge in drawable.edges
    }
    return nodes, edges


def test_single_research_workflow_static_topology() -> None:
    workflow = build_single_research_workflow(_handlers())

    nodes, edges = _topology(workflow.graph)

    assert nodes == {
        "__start__",
        "router",
        "planner",
        "researcher",
        "source_validator",
        "sufficiency_judge",
        "budget_guard",
        "replanner",
        "synthesizer",
        "__end__",
    }
    assert edges == {
        ("__start__", "router", False),
        ("router", "planner", False),
        ("planner", "researcher", False),
        ("researcher", "source_validator", False),
        ("source_validator", "sufficiency_judge", False),
        ("sufficiency_judge", "budget_guard", False),
        ("budget_guard", "researcher", True),
        ("budget_guard", "replanner", True),
        ("budget_guard", "synthesizer", True),
        ("budget_guard", "__end__", True),
        ("replanner", "researcher", False),
        ("synthesizer", "__end__", False),
    }


def test_parallel_research_workflow_static_topology() -> None:
    workflow = build_parallel_research_workflow(
        _handlers(),
        _test_parallel_researcher,
    )

    nodes, edges = _topology(workflow.graph)

    assert nodes == {
        "__start__",
        "router",
        "planner",
        "prepare_executions",
        "dispatch",
        "researcher_worker",
        "fan_in",
        "source_validator",
        "sufficiency_judge",
        "budget_guard",
        "replanner",
        "terminalize_tasks",
        "synthesizer",
        "__end__",
    }
    assert edges == {
        ("__start__", "router", False),
        ("router", "planner", False),
        ("planner", "prepare_executions", False),
        ("prepare_executions", "dispatch", False),
        ("dispatch", "researcher_worker", True),
        ("dispatch", "budget_guard", True),
        ("researcher_worker", "fan_in", False),
        ("fan_in", "source_validator", False),
        ("source_validator", "sufficiency_judge", False),
        ("sufficiency_judge", "budget_guard", False),
        ("budget_guard", "prepare_executions", True),
        ("budget_guard", "replanner", True),
        ("budget_guard", "terminalize_tasks", True),
        ("replanner", "prepare_executions", False),
        ("terminalize_tasks", "synthesizer", True),
        ("terminalize_tasks", "__end__", True),
        ("synthesizer", "__end__", False),
    }

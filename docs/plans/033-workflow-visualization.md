# Workflow visualization

Status: complete

## Goal

Keep an executable, version-controlled view of the active Single Researcher
and Parallel Researcher LangGraph topologies without invoking business
handlers, loading the corpus, or calling a model provider.

## Scope and decisions

- `docs/architecture/single_research_workflow.mmd` is the concise business-flow
  source for the Single Researcher diagram. It intentionally collapses
  internal budget and replan control nodes into the observable Judge branches.
- `docs/notebooks/langgraph_workflows.ipynb` compiles the active graphs with
  handlers that raise if executed, asserts their static topology, displays the
  diagrams, and exports generated PNG/Mermaid artifacts only under ignored
  `artifacts/diagrams/`.
- The Parallel Researcher view is generated from the compiled runtime graph so
  its dispatch, fan-in, guard, and replan nodes remain visible.
- Conditional-edge destination lists are declared explicitly. This preserves
  runtime behavior while allowing LangGraph's static graph renderer to show
  the exact bounded branches instead of an ambiguous all-node fan-out.
- Router-to-Planner is a normal edge because the active production Router
  handler always returns the planning route; provider/runtime failures are
  raised and handled by the existing Run harness rather than represented as a
  graph branch.
- JupyterLab is a development-only dependency. Notebook checkpoints and
  rendered outputs are not versioned.

## Verification

- `tests/test_workflow_visualization.py` asserts the exact nodes, edges, and
  conditional flags for both compiled graphs.
- The canonical `python scripts/check.py` quality gate must pass before this
  change is published.

## Result

The repository now has one documented path for inspecting and exporting both
workflow diagrams, backed by executable static-topology assertions rather than
an independently maintained runtime diagram.

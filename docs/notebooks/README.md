# Development notebooks

These notebooks are executable architecture aids. They compile the active
implementation for static verification. The Single Researcher image renders
the version-controlled business-flow source at
`docs/architecture/single_research_workflow.mmd`; this intentionally hides
internal control nodes while preserving their observable branch conditions.

Install the development dependencies and launch JupyterLab from the repository
root:

```powershell
uv sync --extra dev
uv run --extra dev jupyter lab docs/notebooks/langgraph_workflows.ipynb
```

In JupyterLab, select the project Python kernel and run **Run > Run All Cells**.
The notebook displays both diagrams and writes standalone files to:

```text
artifacts/diagrams/single-research-workflow.png
artifacts/diagrams/parallel-research-workflow.png
```

It also writes `.mmd` Mermaid source files to the same ignored artifact
directory, so the topology remains available if the online PNG renderer is
unreachable.

`langgraph_workflows.ipynb` uses handlers that fail if called. It inspects only
the compiled static topology and never loads the corpus or calls a model. The
notebook assertions remain the implementation-topology contract even though
the primary Single Researcher image is the clearer business view.

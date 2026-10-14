# SpecRAG coding rules

These rules apply to every human- or AI-authored change in this repository.

## Before changing code

- Read `docs/current_status.md` and
  `docs/architecture/package_boundaries.md`.
- For a cross-package feature or architectural change, create or update one
  numbered plan under `docs/plans/`. Small local fixes do not need a plan.
- Extend the existing implementation. Do not create a parallel implementation
  to avoid understanding the current one.

## Package boundaries

- `models`: dependency-free domain and transport models.
- `retrieval`: corpus discovery, indexing, search, fusion, reranking, and
  retrieval planning.
- `tools`: small agent-callable capabilities and their registry.
- `agents`: bounded agent loops; orchestration belongs here, retrieval
  algorithms and provider SDK calls do not.
- `orchestration`: routing and supervision across multiple agents.
- `evidence`: provenance, validation, sufficiency, gap analysis, and synthesis.
- `memory`: private User Memory persistence and lookup.
- `access`: authentication-independent authorization policy and local identity services.
- `specfacts`: reviewed, workspace-scoped specification fact lifecycle and lookup hints.
- `interfaces`: CLI parsing and application command composition.
- `infrastructure`: LLM, embedding, vector-store, and persistence adapters.
- `evaluation`: offline metrics and benchmark runners only.
- `prompts`: versioned prompt text only; provider calls and workflow logic stay
  in their owning packages.

Only create a package when working code needs it. Do not add empty scaffolding.
Dependencies must point inward according to
`docs/architecture/package_boundaries.md`.

## LangGraph runtime boundary

- LangGraph imports belong only in `orchestration`.
- Graph nodes manage state transitions, routing, loops, parallel scheduling,
  checkpoints, and recovery. They call injected business services instead of
  implementing retrieval, evidence, memory, or domain judgment inline.
- Domain, retrieval, tools, agents, evidence, memory, infrastructure, and
  evaluation packages must remain usable without importing LangGraph.
- LangGraph checkpoint state is runtime state. It is never a substitute for
  validated long-term domain memory.

## No version graveyards

- Never add version suffixes such as `_v2`, `_v11`, `_new`, `_old`, `_final`,
  `_backup`, or `_temp` to Python modules.
- Use Git history for implementation versions. Persisted wire formats may have
  an explicit `schema_version`, but each schema has one active implementation.
- When replacing an implementation, update its callers and delete the replaced
  code and tests in the same change.
- There must be one supported CLI and one documented path for each workflow.

## Agent and evidence rules

- Agent loops must have explicit state, bounded turns/tool calls, and recorded
  stop reasons.
- Models may propose queries or hypotheses; only source-backed content may be
  promoted to evidence.
- Every evidence item must retain source path, document hash, and locator.
- Provider-specific code belongs in `infrastructure`; agents depend on stable
  interfaces rather than SDK details.

## Repository hygiene

- Generated indexes, runs, reports, caches, and downloaded model weights never
  belong under `src/` and must remain ignored by Git.
- Tests follow the active package structure. Delete tests for deleted behavior.
- Do not add another dependency manifest; `pyproject.toml` is authoritative.
- Update `README.md` and `docs/current_status.md` when the supported workflow or
  package layout changes.

Before finishing a change, run the canonical quality gate:

```text
python scripts/check.py
```

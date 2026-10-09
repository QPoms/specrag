# Package boundaries

Updated: 2026-10-09

## Design rule

The repository is organized by stable responsibility, not by experiment,
model version, or individual research run. Git records implementation history;
the source tree contains only the active implementation.

## Target package map

```text
src/specrag/
├── models/            shared domain and transport models
├── retrieval/         corpus, indexes, search, fusion, reranking, planning
├── tools/             agent-callable tool interfaces and registry
├── agents/            bounded researcher loops
├── orchestration/     router and supervisor across agents
├── evidence/          validation, sufficiency, gaps, and synthesis
├── memory/            validated long-term knowledge
├── infrastructure/    external service and persistence adapters
├── evaluation/        offline evaluation only
├── config.py
└── cli.py             the single supported product entry point
```

Packages are created only when their first real implementation is added. The
absence of a target package means that capability has not been implemented yet,
not that an empty placeholder should be committed.

## Dependency direction

```text
models
  ↑
retrieval ─────────→ infrastructure
  ↑                       ↑
tools ────────────────────┤
  ↑                       │
agents ───────────────────┤
  ↑                       │
orchestration ────────────┘

evidence → models, retrieval, infrastructure
memory   → models, infrastructure
```

- `infrastructure` cannot import retrieval, agents, tools, evidence, memory, or
  orchestration.
- `retrieval` cannot import agents, tools, evidence, memory, or orchestration.
- `agents` coordinate existing retrieval and tool APIs; they do not implement
  vector search, reranking, or provider clients inline.
- `orchestration` coordinates agents but does not absorb their implementations.

## Current migration debt

The retained research implementation contains four oversized modules. They are
temporarily protected by line-count ratchets in `scripts/check_structure.py`:

- `retrieval/planner.py`
- `retrieval/hypothesis.py`
- `retrieval/hybrid.py`
- `agents/researcher.py`

They may become smaller, but may not grow beyond their recorded ceilings.
Splitting them requires characterization tests first and must preserve one
public implementation rather than creating versioned copies.

## Change protocol

1. Describe cross-package work in one numbered plan.
2. Add or update tests around the behavior being moved.
3. Move code without changing behavior.
4. Run the structure checker and tests.
5. Refactor internals only after the mechanical move is verified.
6. Delete superseded paths and update project status in the same change.

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
├── orchestration/     LangGraph runtime, graph nodes, routing, checkpoints
├── evidence/          validation, sufficiency, gaps, and synthesis
├── memory/            validated long-term knowledge
├── infrastructure/    external service and persistence adapters
├── evaluation/        offline evaluation only
├── prompts/           versioned model instructions without runtime logic
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
- `prompts` contains prompt text and version identifiers only. It does not call
  providers or implement business decisions.
- LangGraph imports are allowed only in `orchestration`. Graph nodes call stable
  application services; they do not contain retrieval, evidence, memory, or
  domain judgment logic.
- Checkpoint state is runtime state owned by `orchestration`. It is distinct
  from validated domain knowledge owned by `memory`.

## Current module boundaries

The retained research implementation no longer has oversized-module
exceptions. Every Python module follows the normal 600-line ceiling enforced
by `scripts/check_structure.py`. Public compatibility imports remain in
`retrieval/planner.py` and `retrieval/hypothesis.py`; implementation history
lives in Git rather than parallel source files.

`retrieval/hybrid.py` has been split into corpus, immutable model, index,
lexical, semantic, fusion, and orchestration modules. The Researcher now has
separate model, state, policy, and bounded-loop modules. Neither needs a
line-count exception anymore. Candidate execution and ranking have also been
extracted from the planner. The answer-hypothesis channel now has separate
contract, execution, and merge-policy modules and no longer needs an exception.
Relation expansion, deterministic curation, evidence selection, and the two
answer-synthesis paths also have explicit modules behind the planner facade.
Version selection is a retrieval concern expressed by one immutable
`CorpusScope` per run. The active scope must propagate into index and execution
artifacts. Uniform Release scope is the default; explicit exact and mixed scopes
must preserve each source document's exact version and Release identity.

The first orchestration implementation is `orchestration/workflow.py`. It owns
the `StateGraph`, conditional edges, budget guard, checkpointer, and recovery
API. Node business behavior is supplied through `ResearchNodeHandlers`.
The evidence path has one deterministic `source_validator` node followed by
one model-backed `sufficiency_judge`: source validation checks provenance and
exact locators, while the judge alone selects supporting evidence and decides
task completeness. A Researcher submission is only a handoff to these nodes,
not a sufficiency verdict.
Framework-independent workflow contracts live in `models/research.py`, and the
tool protocol remains under `tools/`. `memory/service.py` owns validated-only
distillation, search, supersession, conflict, and invalidation rules; its JSONL
adapter lives in `infrastructure/memory_store.py`. Researcher services may use
the memory search interface read-only, but no Researcher path receives a
memory-write capability.

## Change protocol

1. Describe cross-package work in one numbered plan.
2. Add or update tests around the behavior being moved.
3. Move code without changing behavior.
4. Run the structure checker and tests.
5. Refactor internals only after the mechanical move is verified.
6. Delete superseded paths and update project status in the same change.

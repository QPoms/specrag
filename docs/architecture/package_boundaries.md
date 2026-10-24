# Package boundaries

Updated: 2026-10-21

## Design rule

The repository is organized by stable responsibility, not by experiment,
model version, or individual research run. Git records implementation history;
the source tree contains only the active implementation.

## Target package map

The active package responsibilities are:

- `models`: dependency-free identity, Run/event, research, Evidence, SpecFact,
  Core/Topic Memory, corpus-version lifecycle, proposal, maintenance, and
  transport contracts.
- `access`: default-deny policy and local identity/session application service.
- `context`: provider-independent Token budgeting, tool-result presentation,
  message-history compaction policy, checkpoints, and reconstruction.
- `retrieval`: public corpus/version discovery, non-destructive synchronization
  mechanics, indexing, search, fusion, reranking, and retrieval planning.
- `tools`: small agent-callable capabilities and registry.
- `agents`: bounded model-controlled research loops.
- `evidence`: provenance validation, sufficiency, gaps, and synthesis.
- `specfacts`: workspace-scoped proposal, review, lifecycle, and hint lookup.
- `memory`: authorized User/Project Core, Topic recall, reviewable extraction,
  consolidation, pruning, and immutable Run snapshots.
- `orchestration`: LangGraph state, graph scheduling, durable checkpoints,
  runtime controls, Run Harness, and event emission.
- `infrastructure`: provider, vector-store, SQLite, identity, Run/event,
  SpecFact, Memory, corpus-registry, Context Harness persistence, and structured
  model adapters.
- `interfaces`: CLI and HTTP transport adapters, request authentication, error
  mapping, and application composition. Authorization decisions remain in
  `access`; business workflows remain in their owning packages.
- `evaluation`: offline evaluation only.
- `prompts`: versioned prompt text only.

```text
src/specrag/
├── models/            shared domain and transport models
├── retrieval/         corpus, indexes, search, fusion, reranking, planning
├── tools/             agent-callable tool interfaces and registry
├── agents/            bounded researcher loops
├── orchestration/     LangGraph runtime, graph nodes, routing, checkpoints
├── evidence/          validation, sufficiency, gaps, and synthesis
├── memory/            Core/Topic Memory, recall, proposals, maintenance
├── specfacts/         reviewed workspace-scoped specification facts
├── access/            local identity and default-deny authorization
├── context/           request budgets, result paging, compaction policy
├── infrastructure/    external service and persistence adapters
├── evaluation/        offline evaluation only
├── prompts/           versioned model instructions without runtime logic
├── config.py
└── cli.py             the stable local CLI entry point
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
  orchestration. It may implement provider and persistence boundaries declared
  by `context`.
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
  from reviewed specification facts owned by `specfacts` and preferences or
  research context owned by `memory`.

Context-compaction checkpoints are owned by `context`, not LangGraph. They
rebuild an LLM request from a structured summary plus recent canonical
messages. Full message and tool-result records remain durable; Evidence, Run
state, corpus bindings, and Memory stay authoritative in their existing
packages and are never reconstructed from a summary.

## Current module boundaries

The retained research implementation no longer has oversized-module
exceptions. Every Python module follows the normal 600-line ceiling enforced
by `scripts/check_structure.py`. Public compatibility imports remain in
`retrieval/planner.py` and `retrieval/hypothesis.py`; implementation history
lives in Git rather than parallel source files.

`access` depends only on `models`; `specfacts` and `memory` depend on the
trusted policy and pure models. `infrastructure` implements their persistence
protocols without importing those business packages. `interfaces` is the
outer composition and transport layer and may depend on all application
packages. HTTP handlers authenticate transport credentials and construct
trusted contexts, then delegate role decisions to `access` and behavior to the
existing services; they do not implement a parallel permission system. The
structure checker enforces these directions and continues to permit LangGraph
imports only in `orchestration`.

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
The corpus version registry persists discovered/building/ready/active/draining/
failed state and one active pointer per Release/specification. Synchronization
is explicit and non-destructive: historical source documents are retained,
build failure cannot replace an active pointer, and no package performs
automatic physical retirement. Provider discovery and source materialization
are injected at the retrieval boundary; provider SDK/network code belongs in
`infrastructure`. Durable Runs persist exact version bindings before execution.

The shared orchestration primitives live in `orchestration/workflow.py`; the
product research application uses the dependency-aware fan-out/fan-in graph in
`orchestration/parallel.py`. Graph code owns conditional edges, TaskExecution
scheduling, the all-task terminal join barrier, budget guards, checkpointers,
and recovery. Node business behavior is supplied through
`ResearchNodeHandlers`.
The evidence path has one deterministic `source_validator` node followed by
one model-backed `sufficiency_judge`: source validation checks provenance and
exact locators, while the judge alone selects supporting evidence and decides
task completeness. A Researcher submission is only a handoff to these nodes,
not a sufficiency verdict.
Framework-independent workflow contracts live in `models/research.py`, and the
tool protocol remains under `tools/`. `specfacts/service.py` owns the migrated
validated-claim behavior plus pending review. `memory/service.py` owns explicit
Core and Topic writes and immutable snapshots; `memory/recall.py` owns bounded
Core/pinned assembly and local Topic BM25; `memory/proposals.py` applies only
reviewed mutations; `memory/extraction.py` and `memory/maintenance.py` own the
provider-neutral Extractor, consolidation, and capacity policies. Provider
calls and SQLite adapters stay under `infrastructure/`. Configuration kinds
stay out of model context. Every injected entry is marked as untrusted.
Researcher receives authorized, read-only revision snapshots containing Memory
IDs; it does not receive Memory write services. Pending Memory and active
Memory cannot enter Source Validator, Sufficiency Judge, Evidence, or SpecFact
creation as trusted assertions.

`orchestration/checkpointing.py` is the only durable LangGraph checkpointer
adapter. It uses the same local SQLite file as domain stores while preserving
strict deserialization allowlisting. `orchestration/harness.py` authorizes a
Run before resolving its thread, records events and model receipts, applies
deadlines/cancellation, and makes resume side effects idempotent by operation
key. Tests may continue to inject the in-memory checkpointer.

## Change protocol

1. Describe cross-package work in one numbered plan.
2. Add or update tests around the behavior being moved.
3. Move code without changing behavior.
4. Run the structure checker and tests.
5. Refactor internals only after the mechanical move is verified.
6. Delete superseded paths and update project status in the same change.

# Current project status

Updated: 2026-10-10

## Current state

The historical generated runs, benchmark tensors, retrieval indexes, staging
models, analysis outputs, and temporary caches have been removed. The workspace
has been reduced from approximately 23.9 GB to 5.5 GB.

The earlier risk-seed flaw-discovery pipeline and the protocol-context/attack-
chain generation pipeline have also been removed. User questions now enter the
RAG project directly; they are not derived from those upstream research flows.

Removed source areas include the old `run.py`/LangGraph entry, `flow/`,
`nodes/`, `kg/`, `rag/`, `seeds/`, their shared schemas and prompts, and the CR
evaluation package that depended on the attack-chain flow. The old sentence-
level JSONL inputs were removed as well. The Markdown specification corpus was
retained.

The standalone AttackTest IR, the attack-chain/CR evaluator, and the unrelated
B-static source-code analysis and benchmark pipeline have also been removed.
Their CLI registration, Joern adapters, hidden-gold benchmark contracts, and
dataset-specific helper script were removed with them.

The historical attacker-capability validation line has now been removed in
full. This includes all versioned capability agents and validators, attacker
profiles, condition gates, deterministic capability methods, Gold/Shadow/Rel-17
evaluations, and the complete `threatmodel/` experiment tree. The retained
retriever was decoupled from the deleted threat model and now indexes only the
3GPP Markdown specification corpus.

The reusable code has been mechanically migrated out of the ambiguous
`screening/` package:

- retrieval mechanics and planning are under `specrag/retrieval/`;
- the bounded researcher loop is under `specrag/agents/`;
- model and structured-agent adapters are under
  `specrag/infrastructure/`.

The runtime architecture now uses LangGraph only in
`specrag/orchestration/` for graph state, scheduling, routing, loops,
parallel fan-out/fan-in, checkpoints, and recovery. Retrieval, evidence,
memory, evaluation, and domain decisions remain framework-independent.

The repository, distribution, and Python import package now consistently use
the name `specrag`. The retrieval configuration key is `retrieval`, generated
indexes use `.local/indexes/specifications/`, and the old `capability_*`
package and configuration names are no longer active.

The data tree has three explicit responsibilities: `data/3gpp/` retains every
local historical specification version, `data/memory/` stores only validated
long-term knowledge and remains ignored by Git, and `data/QA/` contains the
flat curated development-question file plus ignored upstream snapshots.
Redundant `source_corpus/v0_1/specifications` wrappers and obsolete research
catalogs were removed.

Repository-wide guardrails now live in `AGENTS.md`, package ownership is
documented in `docs/architecture/package_boundaries.md`, and
`scripts/check_structure.py` enforces the main rules automatically.
`python scripts/check.py` is the canonical local quality gate and runs both the
structure contract and the test suite.

## Verified reusable path

The immediate migration target is:

```text
research task
-> retrieval planning
-> hybrid/agentic retrieval
-> source-verified evidence
-> sufficiency/gap analysis
-> bounded re-retrieval
-> grounded answer
```

All migrated modules import successfully. The canonical quality gate now
passes the structure contract, validates every curated question's local source
hash, R17 version, line locator, and exact excerpt, and passes all 80 focused
tests.

The first behavioral refactor is complete. Corpus discovery/chunking, immutable
retrieval contracts, BM25, semantic candidate search, reciprocal-rank fusion,
snapshot persistence, and hybrid orchestration now have separate modules.
Chroma-specific construction is isolated under `infrastructure/`. The bounded
Researcher is split into explicit models, evidence state, action policy, and
loop execution. Its agent-finish, round-budget, and agent-error exits are
covered by characterization tests.

Deterministic planner behavior is now characterized for exact source spans,
explicit facets, unresolved-reference fallback, and explicit-specification
routing. Planning-input, execution-result, evidence-selection, repair, and
answer-summary contracts have been extracted from the monolithic planner.

Question-to-plan compilation and specification routing now live in separate
`planning.py` and `routing.py` modules. The original planner remains the single
compatibility entry point.

Candidate retrieval execution and ranking are now separate, cohesive modules.
Execution owns exact-anchor, semantic, BM25, and evidence-driven reference
searches; ranking owns deterministic scores, optional cross-encoder reranking,
and final hit materialization. A newly characterized exact-anchor path also
exposed and fixed a stale constant reference.

The answer-hypothesis channel is no longer an oversized module. Its validated
contracts, per-requirement retrieval execution, and literal/hypothesis channel
merge policy are separate responsibilities behind the existing
`retrieval.hypothesis` public interface. The merge behavior is characterized
to preserve both route provenances when two channels retrieve the same chunk.

The core refactor milestone is complete. Grounded relation discovery,
deterministic curation, second-stage expansion, source-ID evidence selection,
and answer synthesis now live in cohesive modules behind a small planner
facade. Every retained Python module is within the standard 600-line ceiling,
so the temporary planner and hypothesis exceptions have been removed. Answer
synthesis is characterized to reject model-proposed evidence IDs that are not
present in the supplied source catalog.

Version-aware retrieval now has one explicit per-run `CorpusScope`. The active
scope supports the highest available version inside one Release, latest per
specification, one exact version, or an explicit set of mixed specification
versions. The default project scope remains Release 17. The scope is part of
the index fingerprint and propagates through retrieval, expansion, hypothesis,
and Researcher execution artifacts. Source documents, chunks, retrieval hits,
selected evidence, and tool evidence retain Release, exact version, source
path, document hash, heading path, and line locator.

The retrieval tool milestone is complete. Framework-independent request,
result, evidence, registry, JSON Schema, and capability-description contracts
now exist together with Keyword, Semantic, Hybrid, Reranker, Clause, and
same-section adapters. Every emitted evidence item preserves its active corpus
scope, Release, exact version, source path, document hash, heading path, and
line locator. The bounded Researcher now invokes these capabilities only through
the Tool Registry; its old direct retrieval branch has been removed. All six
tools are verified independently from the agent loop.

The LangGraph runtime lives under `specrag/orchestration/`.
It defines one unified `ResearchState`, injected handlers for Router, Planner,
Researcher, Validator, Sufficiency Judge, Re-planner, and Synthesizer nodes, a
budget guard, conditional re-planning, an in-memory checkpointer, and thread
resume helpers. Framework-independent research task, evidence, gap, tool-call,
budget, and stop-reason contracts live under `specrag/models/`. LangGraph
1.2.14 is recorded in `pyproject.toml` and `uv.lock`. The repository-local
Python 3.12 environment is reproducible with `uv sync --frozen`; a minimal
runtime check confirms graph compilation, bounded termination, and checkpoint
persistence. A graph-state adapter now copies registry-backed Researcher hits,
tool-call audit records, budget consumption, errors, and worker stop reasons
into `ResearchState` without performing validation or sufficiency decisions.
The Single Researcher handler selects dependency-ready work and respects the
global tool budget. The parallel workflow dynamically fans independent tasks
out with LangGraph `Send`, gathers isolated worker results through one reducer,
and then enters the shared validation/sufficiency path. Dependent tasks remain
ineligible until their prerequisites complete. When they become eligible, both
Researcher paths receive direct predecessor task metadata, sufficiency rationale,
and only evidence selected by a sufficient assessment and validated as
source-backed support. Dependency evidence keeps its original task ownership
and scopes downstream retrieval instead of being relabelled as new evidence.
A `missing_relation` test
confirms the conditional Gap → Re-plan → Researcher → Synthesis path.

## Known limitations

- Local model weights remain under `models/` but are ignored by Git.
- No supported end-to-end product CLI has been selected yet.
- Evidence validation, sufficiency, conflict handling, and grounded synthesis
  are now clean product modules. Long-term Memory has a validated-only
  distillation/store/search path with JSONL persistence, supersession, conflict,
  invalidation, time-based expiry, and read-only Researcher context.
- Cross-version retrieval now preserves two isolated Release executions, but
  claim-level change synthesis remains part of the future evidence system.
- The parallel Researcher path is implemented, but its complete multi-hop
  dependency lifecycle with real model and retrieval services still needs a
  dedicated end-to-end test; handler-level dependency handoff is covered.
- The README and package metadata now describe the transition toward the new
  Research Agent, but the final package layout and public CLI are not in place.

## Next milestone

Validate the workflow one stage at a time. Router now receives question text
only and emits `single_task/simple` or `decompose/complex` plus one concise
rationale. It does not extract goals, infer specification topology, answer the
question, or select a Release. Goal extraction and task counts belong to Planner;
Release is applied later as a retrieval-run parameter.
After Router behavior is accepted, inspect Planner decomposition before enabling
retrieval. All 50 current questions are development data; a fresh external test
set will be collected after the complete workflow is optimized. Formal benchmark
design remains intentionally deferred to the project owner.

The first revised Router completed all 50 curated questions without schema or
runtime errors and achieved 29/40 development-label matches. After defining a
goal as one cohesive retrieval-and-evidence chain, the second 50-question run
improved to 34/40. It corrected all 11 simple-to-complex errors, but over-grouped
six complex questions as simple. Five contain distinct stages, actor decisions,
or requirement families that still need separate research. One EPS-detach case
is complex only because its curated evidence later spans specifications, which
a question-only Router cannot observe without violating its agreed boundary.
The active v3 contract now removes goal extraction entirely and emits only
`route`, `complexity`, and `rationale`; Planner owns all decomposition. Its
50-question run matched 31/40 existing development labels. Review confirms that
the label set partly conflates pre-retrieval task decomposition with
post-retrieval source topology. Planner execution remains paused until the
development routing labels are adjudicated using the new question-only
definition. These metrics are retained as historical diagnostics from before
the labels were separated.

The active Router prompt and version identifier now live in
`specrag/prompts/router.py`; the infrastructure service imports that contract
instead of embedding prompt text. A uniform question-text-only review draft now
covers all 50 questions and proposes 31 `single_task` and 19 `decompose` routes.
It differs from the earlier difficulty field on three cases.

The project owner accepted the uniform 50-question review. After focused
boundary review, the curated data now contains 34 `simple` and 16 `complex`
labels, all under the development split. Question 017 is one NAS-security
objective; question 035 is one connected NH-handling chain; and question 044 is
one mobility-setting negotiation with acceptance/refusal branches. Multiple
source clauses or specifications do not create independent research objectives
by themselves.
The expected Router action is derived from difficulty rather than stored as a
duplicate field. Routing difficulty remains independent of `reasoning_scope`,
so a single task may legitimately require cross-specification evidence. The
temporary internal holdout policy has been retired.

The simplified `research-router/v3.1` prompt was then run against all 50
reviewed questions. All calls completed with valid structured output. It matched
41 labels (82%), predicting 34 simple and 16 complex questions. Six complex
labels were grouped as simple and three simple labels were decomposed. Review of
the rationales shows a remaining contract ambiguity: several labels call a
multi-stage or cross-boundary question complex even when it follows one entity,
message, or procedure, while the prompt explicitly calls such a cohesive chain
simple.

The `research-router/v3.3` contract defined complexity by independent
research objectives. One coherent objective supported by one primary evidence
chain is simple. Complex questions require at least two objectives to be
retrieved and validated independently before synthesis. Explicit boundaries
across specifications, protocol layers, procedures, stages, interfaces, or
actor responsibilities favor decomposition only when they create those
independent objectives. Multiple clauses or sources in the eventual answer do
not retroactively make a question complex. Planner and retrieval have not been
run on this dataset.

The complete v3.3 Router run produced valid output for all 50 questions and
predicted 34 simple and 16 complex. It originally matched 42 labels. After
questions 017 and 044 were adjudicated as simple, the same outputs match 44/50
(88%). Three simple labels (014, 019, and 029) were decomposed, while three
complex labels (040, 042, and 046) were grouped as single tasks.

The rejected `research-router/v3.4` prompt added an explicit decomposition test.
Causes, consequences, branches, internal mechanisms, and consecutive steps of
one governing rule remain one task. Parts become complex only when each can be
stated and validated as a standalone normative research question and their
findings must then be integrated.

The complete v3.4 run produced valid output for all 50 questions but regressed
to 37/50 (74%), predicting 45 simple and only 5 complex. It corrected simple
questions 014 and 029, but grouped 12 complex questions as single tasks. The
generic instruction to keep consecutive steps of one end-to-end flow together
was too broad and overrode explicit cross-specification, layer, interface, and
actor-responsibility boundaries. V3.4 is therefore rejected as the Router
baseline. The best result remains the v3.3 output rescored against the latest
labels at 44/50 (88%). Planner execution remains paused.

The implemented `research-router/v3.5` keeps the public Router contract at
exactly `route`, `complexity`, and `rationale`, while using one structured
model call to assess a bounded `independent_goal_count` and whether the goals
form a `single_chain` or `require_synthesis`. Code maps that assessment
deterministically to the public route. The internal assessment is stored only
in ignored Router-run artifacts for diagnosis; it does not enter
`ResearchState` and does not construct Planner tasks. The full quality gate
passes.

The complete v3.5 run produced valid structured output for all 50 questions
and matched 43 labels (86%), predicting 39 simple and 11 complex questions.
It produced one simple-to-complex error (019) and six complex-to-simple errors
(040, 042, 045, 046, 049, and 050). Compared with v3.3, it corrected 014 and
029 but newly grouped 045, 049, and 050 as single tasks. The bounded internal
goal count therefore improves diagnosis but does not by itself improve routing
accuracy. V3.3 remains the best observed result at 44/50 (88%), and Planner
execution remains paused.

The `research-router/v3.6` prompt redefines an internal goal as a
planner-separable normative conclusion rather
than a logically independent objective. Related conclusions may have causal or
task dependencies and still require decomposition. The prompt now distinguishes
one assertion and its direct protocol representation from separate definitions,
procedures, actor responsibilities, interface contracts, stages, and requirement
families. It also prevents a shared topic or end-to-end narrative from serving
as sufficient reason to merge separate conclusions. The public three-field
Router output and the artifact-only diagnostic schema are unchanged.

The complete v3.6 run produced valid structured output for all 50 questions
and matched 38 labels (76%), predicting 22 simple and 28 complex questions.
It corrected every v3.5 complex-to-simple error (040, 042, 045, 046, 049, and
050) and kept 019 simple, but introduced 12 simple-to-complex errors (002, 004,
005, 017, 020, 023, 025, 026, 027, 034, 036, and 037). No complex question was
classified as simple. The prompt therefore over-corrected: it treats separately
verifiable attributes or slots within one normative contract, such as purpose
and format or inputs and outputs, as separate Planner tasks. V3.6 is rejected
as the Router baseline; v3.3 remains the best observed result at 44/50 (88%).

The prepared `research-router/v3.7` adds a contract-cohesion boundary to v3.6.
Purpose/format, scope/value meaning, inputs/outputs, reported content/required
fields, initiators/purposes, and state/persistence remain one goal when they are
slots of one named normative object that one retrieval context can answer.
Complexity now requires separate research entry points or normative contexts,
while retaining the v3.6 boundaries for different actors, interfaces, stages,
and requirement families from distinct profiles or rule sets. The acceptance
target is at least 84% overall accuracy; a limited simple-to-complex bias is
acceptable in order to reduce missed complex questions. The reviewed prompt
also limits the consecutive-step exception to one locally governed rule,
qualifies actor and state slots so they cannot hide separate responsibilities,
and requires different research entry points or normative contexts in its final
decomposition test.

The complete v3.7 run produced valid structured output for all 50 questions
and matched 47 labels (94%), exceeding the 84% acceptance target. It predicted
31 simple and 19 complex questions. All 16 complex labels were identified
correctly; the only errors were three accepted conservative decompositions of
simple questions (017, 019, and 034). There were no complex-to-simple errors.
V3.7 is therefore the accepted Router baseline, and prompt optimization is
paused so the workflow can proceed to Planner inspection.

The unchanged `research-planner/v1.0` instructions now live under
`specrag.prompts`; code enforces two to four ordered `T1...Tn` pending tasks,
and the existing model contract rejects invalid dependencies and cycles.
Release remains outside Router and Planner inputs and is bound only at
retrieval time. The first Planner baseline completed for all 16 reviewed
complex questions using their accepted v3.7 decisions. Eight plans contained
three tasks and eight contained four. All 16 added a terminal integration task
dependent on the preceding factual tasks. That consistent behavior exposes the
main v1.0 failure mode: the prompt asks for retrieval objectives while also
encouraging an integration task, leading the model to create synthesis-oriented
work that belongs after evidence collection rather than in the retrieval task
DAG. The v1.0 result is retained as a diagnostic baseline and is not accepted.

The prepared `research-planner/v1.2` replaces the v1.0 decomposition text with
a distinct-research-objective contract. It retains relationship tasks when the
relationship itself requires research or validation, while avoiding an
automatic integration task when the established findings already answer the
connection. Dependencies now mean validated information flow rather than answer
presentation order. Tasks must be emitted in dependency order, and the domain
contract rejects dependencies that do not reference earlier tasks. The revised
prompt was run on all 16 reviewed complex questions using their dataset labels
directly, without invoking Router. All 16 calls completed and passed the
structural contract. The run produced 45 tasks: three questions had two tasks
and thirteen had three, compared with 56 total tasks under v1.0. The automatic
terminal integration task disappeared while the substantive question parts
remained represented. Six questions contained nine dependency-bearing tasks.
Some are clear information dependencies, while others may still encode
procedural or presentation order and require focused review before v1.2 is
accepted.

The staged `specrag plan` command now uses reviewed difficulty labels as its
branch decision and records `route_source=reviewed_label`; it no longer reruns
Router during Planner experiments. This keeps Router nondeterminism separate
from Planner behavior while continuing to exclude Release, expected answers,
facts, and evidence targets from Planner model input.

The implemented `research-planner/v1.3` keeps the 2-4 task contract while
constraining every task to the minimum scope needed by the original question.
It defines dependencies as completed, validated predecessor findings supplied
to a downstream Researcher, and explicitly excludes protocol sequence,
procedural order, shared terminology, answer order, and later synthesis as
sufficient reasons for a dependency. Relationship tasks remain allowed only
when the relation requires its own source-backed research or validation. The
focused v1.3 run completed on all six questions that carried dependencies under
v1.2. It reduced their tasks from 18 to 14 and dependency-bearing tasks from
nine to two. Questions 008, 042, 043, and 047 now contain only independent
tasks. Question 040 retains one source-backed decision-to-service mapping task
dependent on its two inputs, while question 048 retains one content-evolution
task dependent on the signaling-path result. The generated scope is materially
narrower. The unchanged prompt then completed the full 16-question acceptance
run with valid structure for every question, producing 38 tasks: ten two-task
plans and six three-task plans. No generic terminal synthesis task or substantive
coverage omission was found. Only questions 008, 045, and 048 retained
dependencies, representing a cross-specification relationship, a successful
continuation branch, and path-dependent content transformation respectively.
V1.3 is therefore the accepted and frozen Planner baseline. The 045 dependency
remains an observation point for real Researcher execution rather than a reason
to revise the prompt.

Planner execution metadata now survives the service boundary. The existing
`plan()` method remains the runtime-compatible plan-only interface, while
`plan_with_receipt()` returns a structured envelope containing the validated
plan, deterministic-or-model mode, active contract version, prompt version, and
model receipt. Planning-run artifacts use schema v3 and persist that envelope;
simple questions record deterministic mode with no prompt receipt because no
Planner model call occurs.

The deterministic simple path has now been exercised on all 34 reviewed simple
questions. Every case produced exactly one dependency-free task and preserved
the original question verbatim. This includes the four reviewed simple cases
whose evidence targets span more than one clause or specification: their
question still expresses one cohesive research objective, so source count does
not justify a second high-level Planner call. These tasks proceed directly to
Researcher, where the existing constrained retrieval-planning pass extracts
question-grounded protocol anchors before search. The simple branch is therefore
accepted without an additional task-decomposition model call.

Dependency scheduling now distinguishes normal plan execution from corrective
re-planning. Pending tasks are not assessed before their dependencies complete,
so they do not create synthetic evidence gaps. When a predecessor becomes
sufficient, its newly ready dependent task proceeds directly to the next
Researcher wave without consuming `replans_used`. Gap Re-planner is reserved for
tasks that have actually run and remain incomplete or blocked, and it reopens
only tasks named by unresolved gaps.

The live eight-stage progress checklist is maintained in
`docs/plans/003-optimization-roadmap.md`.

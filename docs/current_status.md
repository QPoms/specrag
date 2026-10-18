# Current project status

Updated: 2026-10-17

## Current state

The current internship-project scope is frozen around the local 3GPP research
workspace: authenticated multi-user API, durable Run/Worker lifecycle,
evidence-backed answers, reviewed Memory/SpecFacts, and the browser client.
No new corpus provider, private-document upload, external broker, production
deployment layer, or automatic whole-Run replay is part of the supported
workflow. Remaining work is hardening, documentation, reproducibility, and
answer evaluation rather than another product slice.

The first specification-version synchronization slice is implemented as an
explicit, dormant-by-default maintenance capability. SQLite now records every
discovered immutable specification document, its build lifecycle, the active
pointer for each Release/specification, and exact Run bindings. Local history
is append-only: activating a new version marks the old one `draining` and does
not delete its source or index. A failed document build leaves the prior active
version unchanged.

`specrag corpus sync` previews local version changes by default; `--apply` is
required to perform the per-document chunk/BM25/vector build, completeness
validation, and activation. There is no background scheduler and no automatic
retirement or garbage collection. Provider discovery and source materialization
have injected contracts. The official adapter under
`infrastructure/threegpp.py` reads the 3GPP FTP archive, downloads and safely
extracts one ZIP, converts DOC/DOCX to Markdown, and reuses the existing
build-before-activate flow. It is only reachable through the explicit
`specrag corpus sync --official` command; the default command still scans local
Markdown. Supported provider Releases are configured as 17, 18, and 19.

Semantic vectors use one append-only cache per embedding/chunk configuration.
Each ingestion adds only missing chunk IDs; historical vectors remain available
for pinned Runs. Dense queries always filter this shared cache by the exact
document hashes in the Run scope, so retaining old versions cannot leak them
into a newer snapshot and changing one active version does not re-embed every
unchanged document.

New Research Runs freeze the exact per-specification versions selected at
start. Once the active registry completely covers a Release, it is the source
of that snapshot; until then the existing highest-local-version behavior is
used. Resume reconstructs retrieval from the stored exact binding, so an active
pointer change cannot mix versions inside a Run. The Run wire contract is now
`research-run/v1.1`, while persisted `research-run/v1` payloads remain readable.

The multi-user local workspace milestone is implemented. The former
`ValidatedKnowledge` and `ValidatedMemoryService` implementation has been
mechanically migrated to `SpecFact` under `models/specfacts.py` and
`specfacts/service.py`; the replaced modules and names were deleted. A
SpecFact begins as an idempotent pending proposal created by an independent
distiller from a completed answer and source-valid evidence. Final synthesis no
longer creates claims or automatically creates SpecFact proposals. Workspace
reviewers approve or reject independently extracted proposals.
Approved facts retain Release, exact version, source path, document hash,
heading, and line locator and can only guide a fresh source lookup. They never
become final Evidence directly.

`memory/` now implements a compact Core/Topic design. User Core remains private
and stores typed configuration, preferences, research habits, followed topics,
and saved leads. Project Core and Topic Memory use the existing Workspace as
their project boundary. Core and pinned Topics are assembled by default within
entry/character budgets; non-pinned Topics are ranked by a local BM25 path and
at most three are selected. SQLite is authoritative; there is no parallel
Markdown store.

Explicit writes retain stable content IDs, immutable predecessor chains,
operation keys, expiry, retraction, and archival. Structured Memory Extractor
output creates pending add/supersede/retract/create-topic proposals. A user
approves their private proposal; a workspace reviewer/admin approves a Project
proposal. Consolidation requires both 24 elapsed hours and 10 accepted changes,
can merge only within one scope/Topic/kind, cannot introduce a source reference,
and preserves all input IDs. Capacity maintenance runs separately and proposes
reversible archival only for non-pinned Topic entries.

Configuration still resolves command argument, active User Core, then
`config.yaml`. A Run records the exact SpecFact and Memory revision IDs used at
start; resume loads only those authorized historical revisions, so later edits,
pin changes, consolidation, or archival do not alter the Run. Workspace admins
can operate a member Run but cannot browse that member's User Core. Every entry
injected into Researcher is marked as untrusted context and cannot enter
Evidence, Source Validator, Sufficiency Judge, or SpecFact proposal logic.

Local `scrypt` accounts, 12-hour sessions, workspaces, and member/reviewer/admin
roles are active through the `specrag` CLI and the first authenticated HTTP API
slice. SQLite state lives at
`.local/state/specrag.sqlite3`, uses foreign keys and WAL, and applies monotonic
`schema_migrations`. The database stores only token SHA-256 digests; the local
CLI bearer token is kept in ignored `.local/state/session-token`. Authorization is
default-deny and occurs before a Run's checkpoint thread is resolved. Members
manage their own Runs and memories, reviewers review their workspace's facts,
and admins manage members and all workspace Runs. The API accepts only an
explicit bearer session, reconstructs the trusted actor and current membership
on every workspace request, rejects client-supplied identity/role fields, and
maps inaccessible and unknown Runs to the same response. It exposes
login/logout, workspace listing/creation and context, member administration,
Run inspection/cancellation, and workspace-scoped Memory/SpecFact dashboards
and proposal review. User Core responses are always restricted to the current
user; Project/Topic Memory and SpecFacts remain restricted to the current
Workspace, and review actions retain the existing owner/reviewer/admin policy.
Workspace admins can list the member directory, create or attach local
accounts, change other members' roles, and remove other memberships. The
identity service rejects self-demotion and self-removal, while non-admin roles
cannot read the member directory.
API research submission now persists a queued
Run and returns immediately; an in-process background Worker executes the same
RunHarness used by the CLI. Authorized clients can consume ordered durable Run
events through an SSE endpoint, and terminal streams close after the final
event. The same Worker now queues checkpoint Resume jobs and fresh Retry jobs.
Resume keeps the Run ID and its frozen context, clears the prior terminal
timing/cancellation markers, grants a new elapsed-time window, and is rejected
unless a durable checkpoint exists. Retry creates a new Run with the original
research parameters, a fresh context snapshot, and a persisted
`retry_of_run_id` link. Resume scheduling is de-duplicated per Run/action and
terminal event operation keys include the resume attempt, so a resumed Run's
final SSE event is not collapsed into its earlier terminal event. Domain
SQLite access now uses one connection per API/Worker thread over the same WAL
database, avoiding unsafe concurrent use of one connection while retaining
transaction boundaries and foreign-key enforcement. Run jobs now live in a
durable SQLite queue with atomic leases, heartbeats, expiry recovery, and
startup repair for queued or interrupted Runs. The API can embed this Worker
for local development, or run in external-worker mode beside the standalone
`specrag-worker` process; both modes use the same RunHarness and queue. A
broker such as Redis remains a later scale-out option. Anonymous bootstrap,
credential cookies, permissive
CORS, and private document upload are not exposed. Workspace members read
Project/Topic Memory and submit proposals; reviewers/admins publish it, manage
pinning, and approve maintenance proposals.

The durable `RunHarness` provides `start`, `resume`, `inspect`, and `cancel`.
It uses `langgraph-checkpoint-sqlite` with the existing strict serializer
allowlist, opaque Run/thread IDs, elapsed-time and cancellation checks at graph
nodes and Researcher rounds/tool calls, stable operation keys, redacted model
receipts, and per-stage latency/token inspection. Events cover transitions,
task dispatch, tool calls, evidence selection, source validation, Judge, Gap,
replan, synthesis, resume, cancellation, and stop. The runtime state wire
contract is now `research-state/v1.6`.
Outbound model API failures use a safe taxonomy covering transient, rate-limit,
timeout, provider-unavailable, authentication/configuration, invalid-request,
structured-output, and internal/unknown failures. The LLM adapter retains
bounded request-level retries; Harness persists only the failure kind,
retryable flag, attempt count, and provider status code, and emits the same
redacted metadata in the terminal event. Whole-Run automatic replay remains
disabled; the API can show whether a manual Retry is appropriate. HTTP errors
now also carry a stable machine-readable `code` alongside `detail`, request
validation uses `validation_error`, and Run submission conflicts are mapped
explicitly instead of escaping as an unclassified server error.

The top-level `cli.py` is a small stable entry. Product command handling lives
under `interfaces/`, with `auth`, `workspace`, `research`, `specfact`, and
`memory` command groups. The same package now owns the FastAPI transport and
`specrag-api` entry point while delegating all authorization and business work
to the existing services. Existing route/plan/research-plan/baseline
diagnostics remain on the same CLI. The public 3GPP corpus remains the only
supported document corpus in this milestone; private-document upload, rate
limiting, and production deployment are still outside the active scope.

The first local browser client is now implemented under `frontend/` as a React
+ TypeScript + Vite application. It uses the API-only security boundary, keeps
the server-issued bearer token in `sessionStorage`, and provides a responsive
login view, workspace switcher, research-desk shell, async Run submission
entry point, and an authenticated Fetch-based SSE reader for live Run stages.
It does not access SQLite directly. Completed Runs now expose only supported
evidence cited by the final answer through the authorized inspection response,
and the
research desk renders the directly generated answer, inline evidence IDs,
source path, exact version,
heading, line locator, excerpt, and document hash. The knowledge-management
surface now shows private User Core, shared Project Core, Topic Memory,
reviewable Memory proposals, approved SpecFacts, SpecFact proposals, source
locators, and review status. Authorized users can approve or reject pending
proposals in place, with a mandatory reason for rejection. An admin-only
People & Permissions panel now shows the workspace directory and role model,
creates local members, changes roles, removes memberships with confirmation,
and protects the signed-in administrator's row. Non-admin users see an
explicit locked state without requesting directory data. Opening a Run now
shows lifecycle metrics, checkpoint availability, recent durable events, a
redacted failure summary, and contextual Cancel, Resume, or Retry controls.
Cancellation remains cooperative; Resume continues the same checkpoint; Retry
creates and follows a new linked Run. Run history is now loaded from the
authorized SQLite-backed API after browser refresh or workspace switching.
Members and reviewers see their own Runs; admins see all Runs in the selected
workspace. Persisted active Runs reconnect to the SSE monitor, and historical
results can be reopened from the research desk.

Focused tests cover password verification, expired/revoked sessions, role
checks, cross-workspace denial, typed and immutable memory revisions, exact
memory snapshots, configuration precedence, SpecFact
proposal/review/conflict/supersession/invalidation,
idempotent Run events, cancellation, elapsed deadlines, interrupted-process
resume, SQLite checkpoint restart, specification synchronization, and exact
Run corpus pinning, API authentication, membership revalidation, role denial,
hidden cross-owner/cross-workspace Run access, member-directory authorization,
self-demotion/removal protection, asynchronous Run submission, queued Resume,
linked Retry, attempt-aware terminal events, safe failure projection, worker
execution, SSE isolation, evidence filtering, knowledge visibility, and
proposal review authorization. The complete suite now contains 204 tests. The
historical 2026-10-13 quality batch remains 47/50; focused
reruns, including 003, 044, and 050, are regression observations and are not
combined into a 48/50 claim.

Final synthesis now returns one complete user-facing answer directly, together
with the ordered evidence IDs cited inline as `[evidence_id]`. The service
rejects citations outside the Judge-supported set, mismatches between inline
markers and declared IDs, and citation sets that omit a sufficient
ResearchTask. The separate synthesis-claim representation and claim-joining
answer derivation have been removed for both simple and complex questions.
Run inspection and the browser expose only the cited supported evidence.
Historical claim-based JSON evaluation artifacts remain readable. SpecFact
distillation is now an independent reviewed operation and is no longer invoked
automatically at Run completion.

A live direct-answer acceptance run completed simple question `r17-qa-001`
and complex question `r17-qa-008` against the local Release 17 corpus. Both
Runs stopped normally with sufficient evidence, zero replans, and zero errors;
the simple case used one ResearchTask and the complex case covered all three
ResearchTasks. Both persisted synthesis objects contain only `answer` and
`evidence_ids`. Deterministic evaluation reported target recall 1.0 and
citation validity 1.0 for both answers. The same run also completed the local
Chroma embedding cache at 27,858 chunks, so later Runs over this unchanged
corpus snapshot reuse the populated vector cache.

The next active milestone is answer evaluation only. The deterministic offline
runner `scripts/evaluate_answers.py` aligns final-answer citations and evidence
to the curated QA targets and reports citation validity, citation relevance,
supported-claim rate, target recall, and all-target coverage by difficulty and
reasoning scope. The existing structured LLM rubric remains optional and is
reported separately. The first 50-question deterministic baseline has now
been run against the retained native-tool-calling answer batch from
`artifacts/runs/full-workflow/20261013-native-tool-calling-50`; the report is
under `artifacts/evaluations/20261017-native-tool-calling-50/summary.json`.
All 47 completed answers had non-empty answers, citation validity 1.0, and
claim support rate 1.0. Target recall was 0.9362 and all-target coverage was
0.8936 across the 47 completed answers; three Runs were skipped because they
ended at a tool or replan budget before synthesis. This is a retrospective
baseline over existing generated artifacts, not a regenerated current-model
run. Runtime performance, Worker throughput, and Memory ablation remain
excluded from this first answer benchmark.

The component-evaluation metric contract is documented in
`docs/plans/031-component-evaluation-system.md`, and its shared implementation
is complete for the 50-question Release 17 development set. It isolates each component with reviewed upstream
inputs and deliberately keeps a small engineering scorecard: Router route
accuracy; Planner task coverage, overlap, and out-of-scope rate; fixed-request
retrieval target Recall@5/@12 and nDCG@5/@12; Researcher evidence coverage and
precision; whole-question coverage, precision, and overlap for the merged
Judge-supported Final Evidence; and Synthesizer answer coverage, groundedness,
and claim-level citation correctness. Humans approve gold ResearchUnits,
EvidenceUnits, AnswerUnits, retrieval qrels, and gold-sufficient Judge inputs;
structured LLM judges may perform repeatable semantic mappings, with initial
and ambiguous judgments reviewed by a human. Judge sufficiency classification,
gap diagnosis, corrective-retrieval recovery, and end-to-end performance are
outside this first component slice.

Execution is staged rather than starting with all 50 questions. The first
reviewed pilot fixes ten Router cases (`001`, `002`, `005`, `019`, `034`,
`007`, `008`, `035`, `044`, and `046`) and evaluates Planner on the five
reviewed decomposition cases. The set intentionally includes two frozen-v3.7
over-decomposition disagreements and two under-decomposition disagreements.
The versioned shared annotation models, metric functions, and the single
`scripts/evaluate_components.py` entry point are implemented. The frozen Router
aligns on 6/10 cases (accuracy and Macro-F1 0.60). Planner over 007, 008, 035,
044, and 046 reports full unit coverage with no judged overlap or out-of-scope
tasks. All ten annotations and five Planner judgments are reviewed, and both
formal reports state `calibration_only=false`.

The evidence-path metric slice is also implemented through the same runner.
Its first reviewed baseline uses 001, 007, and 008 so the sample contains a
single-task case, a multi-task case, and a cross-specification multi-passage
chain. Three fixed ranked calls report Target Recall@5/@12 of 0.6667 and
nDCG@5/@12 of 0.7925. Four fixed Researcher tasks report Evidence Coverage
1.0 and Evidence Precision 0.675. Three whole-question Judge selections report
Final Evidence Coverage 1.0, Precision 0.8, and Overlap 0.2 after exact
cross-task provenance deduplication. Three fixed-evidence Synthesizer cases
report Answer Coverage 0.9583, Groundedness 1.0, and Citation Correctness 1.0;
question 007 is partial because one required identity relationship is implied
rather than stated explicitly. All underlying units, qrels, evidence mappings,
and claim/citation judgments are reviewed, and the four component summaries
state `calibration_only=false`.

Generated component reports are write-once unless a rerun produces identical
bytes. Per-case reports include source-artifact hashes, schema/review status,
fixed retrieval scope where relevant, and available runtime/judge metadata;
summaries include annotation and judgment file hashes. The runner refuses to
mix ranked-search metrics with clause-navigation calls.

The full component annotation set now covers all 50 Release 17 development
questions in `data/QA/component_evaluation_release17.jsonl`. Every record is
reviewed and contains linked ResearchUnits, GoldResearchTasks, EvidenceUnits,
and AnswerUnits. The set contains 32 single-task and 18 decomposition labels.
The formal frozen-v3.7 Router baseline is not calibration-only and scores 0.90
Route Accuracy and 0.8927 Macro-F1, with three over-decompositions and two
under-decompositions.

Planner semantic judgments are reviewed for all 18 decomposition questions in
`data/QA/component_planner_release17_judgments.jsonl`. Manual audit tightened
seven LLM-drafted partial mappings so an omitted required subcondition cannot
receive full unit credit. The formal Planner baseline reports mean Task
Coverage 0.8241, Full Task Coverage Rate 0.6111, Task Overlap Rate 0.0, and
Out-of-Scope Task Rate 0.0. Both full reports record all inputs as reviewed and
set `calibration_only=false`.

The completed review is recorded in `docs/component_evaluation_review.md`.
The project owner explicitly delegated the manual content audit to Codex, so
every accepted record transparently identifies
`project-owner-delegated:codex-manual-audit`; this is an internal engineering
baseline rather than independent academic adjudication. The audit split 044's
acceptance and refusal units, classified non-essential 007/T2 and secondary
001 context as background, and set 017 to `single_task`. Fixed-route Planner
artifacts propagate the actual annotation status, and the historical 017
two-task artifact is excluded from the reviewed baseline.

The four evidence-path baselines are now also complete over all reviewed
GoldResearchTasks/questions. Researcher was rerun directly from 78 fixed
GoldResearchTasks; every task produced evidence and at least one successful
ranked call. Reviewed judgments are stored in
`data/QA/component_retrieval_release17_judgments.jsonl`,
`data/QA/component_researcher_release17_judgments.jsonl`,
`data/QA/component_final_evidence_release17_judgments.jsonl`, and
`data/QA/component_synthesizer_release17_judgments.jsonl`.

Formal non-calibration results are Retriever Target Recall@5/@12 0.8761 and
nDCG@5/@12 0.8738 across 78 calls; Researcher Evidence Coverage 0.9872 and
Precision 0.6178 across 78 tasks; Judge Final Evidence Coverage 0.98,
Precision 0.908, and Overlap 0.102 across 50 questions; and Synthesizer Answer
Coverage 0.975, Groundedness 0.9937, and Citation Correctness 0.9567 across 50
questions. All four reports have zero errors and complete reviewed counts.

Judge candidate pools are explicitly gold-sufficient and contain runtime
evidence plus exact targets. Synthesizer evaluation is independently isolated:
it receives fixed approved gold evidence rather than the runtime Judge's
selection. Task-bound duplicate evidence IDs remain eligible for citation;
only Judge's whole-question Final Evidence metric deduplicates identical
provenance.

The component-evaluation foundation passes the canonical repository gate:
structure and QA-data checks pass and all 204 tests pass. The formal full-set
reports for Router, Planner, Retriever, Researcher, Judge, and Synthesizer have
complete reviewed counts and were generated without proposed-input overrides.

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

The data tree has explicit responsibilities: `data/3gpp/` retains every
local historical specification version, `.local/state/` stores ignored mutable
workspace state, and `data/QA/` contains the
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
-> Researcher first-round action planning
-> trusted retrieval tool execution
-> bounded observe-and-adapt loop
-> source-verified evidence
-> sufficiency/gap analysis
-> bounded re-retrieval
-> grounded answer
```

All migrated modules import successfully. The canonical quality gate now
passes the structure contract, validates every curated question's local source
hash, R17 version, line locator, and exact excerpt, and passes all 105 focused
tests.

The first behavioral refactor is complete. Corpus discovery/chunking, immutable
retrieval contracts, BM25, semantic candidate search, reciprocal-rank fusion,
snapshot persistence, and hybrid orchestration now have separate modules.
Chroma-specific construction is isolated under `infrastructure/`. The bounded
Researcher is split into explicit models, evidence state, action policy, and
loop execution. Its submit-for-validation, round-budget, and agent-error exits are
covered by characterization tests.

The former deterministic question compiler, which extracted anchors and facets
and generated suggested queries before the Researcher loop, has been removed
from the supported path. Candidate-specification routing for retained retrieval
workflows remains isolated in `routing.py`. Execution-result,
evidence-selection, repair, and answer-summary contracts remain extracted from
the monolithic planner.

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
now exist together with Keyword, Semantic, Hybrid, HyDE, Reranker, Clause, and
bounded source-context adapters. HyDE fuses exact lexical anchors with an
untrusted hypothetical normative passage generated by the controller, while
only retrieved source chunks can become evidence. Every emitted evidence item preserves its active corpus
scope, Release, exact version, source path, document hash, heading path, and
line locator. The bounded Researcher now invokes these capabilities only through
the Tool Registry; its old direct retrieval branch has been removed. All seven
tools are verified independently from the agent loop. Clause lookup accepts an
exact clause identifier, a title, or their combined heading and normalizes the
missing whitespace found in converted 3GPP headings. A model-proposed clause is
treated only as an untrusted retrieval hypothesis; only returned source content
can enter the evidence pool. Specification scope remains limited to values
grounded in the task, dependency context, or observed evidence. HyDE retrieval
fuses BM25 over exact protocol anchors with embedding retrieval over the
hypothetical normative passage; the hypothetical text is never evidence.
Source-context expansion stays within one document and parent section while
covering the seed subsection and its immediately adjacent sibling subsections.
When observed evidence identifies an answer-bearing specification and clause,
the Researcher is instructed to use clause lookup before attempting repeated
local source-context expansion, and to keep specification scope separate from
the clause value.

The Researcher no longer prepares a retrieval plan before its bounded loop,
either through a separate model call or through deterministic anchor, facet,
and query generation. The first Researcher controller round receives
the Planner's `ResearchTask` items directly, together with the registered tool
capabilities, active corpus scope, dependency context, and budget. It generates
the initial query strings and executable tool calls itself. Every substantive
answer unit must be directly targeted by at least one first-round action. The
action list has no separate per-turn count ceiling, while trusted validation
enforces one total tool-action budget shared by every controller round. Search
and source-context expansion no longer have separate quotas, though their usage
counts remain in the execution artifact for audit. The production Researcher
also retains a separate maximum-round limit.
Keyword search is reserved for short exact protocol anchors and must not merely
repeat a hybrid lexical anchor; hybrid search covers exact anchors combined
with conceptual behavior or relationships. Distinct rules or mechanisms receive
separate calls even when they belong to one simple task. An empty specification
list searches the complete active Release corpus. A call is restricted to named
specifications only when an exact number appears in the task, validated shared
context, or trusted observed evidence, preventing guessed scope from hiding the
correct source. V1.5 follows a normative cross-reference only when the current
passage delegates answer-bearing detail to it; incidental references do not
block completion. Subsequent rounds use the same controller
contract to observe the evidence ledger and choose only the next bounded
actions. The versioned prompt text lives under
`specrag.prompts`.

A planning-only Researcher boundary now exposes that first turn before any
retrieval execution. It records the exact task/context/tool/budget payload, the
model decision and receipt, and per-action trusted validation. The
`research-plan` CLI uses reviewed Planner labels, discovers only the selected
corpus document identities, and does not build retrieval indexes or invoke a
tool. Independent dependency-ready tasks receive separate proposals; tasks
whose dependencies have not produced validated evidence are recorded as
deferred rather than planned prematurely.

The first `researcher-controller/v1.4` simple-question trial used questions
001-006, 019, 025, 035, and 044. All ten plans passed trusted validation and
contained 23 actions in total. Executing those exact frozen actions, without a
second controller round, completed every tool call and strictly matched 10 of
12 annotated evidence targets; 8 of 10 questions matched all of their targets.
Question 003 retrieved a neighboring sidelink radio-link-failure procedure
instead of the annotated connection-release rule. Question 035 retrieved the
complete same-ng-eNB resume rule but its separate freshness query drifted to
EPS key-chaining material and missed 33.501 Annex A.10. HyDE was selected for
seven questions and reached an annotated target in six; its generated passage
remains query-only. The reusable offline runner under
`scripts/evaluate_first_turn.py` scores source targets only after executing the
frozen model plan, so expected answers and target locators never enter planning
or retrieval inputs.

The same ten simple questions were then run through the complete bounded
`researcher-controller/v1.5` loop. All 12 annotated targets were covered across
all ten questions. Question 003 used evidence-grounded clause lookups and
source-context expansion to close delegated sidelink release actions; it ended
at the three-round limit after collecting the target evidence. Question 044
used `expand_source_context` to move from successful-operation seeds to the
adjacent X2AP and XnAP unsuccessful-operation subsections, then finished in its
third round. No action was rejected or unavailable, and the only restricted
specification calls were the two 38.331 clause lookups grounded directly in
question 003's observed evidence.

A fresh complete-loop run over all 34 reviewed simple questions did not confirm
that stability strongly enough to freeze the controller. It covered every
annotated target for 32 questions and 36 of 38 targets overall. No tool action
was rejected, unavailable, or returned no hits. Twenty-eight questions used one
retrieval batch; six performed later evidence-driven retrieval. Those later
calls completed missing targets for questions 013, 015, 016, 036, and 037, and
added source context for 022. Question 003 stopped after the first batch with
the correct top-level sidelink-RLF procedure but without the annotated concrete
SRB entity-release details. Question 044 retrieved the XnAP success and failure
messages but missed the annotated high-level cause, trigger-delta, response, and
OAM constraints. Both controllers nevertheless declared the evidence complete,
whereas both questions had succeeded in the earlier ten-question v1.5 run.
This exposed unstable Researcher self-assessment rather than a rejected-action
or tool-availability failure. The architecture now treats that self-assessment
only as a submission decision; formal completeness is decided downstream.

`researcher-controller/v1.6` renames the terminal Researcher action from
`finish` to `submit_for_validation`. Submission means only that the current
candidate set is ready for the external evidence stages, or that no useful
bounded retrieval action remains. The former per-passage model support assessor
and separate model conflict detector have been removed. The graph now runs
`Researcher -> source_validator -> sufficiency_judge`: the source validator
deterministically deduplicates and checks allowed paths, document hashes, line
locators, and exact excerpts; one task-level `evidence-sufficiency/v1.6` call
selects the supporting evidence IDs, decides complete/partial/insufficient/
conflict, and creates the next `ResearchGap` when needed. Its evidence contract
allows explicit factual premises across passages to establish the task
collectively at the scope requested by the question; the task objective may
clarify that scope but cannot broaden its required answer granularity. Explicit
facts across passages may collectively establish descriptions, mappings,
comparisons, and evolution. A task asking for an overall permitting,
triggering, continuation, selection, or success criterion instead requires an
explicit statement of that overall criterion; a locally successful check and
its next procedure step establish only that branch. Failure or fallback evidence
is required only when requested by the task. Only selected, source-valid
evidence can reach dependency context, synthesis, or Memory.

A controlled `evidence-sufficiency/v1.6` replay separates these two judgment
cases using frozen, source-valid evidence. Question 045 T1 without the direct
TS 38.331 overall continuation criterion remains `partial`; its gap asks for an
explicit criterion connecting context identification, retrieval, and validation
to continuation. Question 048 T2 remains `sufficient` because explicit
before/after message contents and interface rules collectively establish the
requested content evolution. These replays do not use QA answers or evidence
targets as model input.

The subsequent frozen-plan end-to-end rerun of complex question 048 completed
successfully with `researcher-controller/v1.11` and
`evidence-sufficiency/v1.6`. T1 used six calls and was validated from TS 29.274
for the PGW-to-SGW S5/S8 and SGW-to-MME S11 path. Its selected, validated
evidence was available before T2 was dispatched. T2 used five calls, including
direct lookups of TS 23.401 clause 5.4.1 and TS 29.274 clause 7.2.3 plus an
unrestricted cross-stage hybrid search, and was judged sufficient from explicit
before/after message contents and interface-specific forwarding rules. The run
finished `completed` with no gaps or errors and separate per-task call counts;
T1's use of all six calls despite an early exact path hit remains an efficiency
observation, not a correctness failure. The audit artifact is under
`artifacts/runs/complex-freeze/20261012-v111-researcher-v16-sufficiency/`.

The same frozen-plan evaluation entry point now supports resumable directory
batches. It atomically writes each question artifact and refreshes a batch
`summary.json` after every result. Re-running with `--batch --resume` skips
valid existing artifacts, retries missing, truncated, or mismatched outputs,
and therefore preserves completed-question progress across process or machine
interruptions without introducing a second evaluation implementation.

Researcher action validation also enforces the stable retrieval request length
limits before request construction. Oversized model-generated query or clause
fields are recorded as rejected actions, consume no tool-call budget, and are
returned in the next controller history for correction instead of escaping the
worker as a Pydantic validation exception.

A live v1.6 compatibility run on simple questions 003, 033, and 044 confirmed
that the configured model and trusted action validator accept the renamed
contract. Questions 003 and 033 each used two first-round searches and then
emitted `submit_for_validation` in round two, with no rejected actions or
errors. Question 044 emitted only valid retrieval actions, but used all six
actions across three rounds and stopped at the action budget. Its first round
filled the 12-item evidence ledger with negotiation, successful-operation, and
high-level parameter evidence; later refusal-focused searches could not add a
new passage. This is retained as an evidence-cap/budget observation rather than
an action-contract incompatibility. The compatibility artifacts contain no QA
answer or evidence-target fields and live under
`artifacts/runs/researcher-loop/20261012-v16-compat-003-033-044/`.

`researcher-controller/v1.7` replaces the first-arrival evidence cap with a
two-level ledger. Every unique tool hit enters a bounded-by-execution candidate
pool with stable evidence IDs and full provenance, while the controller
explicitly maintains at most 12 active items per task. New hits are exposed as
pending candidates; previously reviewed but unretained hits remain available
as abbreviated inactive candidates. Trusted code validates every retained ID,
task ownership, uniqueness, and active-set limit before replacing the active
set. Only active evidence is submitted downstream, while the complete
candidate execution remains in the Researcher artifact. If the last retrieval
batch consumes the round or action budget, one finalization-only controller
turn may review it and submit a final selection but cannot call another tool.
This removes the failure mode observed on question 044, where early broad hits
filled the ledger before later gap-directed evidence could be considered.

`researcher-controller/v1.12` makes the evidence-review order explicit. Later
rounds now review complete candidate excerpts, map evidence to substantive
answer units, replace the bounded active set while preserving direct support,
identify only the remaining evidence gaps, and then choose the next retrieval
actions. Active, pending, and inactive review views no longer truncate source
excerpts before model review. The active-evidence limit still applies
independently to each task, and unretained candidates remain auditable and may
be selected again later.

A live v1.7 rerun of question 044 confirmed the new behavior. Three first-round
searches produced 12 candidates; the controller retained nine and discarded
two redundant message tables plus one unrelated EPS handover passage. A
source-context expansion then added three candidates, including the XnAP
unsuccessful-operation passage as candidate 15. The third controller turn
selected that late candidate into the 12-item active set and submitted without
exhausting the six-action budget. The final evidence directly covers both
annotated sources: the 38.300 negotiation/delta/cause/response/OAM rule and the
38.423 request, acknowledge, failure, cause, and optional-range branches. The
run used three searches and one expansion, had no errors, and is stored under
`artifacts/runs/researcher-loop/20261012-v17-044-rerun/`.

The first complex-question v1.7 gate used frozen Planner output for question
007, whose two ResearchTasks are independent and target TS 23.501 and TS
38.331 respectively. T1 submitted in two rounds after selecting one direct
passage from seven candidates; T2 submitted in three rounds after selecting
four passages from 13 candidates. Both workers stopped normally without errors
or exhausted budgets. Post-run diagnostics confirmed complete coverage of all
four annotated facts and both annotated source locations: CAG-ID uniqueness,
the assumed twelve-ID capacity, the optional display name, and the RRC
PLMN/CAG-ID grouping and PNI-NPN identification rules. The run is stored under
`artifacts/runs/researcher-loop/20261012-v17-complex-007/`.

The first dependency-bearing run used frozen Planner output for question 045
with the existing global six-tool budget. T1 used six accepted calls across
three controller rounds, then its finalization turn selected eight active
passages. All eight passed deterministic source validation, and the Sufficiency
Judge selected six and marked T1 sufficient with no gap. Those passages include
both annotated T1 sources and establish the valid-context retrieval and
verification condition. One attempted scope to TS 38.473 was correctly rejected
because that specification had not yet been grounded; a later unrestricted
search found the relevant TS 38.473 passage. The graph made T2 dependency-ready,
but the global budget guard stopped at `tool_budget` before T2 could run because
T1 had consumed all six calls. This exposes a complex-workflow budget-allocation
issue rather than a source-validation, sufficiency, or dependency-evidence
failure. The diagnostic artifact is stored under
`artifacts/runs/dependency-loop/20261012-v17-complex-045/`.

The LangGraph runtime lives under `specrag/orchestration/`.
It defines one unified `ResearchState`, injected handlers for Router, Planner,
Researcher, deterministic Source Validator, Sufficiency Judge, gap recovery,
and Synthesizer, plus an internal budget guard, an in-memory checkpointer, and
thread resume helpers. Both simple and complex Router outcomes enter Planner;
there is no direct Router-to-END runtime edge. The supported business-flow
diagram folds the budget guard and gap-task reactivation into labeled Judge
branches because neither is a separate reasoning or model stage.
Framework-independent research task, evidence, gap, tool-call,
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
- Source validation, task-level evidence selection/sufficiency/conflict
  handling, and grounded synthesis are now clean product modules. Long-term
  Memory has a validated-only
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
set will be collected after the complete workflow is optimized. The historical
deferral note is superseded by the answer-evaluation milestone described above.

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
the first Researcher controller round, which generates both the queries and
their executable calls against the registered tools. The simple branch is
therefore accepted without an additional task-decomposition or pre-loop
retrieval-planning step.

Dependency scheduling now distinguishes normal plan execution from corrective
re-planning. Pending tasks are not assessed before their dependencies complete,
so they do not create synthetic evidence gaps. When a predecessor becomes
sufficient, its newly ready dependent task proceeds directly to the next
Researcher wave without consuming `replans_used`. Gap Re-planner is reserved for
tasks that have actually run and remain incomplete or blocked, and it reopens
only tasks named by unresolved gaps.

The Researcher now uses Responses API native function calling instead of the
former custom `AgentRetrievalDecision.actions` wire protocol. The seven
retrieval tools expose separate strict argument schemas through the Tool
Registry; provider-specific conversion remains in `infrastructure`, while the
agent loop receives provider-neutral function calls. A normal controller round
returns exactly one `update_research_state` control call plus one or more
retrieval calls. A submission round returns only `submit_for_validation`.
Only executed retrieval calls consume the per-task budget; state updates,
submission, and finalization do not.

Tool results are returned as matched `function_call_output` items, and the next
round also receives the complete ledger-backed Researcher state. Since the
configured stateless endpoint intermittently rejected replayed provider
reasoning with `invalid_encrypted_content`, encrypted reasoning is neither
requested nor retained. The adapter replays only the minimal function-call
items required by the protocol. Call IDs now come from the provider and remain
attached to validation, execution, receipts, and artifacts. The superseded
custom action models and shared empty-field request convention were deleted.

The staged live acceptance passed all ten selected simple questions after that
transport correction: every question completed with sufficient evidence, zero
replans, zero errors, and two or three retrieval calls. Frozen-plan question
039 retained sufficient T1/T2 results and the expected TS 23.501 and TS 23.502
evidence for T3; its known downstream Judge scope issue remained partial.
Question 050 retained the correct direct T1 evidence and sufficient T2/T3
evidence, including the previously fragile T2 passage. Its downstream Judge
nevertheless interpreted the normative phrase "at least Kc and CKSN" as a
requirement to prove no other constituents exist, contrary to the frozen
Judge instruction not to demand exhaustive details. This is recorded as a
Judge boundary, not a retrieval/tool-calling regression, and the migration did
not alter the Judge.

The live eight-stage progress checklist is maintained in
`docs/plans/003-optimization-roadmap.md`.

The Sufficiency Judge scope boundary is now isolated from Planner metadata.
`evidence-sufficiency/v1.9` evaluates only the `ResearchTask.question` and
source-valid evidence; `objective`, dependency metadata, search rationale, and
task status remain in workflow state for Planner/Researcher behavior and audit
but are not sent to the Judge. This prevents a Planner objective from adding
requirements such as exhaustive-set or modification/storage proofs that the
assigned question did not ask for. Source validation, evidence selection,
missing-relation generation, and downstream synthesis contracts are unchanged.

The v1.8 Judge contract made its decision procedure explicit without adding
answer generation: identify the answer units requested by the task question,
map complete candidate passages to those units, select the smallest
non-redundant direct-evidence set, and then assign sufficient, partial,
insufficient, or conflict. Partial assessments retain only evidence that
directly supports established units and describe only the remaining unsupported
unit in `missing_relation`. The earlier long special-case wording for overall
success and continuation criteria is replaced by one general rule: a local step
or successful instance establishes only that part, while an overall condition
or relationship requires evidence for the complete requested conclusion.

The v1.9 contract adds three general boundaries without changing the output
schema or workflow: minimality cannot remove evidence needed for complete task
coverage; a passage that only delegates an answer-bearing fact to another
clause does not directly establish that fact; and multiple passages may be
combined only when their explicit facts directly form the requested conclusion,
not merely because they concern the same procedure.

The v1.9 live check covered simple questions 003 and 044 and frozen-plan complex
question 048. Question 003 remained sufficient from the passage that explicitly
states the task-level sidelink-RLF actions; the presence of delegated subclauses
did not trigger indiscriminate reference expansion. Question 044 selected the
two direct successful/unsuccessful TS 36.423 passages from six X2AP candidates,
showing that minimality removed redundant general, message-table, and range
definition passages. Its candidate pool did not contain the annotated XnAP
variant, which remains a retrieval-scope observation rather than a Judge
selection error. Question 048 completed both tasks: one TS 29.274 passage
established the two-hop interface path, while seven non-redundant table chunks
collectively established the requested before/after message-content evolution.
The cross-passage rule therefore did not regress an explicit mapping case.

A live v1.8 compatibility check completed simple questions 033 and 044. The
single-clause NAS-security question retained one direct TS 23.401 passage and a
complete answer. For the mobility-setting question, the Judge reduced the
previously observed ten-passage X2AP/XnAP mixture to the two direct TS 38.423
successful- and unsuccessful-operation passages. The resulting synthesis used
only NG-RAN/XnAP terminology and covered request, acceptance, refusal, cause,
and optional permitted-range feedback. Both runs completed without a gap or
re-plan.

Judge-reported gaps now receive a separate bounded corrective retrieval phase.
`ResearchBudget.max_tool_calls` is the per-task initial allowance and
`max_gap_tool_calls` is an independent six-call per-task allowance used only
while an unresolved Sufficiency Judge gap is present. Initial and corrective usage are
persisted separately while the aggregate usage remains auditable. Single-task
dispatch and parallel fan-out/fan-in select and charge the same phase-specific
allowance. The Researcher controller contract is `researcher-controller/v2.1`:
on corrective re-entry it reviews complete candidates, maps evidence to answer
units, updates active evidence, states the precise remaining gap, then rewrites
that gap into new retrieval calls in the same model turn. No second query-
rewrite node or model call was added; Planner, retrieval algorithms, Source
Validator, and Sufficiency Judge responsibilities are unchanged. Runtime state
was `research-state/v1.5` for the expanded budget shape; the active multi-user
runtime contract is `research-state/v1.6`.

Final synthesis now has the same explicit scope boundary. The
`evidence-synthesis/v1.2` prompt lives under `specrag.prompts`; the model receives
the original question plus each task ID, task question, and only the
source-valid evidence selected as supported by the Sufficiency Judge. Planner
objectives, dependencies, task status, gaps, search history, rejected evidence,
and other runtime metadata are not synthesis inputs. The original question is
authoritative and task questions are coverage checkpoints.

V1.2 also accepts the Run's resolved answer language and detail level as
presentation controls. They can change wording only; the original question,
eligible evidence, and claim coverage remain authoritative.

The Synthesizer now produces the complete user-facing answer directly. Its
runtime output is `answer + evidence_ids`, with exact `[evidence_id]` markers
placed inline after supported statements. Code requires the declared IDs to
match the inline citations, rejects unknown or ineligible evidence IDs, and
requires cited evidence owned by every sufficient ResearchTask. There is no
parallel synthesis-claim list and no claim-concatenation step. The prompt still
requires minimum sufficient scope and preservation of specification-specific
actors, interfaces, messages, states, and terminology instead of merging
analogous procedures across systems. Historical claim-based JSON artifacts
remain readable by the offline compatibility path.

Live compatibility and quality checks completed for simple questions 003, 033,
and 044 and frozen-plan complex questions 008, 048, and 050. All six completed
the workflow with sufficient evidence and passed the synthesis task-coverage
contract. Question 033 and complex question 050 closely match their reviewed
answers; questions 008 and 048 retain complete main-task coverage with concise,
claim-identical answers. Questions 003 and 044 remain useful upstream boundary
cases: 003 is still declared sufficient from a broader sidelink-RLF action
passage without the annotated delegated SRB-entity release detail, while 044's
Judge-supported set contains both X2AP and XnAP variants. The Synthesizer now
keeps those systems separate, but correctly does not replace the Judge by
discarding otherwise supported input evidence.

An offline semantic evaluation was then run over all 48 complete final-answer
artifacts using `synthesis-quality-evaluator/v1.0`; QA annotations were supplied
only to this evaluator and never to the runtime Synthesizer. The four scored
dimensions are factual completeness, evidence faithfulness, scope precision,
and clarity, each on a 0--4 scale. Simple answers averaged 3.92/4.00 (30 pass,
4 review); complex answers averaged 3.79/4.00 (8 pass, 5 review, 1 fail). The
only fail was complex question 041, whose Judge-supported evidence omitted the
distinct AMF re-allocation registration step. The main review findings were
minor omitted conditions or scope broadening; no answer had an invalid evidence
reference. The two Researcher-insufficient cases, 039 and 047, were excluded
from synthesis quality scoring and remain explicit incomplete runs.

The task-level evidence contract is now `evidence-sufficiency/v1.11`. The
original user question is passed to the Judge as a coverage guard, while the
ResearchTask question remains the authoritative local scope. Exact requested
actors, actions, messages, directions, and return paths must be directly
supported; for requested negotiation procedures the Judge separately checks
the proposal or decision basis, peer response, and initiator-side response
handling or applicable limits. This prevents a broad procedural passage or
success/failure message pair from being treated as proof of a more specific
relation.

Final synthesis uses `evidence-synthesis/v1.6`. It preserves nested branch
antecedents, exceptions, implication direction, and normative limits that
qualify the requested conclusion, while omitting independent evidence-backed
facts that are unnecessary to answer the original question. Focused live
regressions scored 003 at 3.75, 020 at 3.50, and 024 and 044 at 4.00 on the
offline four-dimension evaluator. Questions 041 and 049 remained partial at
the Judge because their evidence did not directly establish the requested
Registration Request/load-rebalancing link and selected PDU Session Type
return/handling link; synthesis correctly did not run for either case.

Judge-gap corrective retrieval is now being migrated from mandatory query
rewriting to evidence-gap-directed acquisition. The runtime contract is
`research-state/v1.7` and the Researcher controller contract is
`researcher-controller/v2.2`. Non-sufficient Judge assessments now carry a
small required proposition, missing semantic dimensions, evidence-need type,
trusted source-grounded anchors, and already-supported evidence IDs. Before a
corrective Researcher run, orchestration restores all source-valid candidates,
the Judge-supported active evidence, and prior tool-call outcomes. The
Researcher can therefore reselect existing evidence, navigate a grounded
clause or local context, rerank candidates, or perform a new search; query
rewriting is only one possible search strategy and no extra model node was
added.

The first live validation of this design completed complex question 049:
T2 recovered the direct V-SMF-to-H-SMF request and H-SMF-to-V-SMF response
evidence and ended sufficient. Question 041 consumed its corrective allowance
but remained incomplete because the retrieved passages still did not directly
connect the AMF load-rebalancing initial NAS message to a Registration Request.
This remaining issue is retrieval-navigation quality rather than loss of
candidate state or incorrect corrective budget accounting.

The corrective Researcher pass is now separated from the initial pass at the
prompt-contract level. Initial retrieval uses `researcher-controller/v2.3`,
while Judge-gap retrieval uses `researcher-controller-gap/v1.0`. They share the
same Researcher service, native tools, evidence ledger, validation policy, and
independent budgets; the runtime selects the prompt and receipt version from
`gap_retrieval_mode`. The dedicated corrective prompt requires review of the
restored evidence first, then grounded clause/context navigation, and only then
a new search when necessary. It also stops repeated searches that produce no
 new candidates.

The follow-up live run of 041 used `researcher-controller-gap/v1.0` for all
corrective turns. It completed with 12 retrieval actions, including targeted
Reroute NAS Request searches and grounded clause/context navigation, but T2
remained partial because the corpus still did not provide a direct statement
identifying the AMF load-rebalancing initial NAS message as a Registration
Request. This confirms the prompt split is active; it does not by itself solve
the underlying cross-clause evidence-closure limitation.

The Sufficiency Judge now uses `evidence-sufficiency/v1.13`. Its evidence
selection rules preserve strict source grounding while allowing a small set of
complementary passages to close one relationship when each link is explicit
and the passages share an aligned actor, message/object identity, stage,
direction, and condition. A source-backed identity mapping is mandatory;
topically related or analogous passages still remain insufficient.

The 041 rerun with `evidence-sufficiency/v1.13` completed successfully. The
Judge selected the TS 23.501 §5.19.4 load-rebalancing passage together with
the TS 23.502 §4.2.2.2.2 Registration Request/Initial NAS identity mapping,
and marked both T1 and T2 sufficient. This confirms that the earlier 041
failure was caused by the former single-passage closure rule, not by missing
retrieval evidence.

The end-to-end and efficiency evaluation milestone is complete. Binary Task
Success combines deterministic
completion/citation/scope gates with a versioned semantic proposal and a
mandatory manual decision over the four semantic gates. Reviewed-only
aggregation reports Task Success Rate plus latency, total Token usage,
per-LLM-call Token records and counts, and ordered tool-call details. Full
workflow receipt capture includes Router, Planner, Researcher, Sufficiency
Judge, and Synthesizer calls as well as failed logical calls; missing provider
usage remains explicit instead of being counted as zero.

Baseline Runs retain a reproducibility manifest containing the question-set
hash, selected IDs, Git revision/dirty state, budget, exact Release 17 corpus
documents and document hashes, retrieval index fingerprint, warm-cache
condition, and sequential execution policy. After the original provider probes
were blocked, a replacement key restored the frozen `gpt-5.6-sol` channel. The
ten-question pilot completed at 7/10 (70.0%) Task Success with complete Token
telemetry. The subsequent fresh 50-question formal run completed with 49 normal
answers and one iteration-budget failure, and all 50 decisions were manually
reviewed. The final baseline is 38/50 (76.0%) Task Success: 29/34 (85.3%) on
simple questions and 9/16 (56.2%) on complex questions. Exact corpus compliance
is checked per question: every cited
evidence path, exact version, Release, and document hash must belong to the
Run's frozen document snapshot. Evaluation batches freeze the expected
question IDs and input hashes; aggregation refuses missing or extra cases, and
resume refuses judgments whose source Run hash changed.

Formal efficiency is 79.183 seconds mean and 191.969 seconds P95 latency,
450 LLM calls, and 289 tool-call attempts. Token telemetry is exact for 49/50
Runs and totals 2,652,796 Tokens for those Runs. A five-attempt Sufficiency
Judge call in `r17-qa-046` lacks provider usage for four failed attempts, so the
exact whole-batch Token total remains unavailable; the observed 2,761,525-Token
lower bound is not misreported as an exact total. Manual audit corrected five
over-permissive Judge proposals that omitted reviewed Answer Units. Full
results, failed cases, and reproduction metadata are documented in
`docs/end_to_end_evaluation.md`, with the metric contract and execution history
in `docs/plans/032-end-to-end-efficiency-evaluation.md`. The canonical quality
gate passes: structure check, 50-question QA dataset check, and all 219 tests.

Workflow visualization is now reproducible from the active LangGraph
implementation. `docs/architecture/single_research_workflow.mmd` holds the
concise Single Researcher business view, while
`docs/notebooks/langgraph_workflows.ipynb` compiles both Single and Parallel
graphs with non-executable test handlers, asserts their exact static topology,
and exports generated files only to ignored `artifacts/diagrams/`. Explicit
conditional destinations make the compiled Parallel graph unambiguous, and
automated topology tests detect drift. JupyterLab remains a development-only
dependency; notebook checkpoints are ignored. The implementation record is
`docs/plans/033-workflow-visualization.md`.

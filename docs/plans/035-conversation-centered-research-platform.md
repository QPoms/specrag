# Conversation-centered research platform

Status: implemented; all 8 execution stages completed on 2026-10-23.

Implementation contracts, transaction boundaries, state models, API/MCP
schemas, persistence layout, and recovery behavior are defined in
`docs/architecture/conversation_research_platform.md`.

## Execution progress

The persistent implementation goal is split into eight reviewable stages. A
stage is complete only after its focused tests and the repository quality gate
pass.

- [x] 1. Domain contracts and access policy: Conversation, Turn, tool call,
  scope revision, Conversation Event, research link, Research Outcome/Result,
  pure lifecycle transitions, and owner-private content permissions.
- [x] 2. Conversation persistence, Context Harness integration, authorized
  Conversation service, and HTTP APIs.
- [x] 3. Frozen Run input snapshots, attempt-aware outcomes, independently
  persisted Research Results, and normal-completion semantics.
- [x] 4. Completion outbox, lifecycle worker, Conversation/Run links, ordered
  events, serialized Turn jobs, and idempotent conditional auto-presentation.
- [x] 5. Internal Streamable HTTP MCP research facade and trusted service
  delegation.
- [x] 6. Bounded Conversation Agent, context assembly, tool loop, and source
  attribution enforcement.
- [x] 7. Conversation-centered frontend and reconnect/event recovery.
- [x] 8. End-to-end acceptance, regression/evaluation hardening, operational
  limits, and final documentation.

Stage 1 introduced the active `conversation/v1`, `conversation-turn/v1`,
`conversation-event/v1`, `research-outcome/v1`, and `research-result/v1`
contracts. It also split private content reads from administrator lifecycle
control. Stage 2 added migration 7, the owner-private Conversation store and
service, atomic user-message/Turn/job/event commits, the serialized Turn lease
queue, canonical Context Harness history, CRUD APIs, and cursor-based SSE
recovery. Restart, migration, concurrent ordering, idempotency, compaction, and
administrator-isolation coverage pass. Stage 3 added migration 8, immutable
semantic Run input snapshots, exact version binding, attempt-aware terminal
Outcomes, an atomic completion outbox, deterministic Research Result
materialization, owner-private Result APIs, and Result IDs in Run inspection.
Normal budget and evidence-insufficiency stops now remain completed research
with an explicit evidence status rather than becoming execution failures.
Restart, replay, migration, revision-chain, provenance-copy, and administrator
isolation coverage pass. The quality gate passed with 266 tests on 2026-10-22.
Stage 4 added migration 9, persistent completion receipts, attempt-serialized
completion leases, Conversation research-link projections, and the backend
Lifecycle Worker. Reconciliation atomically inserts Results, updates links,
emits ordered Conversation Events, and applies the submitted-user watermark,
membership, active-Turn, and one-presentation-at-a-time checks before creating
an idempotent automatic presentation Turn. Late tool-call links are retried;
new user messages atomically skip pending/queued presentations and cancel their
queued Turns. Embedded and standalone Worker modes now process both Run and
completion jobs. Race, duplicate, multi-Run, abnormal completion, and reconnect
projection coverage pass. Stage 5 added the official MCP Python SDK and the
loopback-only `specrag-mcp` Streamable HTTP service with `start_research`,
`get_research_status`, and `get_research_result`. Tool schemas exclude identity
and permission fields. A short-lived signed service delegation carries origin
metadata outside model-controlled arguments; the server verifies it and
re-checks the current user and Workspace membership. `start_research` creates
one idempotent Run, attaches its research link, and waits at most ten seconds
for that same Run without changing its budget. Both synchronous Result and
accepted asynchronous projections use the persisted Run/Result services.
Transport, tamper, revocation, permission, wait-window, and replay coverage
pass. The quality gate passed with 275 tests on 2026-10-22. Stage 6 added the
bounded native-tool Conversation Agent, Context Harness/Memory/Result assembly,
durable tool-call audit and full-result storage, migration 10, service-delegated
MCP execution, and a leased Turn Worker. Final assistant publication checks
owner-private Result bindings and citation membership in the same transaction
that releases the Conversation slot; ordinary direct replies allocate no Run,
and deterministic guards reject direct normative claims or invented evidence
references. Focused routing, budget, persistence, replay, and Worker tests pass.
The quality gate passed with 281 tests on 2026-10-22. Stage 7 changed the
browser's primary research surface to a persistent Conversation list and
thread. It supports create/select/archive/restore/continue, source-attributed
assistant messages, inline Turn/tool/Run/Result state, and historical Run
opening while new messages continue. Authenticated Conversation SSE resumes
from the last cursor, ignores duplicate delivery, refreshes the authoritative
projection after changes, and repairs sequence gaps after disconnect. The Vite
production build and 281-test backend gate pass on 2026-10-23. Stage 8 adds
synchronous and asynchronous whole-platform acceptance, persistent confirmed
scope inheritance, historical Result dependency snapshots, direct-versus-
research routing metrics, per-user Turn concurrency, bounded MCP payloads and
timeouts, pre-audit argument validation, and an administrator-safe Run metadata
projection. Workspace administrators can cancel member Runs without receiving
questions, checkpoints, events, Results, or evidence. Restart, condition-change,
duplicate-completion, automatic presentation, compaction-reference, privacy,
and unsupported-normative regression coverage pass. The canonical backend gate
passes with 288 tests; the final frontend production build also passes.

## Goal

Evolve SpecRAG from a Run-first research application into a persistent,
conversation-centered 3GPP research platform. A user works in one durable
Conversation, may ask ordinary conversational questions, and may start one or
more evidence-backed Research Runs without having to restate the established
research context. Every normally completed Run retains an independently
readable Research Result with exact evidence, applicability constraints, and
unresolved questions.

This is an extension of the current implementation. The existing retrieval,
Researcher, evidence, LangGraph workflow, Run Harness, Worker, Context Harness,
Memory, SpecFact, corpus-version, access, API, and frontend paths remain the
only supported implementations of those capabilities.

## Product invariants

- A `Conversation` is the durable user interaction boundary. It owns ordered
  messages and tool-call history, but does not own research execution state.
- A `ResearchRun` is one independently executable, resumable research task.
  It may belong to a Conversation; existing CLI-created Runs remain readable.
- A `ResearchResult` is the first-phase immutable result snapshot derived from
  a Run. It records the answer, evidence state, conditions, unresolved
  questions, exact versions, and complete cited evidence. It is not a new
  knowledge subsystem and is not an automatically approved `SpecFact`.
- `Conversation History` is the canonical interaction log. Compaction never
  deletes or rewrites it; a Context Harness checkpoint only changes the active
  model projection.
- `Memory` contains reviewed or explicit long-lived user/project context. A
  conversational statement is not Memory until the existing proposal/review
  policy accepts it.
- `SpecFact` remains a reviewed workspace-scoped source-finding hint. Neither a
  Research Result nor a SpecFact may replace fresh source validation when a new
  answer changes scope, conditions, or makes a new normative claim.
- LangGraph remains confined to `orchestration` and executes Research Runs. The
  Conversation Agent uses a bounded ordinary tool-calling loop and does not
  duplicate research planning, retrieval, evidence validation, or synthesis.
- The first MCP server is an internal Streamable HTTP transport adapter over
  the existing research application services. The Conversation Agent uses an
  MCP Client; the server does not duplicate Run or research behavior.
- Backend application services, not either Agent or the model, own cross-cutting
  lifecycle consistency: Result persistence, authorization, Conversation/Run
  association, ordered events, idempotency, and conditional Agent triggering.
- Every read and write is authorized against the current actor and workspace;
  opaque Conversation, Run, Result, Context, message, and tool-result IDs are
  not capabilities.

## Domain boundaries and lifecycle

| Object | Authoritative owner | Lifecycle | Trust and reuse |
| --- | --- | --- | --- |
| `Conversation` | initiating user in one Workspace; future sharing requires an explicit grant model | active, archived | Organizes interaction only; never evidence |
| `ConversationMessage` | Conversation History | append-only ordered user/assistant/tool records | Reconstructed into model context through Context Harness |
| `ConversationToolCall` | Conversation Agent audit log | proposed, running, completed, failed, cancelled | Stores tool name, validated arguments, result reference, related Run, and failure projection |
| `ResearchRun` | existing Run Harness and Worker | queued, running, completed, failed, cancelled, timed out | Frozen corpus, Memory, and SpecFact snapshots; checkpoint state is run-local |
| `ResearchResult` | validated result contract plus infrastructure persistence, coordinated by the backend lifecycle service | sufficient, partial, insufficient, conflict; immutable after publication | Reusable source-backed output; applicability, exact versions, and unresolved gaps are explicit |
| `ContextRecord` and compaction checkpoint | existing `context` package | append-only messages plus one active checkpoint pointer | Model-input projection only; not a product aggregate or knowledge store |
| `Memory` | existing `memory` package | current reviewed lifecycle | Long-lived context, never normative evidence |
| `SpecFact` | existing `specfacts` package | pending review through approved/superseded/invalidated | Reviewed retrieval hint; fresh source lookup remains mandatory |

### Conversation state

The first supported lifecycle is intentionally small:

```text
active -> archived
archived -> active
```

Messages are never edited in place. User correction is a new message. Assistant
generation failure records the failed turn/tool call and allows a later retry;
it does not remove prior records.

Conversation content is private to its creator in the first release. Workspace
administration alone does not grant access to message bodies or private Results.
If administrators retain operational Run controls, policy and API projections
must separate lifecycle metadata/control from question, answer, evidence, and
Conversation content. Future Workspace Result publication is an explicit user
action and does not approve a SpecFact.

### Research relationship model

Each conversation-started Run records `conversation_id` and
`origin_message_id`. A follow-up Run may additionally record
`continuation_of_run_id` and one relationship kind:

- `follow_up`: asks a new question using earlier research as context.
- `refines`: narrows or adds conditions without invalidating the earlier result.
- `rechecks`: validates an earlier conclusion against new evidence or versions.
- `supersedes`: deliberately replaces the earlier result for a stated scope.

The old Run and Result remain immutable and queryable. A new result never
silently rewrites an earlier conclusion.

### Run and Result state separation

`ResearchRun.status` describes execution only: queued, running, completed,
failed, cancelled, or timed out. `ResearchResult.evidence_status` independently
describes the research conclusion: sufficient, partial, insufficient, or
conflict.

A normally completed research workflow always publishes one idempotent Result,
including when evidence is partial, insufficient, or conflicting. Failed,
cancelled, and timed-out Runs retain lifecycle diagnostics and publish no Result
unless an explicit validation step has produced a safe, clearly incomplete
stage result. No Result depends on a live LangGraph checkpoint for readback.

## Target execution flow

```text
authenticated user message
  -> authorize Conversation and append user message
  -> reconstruct bounded context from canonical history + active summary
  -> bounded Conversation Agent tool-calling loop
       -> direct response for non-specification conversation
       -> MCP Client for any specification-backed judgment
            -> create/queue existing ResearchRun
            -> existing Worker + RunHarness + Research Workflow
            -> backend lifecycle service persists ResearchResult
       -> result/read-status tools for later turns
  -> append assistant/tool records
  -> return turn state; continue in the same Conversation
```

Every research request creates exactly one Run. The MCP submission waits up to
10 seconds for that same Run to finish. Completion inside the window returns
the Result directly; otherwise it returns the Run ID and current status while
the existing Worker continues execution. The wait neither changes the Run
budget nor submits or executes a duplicate Run.

Long research completes asynchronously. The backend first persists and links
the Result, emits durable Conversation state, and evaluates the automatic
presentation preconditions. Only then may it enqueue a new internally triggered
Conversation Agent Turn. That Turn is never represented as a delayed result of
the already-ended tool call.

### Asynchronous completion and automatic presentation

Run submission freezes `conversation_id`, `origin_message_id`, `origin_turn_id`,
the current user-message sequence, effective constraints, exact version scope,
and referenced historical Result snapshots.

On completion, a conversation-level serialized lifecycle service performs:

1. idempotent Result materialization and persistence;
2. current authorization and association checks;
3. tool-call and Run-link state updates;
4. ordered `result_available` and task-state Conversation Events;
5. a compare-and-set check that no newer user message exists and the originating
   state remains valid;
6. when no Agent Turn is active, durable enqueue of one new presentation Turn
   using an operation key derived from the Conversation and Result IDs; or
7. when a Turn is active but no user-message conflict exists, leave the Result
   presentation pending and retry scheduling when that Turn finishes.

If a newer user message invalidates the watermark, the Result remains attached
to its originating message and is available for readback, but its presentation
is marked skipped and no response is inserted into the newer topic. Pending
Results are scheduled one at a time per Conversation. Multiple completion
events and process recovery must converge on one Result revision, one logical
status transition, and at most one presentation Turn per Result.

## Tool and MCP contract

Define one provider-neutral research application service and expose it through
the internal Streamable HTTP MCP server. The Conversation Agent calls it with
an MCP Client. Keep transport schemas separate from internal LangGraph state.

Initial tools:

| Tool | Purpose | Required behavior |
| --- | --- | --- |
| `start_research` | Start an evidence-backed task | Accept question, explicit conditions, version scope, prior Result references, language/detail, and a server-selected bounded execution profile; create one Run, wait at most 10 seconds, then return either its Result or Run ID/status |
| `get_research_status` | Poll one authorized Run | Return stable execution lifecycle, progress summary, safe stop/failure projection, and Result ID when available |
| `get_research_result` | Read one published result | Return answer, evidence status, applicability, unresolved questions, exact versions, and complete evidence records |

Starting follow-up research uses `start_research` with explicit prior Result
references and changed conditions; it creates a new Run and preserves the old
input and Result. Listing or selecting historical Results is an authorized
Conversation application concern and need not expand the first MCP tool set.

The first MCP release uses stateless tool requests backed by durable IDs. It
must not expose checkpoint payloads, arbitrary workspace IDs, caller-supplied
identity/roles, raw provider errors, or unrestricted source paths. A trusted
authenticated server context or authorized service-to-service delegation
constructs `ActorContext` outside model-controlled tool arguments. The server
is internal only; public MCP access and MCP resources/prompts are not required.

## Adaptive research depth

Reuse `ResearchRouteDecision` and the current single-task/decomposed planning
path. Do not add a second “simple RAG” implementation.

- A coherent factual question uses the current `single_task` route with smaller
  default iteration/tool budgets, while retaining source validation and
  sufficiency checking.
- A multi-part, conditional, cross-specification, or cross-version question
  uses the current decomposition, dependency, corrective-retrieval, and
  synthesis path.
- The Conversation Agent can state constraints and requested version scope but
  cannot choose evidence, mark sufficiency, or promote unsupported content.
- A direct conversational response is allowed only when no 3GPP normative or
  source-dependent judgment is asserted. When uncertain, call research.
- Assistant messages record `direct_conversation`, `research_result`, or
  `historical_result_readback`, plus the applicable Result IDs. Calling a tool
  alone never proves that an answer is sufficiently supported.

### Scope resolution

Resolve the effective research scope in this order:

1. an explicit scope in the current user message;
2. a confirmed and still-valid scope in the current Conversation;
3. the scope of an explicitly referenced historical Result; and
4. Workspace or system defaults.

Memory is advisory context only and cannot silently select a specification
version or override current user conditions. Ambiguous material conditions
require clarification. Every Run persists the final question, conditions,
scope, and dependent Result snapshots, then binds the requested scope to exact
corpus versions before execution.

## Persistence design

Extend the single SQLite schema with monotonic migrations. Do not introduce a
second database or a parallel message store.

### Conversation and history

- Add `conversations` with indexed workspace, owner, status, title, context ID,
  timestamps, and schema-versioned payload.
- Continue using `context_messages` as the canonical ordered message body. A
  Conversation references exactly one Context; Conversation services perform
  authorization before accessing the Context Harness.
- Add `conversation_tool_calls` for lifecycle/audit fields not represented by
  `context_tool_results`: call ID, Conversation, turn/message, tool name,
  arguments digest/payload, state, result reference, related Run ID, safe error,
  and timestamps.
- Add `conversation_id`, `origin_message_id`, `origin_turn_id`, and submitted
  user-message sequence to new Run payloads and indexed query columns. They are
  optional when reading older CLI/API Runs.
- Make message append and tool-call state transitions idempotent with client or
  agent operation keys. Enforce monotonic message sequence transactionally.
- Add a durable, leased Conversation Agent Turn queue or equivalent serialized
  mechanism. Only it may advance one Conversation's message head.
- Add a small Conversation research-link projection that binds the origin
  message/Turn, Run, latest Result, submitted user sequence, task state, and
  presentation state (`pending`, `queued`, `presented`, or `skipped`).

### Research Results

- Add `research_results` keyed by opaque Result ID, with Run, attempt,
  Conversation, owner, and workspace indexes. A normally completed Run attempt
  produces one active canonical Result; an explicitly published abnormal-stage
  partial remains as an immutable prior revision if that Run later resumes.
- Persist the final answer/conclusions, cited `CandidateEvidence` records,
  evidence status, applicability conditions, exact corpus versions, unresolved
  questions/gaps, creation time, source Run ID, and referenced prior Result
  snapshots. Copy the immutable evidence payload required for independent
  readback; do not rely only on a live LangGraph checkpoint.
- Keep first-phase lineage small: a Run input records referenced prior Results
  and the relationship intent. Do not add a general Result relationship graph.
- Materialization is idempotent for sufficient, partial, insufficient, and
  conflict outcomes. Abnormal Run termination follows the validated-stage-result
  rule above and never promotes unvalidated intermediate model text.

### Conversation Events

- Add an append-only `conversation_events` sequence for messages, tool-call
  state, Run association/status, Result availability, auto-presentation queued
  deferred/queued/presented/skipped, and Agent Turn state.
- Keep detailed retrieval and graph execution in existing Run Events; reference
  Run IDs rather than copying those details into Conversation Events.
- Support ordered reads from a sequence cursor, disconnection catch-up, and
  idempotent duplicate consumption. Database state remains authoritative.
- A frontend reconnect loads the current Conversation projection and then
  continues from its last consumed Conversation Event sequence.

### Retention and deletion

The first slice implements archive/unarchive, not hard deletion. A future
retention policy must treat Conversation messages, tool results, Runs, Results,
evidence, and checkpoints as a dependency graph and must never leave a Result
without its provenance.

## Code placement

No package is created until working code needs it.

| Responsibility | Placement |
| --- | --- |
| Pure Conversation, tool-call, Result, and source-attribution contracts | `models/conversation.py`, `models/results.py`, plus compatible extensions to `models/runtime.py` and API transport models |
| Bounded Conversation Agent loop and research-tool protocol consumption | `agents/conversation.py` and focused supporting modules under `agents/` |
| Agent-callable research facade protocol and tool schemas | `tools/research.py`; it delegates to injected application services |
| Result validation/materialization policy | `evidence/results.py`; it depends on models and evidence contracts, never LangGraph |
| Conversation/Result/event/association persistence and conversation-model provider adapter | `infrastructure/` |
| Research execution plus the backend Conversation/Run/Result lifecycle coordinator | extend existing `orchestration/`; it calls injected stores/services and does not place consistency decisions in model prompts |
| MCP and HTTP transport, authentication, dependency composition | `interfaces/mcp.py`, existing API/composition modules |
| Conversation and Result user interface | extend the existing React client after API behavior is stable |

`interfaces` assembles the MCP server and HTTP surfaces from the existing
Worker/RunHarness, lifecycle coordinator, and Result service. Infrastructure
adapters must not import `agents`, `evidence`, `tools`, or `orchestration`;
they implement protocols declared inward.

## Delivery phases

### Phase 1: durable Conversation foundation

1. Add strict domain and transport models for Conversation and tool-call audit.
2. Add the SQLite migration/store and authorization-aware Conversation service.
3. Reuse the Context Harness message store and compaction path; add the minimum
   Context association/query support without overloading a research Run Context.
4. Add create/list/get/archive/unarchive and append/read message API behavior.
5. Cover owner/workspace isolation, stable ordering, idempotency, restart
   persistence, and old-database migration.

**Gate:** two users in two workspaces cannot discover or mutate each other's
Conversations or Context data; a restarted API reconstructs the same ordered
history, and original messages survive compaction.

### Phase 2: Run association, Research Results, and lifecycle consistency

1. Extend new Research Runs with optional Conversation/origin/continuation
   metadata and submitted user-message sequence while preserving old
   `research-run/*` payload readability.
2. Persist one immutable Result from every normally completed workflow,
   including supported/partial/insufficient/conflict state, copied cited
   evidence, unresolved questions, and exact version scope.
3. Add the backend lifecycle coordinator, Conversation Event store, serialized
   Turn queue, compare-and-set auto-presentation decision, and durable operation
   keys. The Agent never owns these consistency transitions.
4. Add authorized Result list/get and prior-Result snapshots for follow-up Runs.
5. Update Run inspection to expose Result ID and Conversation linkage without
   broadening evidence visibility.

**Gate:** one Conversation can own multiple independently resumable Runs; a
completed result remains readable after process restart without loading its
checkpoint; insufficient evidence still produces a valid Result; changed
conditions create a new Run/Result and preserve the old pair; repeated
completion delivery for one Run attempt creates at most one Result revision and
one presentation Turn.

### Phase 3: research facade and MCP server

1. Implement `start_research`, `get_research_status`, and
   `get_research_result` over the existing Worker, RunHarness, lifecycle
   coordinator, and Result service, using strict schemas and safe error codes.
2. Implement the 10-second wait as observation of the same durable Run, never
   as a second inline execution path.
3. Add the internal Streamable HTTP MCP transport with trusted authenticated
   actor construction or service delegation and the same default-deny policy
   as HTTP.
4. Add contract tests proving MCP and application-service calls have equivalent
   domain results and cannot select another identity or workspace through tool
   arguments.
5. Add a `specrag-mcp` entry point only when the working server exists; keep
   `pyproject.toml` authoritative and document local startup/transport choice.

**Gate:** an MCP client can submit, poll, and read a result across process
restart; the resulting Run follows the same event, corpus binding, evidence,
and authorization path as an HTTP-submitted Run.

### Phase 4: Conversation Agent

1. Add one bounded turn state with maximum model turns, tool calls, elapsed
   time, cancellation, and an explicit stop reason.
2. Assemble the active prompt from Conversation summary/recent history,
   authorized Memory recall, selected prior Result metadata, and tool schemas.
   Mark recalled Conversation/Memory/Result text as context, not Evidence.
3. Persist assistant messages, tool proposals, arguments, results, failures,
   and Run links before advancing the loop. Make replay after interruption
   idempotent.
4. Enforce the policy that specification-backed judgments require the research
   service or a valid applicable historical Result. Conversation Agent
   responses may present completed Result content
   but cannot fabricate citations or reinterpret a pending Run as complete.
5. Record every assistant response source kind and Result references, then
   validate evidence references before publishing a technical response.
6. Support direct ordinary conversation without allocating a Research Run.
7. Consume internally triggered auto-presentation jobs as new Agent Turns; the
   lifecycle coordinator, not the model, decides whether such a job is valid.

**Gate:** a multi-turn acceptance case starts with ordinary clarification,
launches research, reads the result, modifies a condition, launches a related
Run, and correctly distinguishes both Results after restart and compaction.

### Phase 5: product surface and hardening

1. Change the frontend entry from the current Run-first desk to a Conversation
   list and thread while preserving Run lifecycle, event, evidence, and admin
   panels as linked views.
2. Load a Conversation projection and reconnect its ordered event stream after
   reload, filling sequence gaps before applying live updates. Surface queued/
   running/failed/completed tool states inline in the originating turn.
3. Add conversation-specific evaluation: tool-use routing, forbidden direct
   normative claims, context carry-over, condition preservation, Result
   selection, citation fidelity, and long-thread compaction recovery.
4. Add operational limits for conversation turns and tool calls, MCP request
   size/timeouts, per-user concurrency, audit redaction, and retention.
5. Update README, architecture diagrams, current status, and supported CLI/API/
   MCP workflows as each slice becomes real.

**Gate:** the canonical quality check passes, the browser acceptance flow works
after process restart, and an evaluation set demonstrates both direct-chat and
research-required routing without weakening existing answer/evidence metrics.

## Compatibility and migration

- Existing Runs are not backfilled into synthetic Conversations. They remain
  visible through current Run history and may later be explicitly attached by
  an authorized migration operation.
- Existing Run schema versions remain readable. Only newly created Runs carry
  Conversation linkage and produce Results after the materializer is enabled.
- The current `/api/v1/workspaces/{workspace}/runs` API and CLI research path
  remain supported during the transition. The Conversation UI calls the same
  underlying Run service.
- Existing `ContextRecord.run_id` remains valid for Researcher tool-result
  contexts. Conversation Contexts use the Conversation record's `context_id`;
  they are not forced into a fake Run.
- Existing Memory and SpecFact data and review flows are unchanged. Any later
  extraction from Conversation or Results must create pending proposals and
  use the existing review authorization.

## Verification matrix

Every implementation phase adds focused tests before broad acceptance:

- domain validation and state transitions;
- SQLite migration, restart, idempotency, and concurrency ordering;
- creator/workspace authorization for Conversation, Run, Context, Result, tool
  results, and MCP calls;
- administrator lifecycle control without private message/Result content read;
- exact Release/version/source/hash/locator preservation in Results;
- no evidence creation from Conversation, Memory, Result prose, or model text;
- Run cancellation, Resume, Retry, worker lease recovery, and Result
  materialization exactly once;
- same-Run completion inside or after the 10-second MCP wait boundary;
- auto-presentation compare-and-set behavior with new messages, active Turns,
  concurrent Run completion, duplicate delivery, and process recovery;
- context compaction that preserves canonical history and Run/Result links;
- simple versus complex routing through the existing workflow;
- provider failure taxonomy and safe transport projection;
- frontend reload/reconnect behavior and multi-turn acceptance;
- regression of existing research, evaluation, Memory, SpecFact, corpus, API,
  worker, and package-boundary tests.

Run `python scripts/check.py` before finishing every implementation slice.

## Explicitly deferred

- Cross-workspace Conversation sharing or public Result publication.
- Collaborative concurrent editing of one Conversation.
- Automatic approval of Memory or SpecFacts from chat content.
- Replacing SQLite with an external broker/database or scaling the MCP server
  across multiple nodes.
- Automatic whole-Run replay and automatic hard deletion/garbage collection.
- Private-document upload or a corpus other than the supported 3GPP source.
- Treating an LLM-generated Conversation summary as authoritative knowledge.

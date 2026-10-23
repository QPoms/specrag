# Conversation-centered research platform detailed design

Status: implemented baseline for plan 035
Updated: 2026-10-23

## 1. Purpose and scope

This document turns
`docs/plans/035-conversation-centered-research-platform.md` into an
implementation contract. It covers the first supported Conversation product
slice: persistent private conversations, bounded Conversation Agent Turns,
internal Streamable HTTP MCP research tools, unified Research Runs, durable
Research Results, automatic asynchronous result presentation, ordered
Conversation Events, and browser recovery.

The existing Research Workflow, RunHarness, Run Worker, Context Harness,
Memory, SpecFact, corpus registry, workspace identity, and SQLite database are
extended in place. This design does not introduce a second retrieval pipeline,
a second message history, or an Agent-owned reliability mechanism.

## 2. Architectural authority

The system has three deliberately different responsibilities:

| Layer | Owns | Must not own |
| --- | --- | --- |
| Conversation Agent | Understanding the current turn, deciding between direct conversation and MCP tools, and composing a response | Run/result persistence, authorization, ordering, idempotency, completion recovery, or evidence sufficiency |
| Research Workflow | Routing, planning, retrieval, source validation, sufficiency, gaps, and evidence-backed synthesis | Conversation lifecycle, user-visible message ordering, MCP authentication, or automatic reply decisions |
| Backend application services | Conversation/Run/Result association, authorization, durable jobs/events, atomic transitions, idempotency, and conditional Agent Turn scheduling | Model judgment about specification facts or inline retrieval algorithms |

The backend remains authoritative even when the model proposes a tool call.
Every external or model-driven operation is validated and persisted before it
can advance a lifecycle.

The logical component design is stored in
[`conversation_research_components.mmd`](conversation_research_components.mmd).

Text summary: the Conversation Agent reaches the existing research application
only through the internal MCP boundary, while durable queues and backend
services coordinate both sides through SQLite.

## 3. Existing implementation reuse

| Existing implementation | Reuse and required extension |
| --- | --- |
| `RunHarness.submit` and `RunWorker` | Remain the only asynchronous Run submission/execution path; add idempotent request keys, Conversation input snapshot fields, and atomic terminal outcome publication |
| `ResearchApplication` and LangGraph workflow | Remain the only research engine; adapt route budgets for light/full execution without creating another workflow |
| `RunEvent` | Continues to describe internal research execution; it is referenced, not copied, by Conversation Events |
| `ContextHarness` and `context_messages` | Remain the canonical message body, compaction, tool-result storage, and reconstruction path |
| `AccessPolicy` and `ActorContext` | Gain owner-private Conversation/Result permissions and separate Run lifecycle management from Run content reads |
| `SQLiteRunQueue` | Provides the lease/recovery pattern for new completion and Conversation Turn queues |
| `RunInspectionResponse` | Remains supported for existing Run-first clients; new Conversation surfaces read `ResearchResult` instead of checkpoint projections |
| `WorkspaceServices` | Remains the composition root and wires stores, lifecycle services, MCP transport dependencies, and Workers |

One current behavior must change: a workflow that returns normally with
`insufficient_evidence`, `iteration_budget`, `tool_budget`, or `replan_budget`
is an execution-complete Run, not an infrastructure failure. Its evidence
quality is represented by `ResearchResult.evidence_status`. Exceptions,
cancellation, and elapsed-time interruption retain failed/cancelled/timed-out
Run states.

## 4. Domain model

The aggregate relationships are shown in
[`conversation_research_data_model.mmd`](conversation_research_data_model.mmd).

Text summary: a private Conversation owns one Context, ordered messages, Turns,
tool calls, events, and research-link projections; each link binds one Run and
its independently readable Result revisions.

### 4.1 Conversation

`Conversation` is an owner-private aggregate:

| Field | Contract |
| --- | --- |
| `schema_version` | `conversation/v1` |
| `conversation_id` | opaque `CONV-<32 hex>` |
| `workspace_id`, `owner_user_id` | immutable trusted scope |
| `context_id` | immutable one-to-one Context Harness binding |
| `status` | `active` or `archived` |
| `title` | optional user-visible label; never research evidence |
| `revision` | monotonic optimistic-concurrency value |
| `last_message_sequence` | latest canonical message sequence |
| `last_user_message_sequence` | latest user message sequence used by automatic reply CAS |
| `active_turn_id` | nullable serialized Agent Turn owner |
| `active_scope_revision_id` | nullable confirmed Conversation research scope |
| timestamps | created and updated UTC instants |

Archiving prevents new user or automatic Turns but does not delete messages,
Runs, Results, or evidence. Unarchive is an authorized owner action.

### 4.2 Conversation message payload

`ContextMessage` remains the storage record. Conversation code validates its
`content` as a versioned `conversation-message/v1` envelope:

| Field | Contract |
| --- | --- |
| `text` | user-visible content |
| `source_kind` | assistant only: `direct_conversation`, `research_result`, or `historical_result_readback` |
| `research_result_ids` | Results supporting the assistant message |
| `run_ids` | related pending or terminal Runs |
| `citations` | evidence IDs actually cited in the text |
| `status_kind` | optional non-normative queued/running/failure notification |

For `research_result` and `historical_result_readback`, every citation must
exist in an authorized referenced Result. `direct_conversation` cannot carry
evidence citations. Tool invocation alone never changes the source kind to a
verified research response.

### 4.3 Conversation Turn

A `ConversationTurn` is one bounded Agent execution, triggered either by a user
message or by an internal Result presentation job:

| Field | Contract |
| --- | --- |
| `turn_id` | opaque stable ID |
| `conversation_id` | owning Conversation |
| `trigger` | `user_message` or `research_result_ready` |
| `trigger_message_id` | required for user Turns |
| `trigger_result_id`, `trigger_run_id` | required for automatic presentation |
| `submitted_user_sequence` | message watermark captured at creation |
| `status` | `queued`, `running`, `completed`, `failed`, `cancelled` |
| budgets | max model steps, tool calls, research submissions, and elapsed seconds |
| `operation_key` | unique logical execution identity |
| output/failure fields | reply message ID or safe failure metadata |

The state machine is stored in
[`conversation_turn_state.mmd`](conversation_turn_state.mmd).

Text summary: a leased Turn either commits one terminal outcome or returns to
queued after lease expiry; it never waits in memory for a long Research Run.

Recommended first defaults are four model steps, six total tool calls, one new
research submission per user Turn, and 60 seconds of Turn elapsed time. An
automatic presentation Turn receives only Result read tools and cannot start a
new Research Run without a new user request.

### 4.4 Conversation tool call

`ConversationToolCall` records the Agent-facing call separately from the
long-running task it may create:

- states: `proposed`, `running`, `completed`, `failed`;
- immutable tool name and canonical argument digest;
- validated argument payload and safe result reference;
- optional `tool_result_id`, `run_id`, and `result_id`;
- stable operation key derived from Conversation, Turn, and call IDs.

An asynchronous `start_research` MCP request still completes as a tool call
when it returns `accepted`. The related Run continues independently. Its later
status is represented by Run/Result linkage and Conversation Events, not by
reopening or mutating the old tool response.

### 4.5 Conversation research link

`ConversationResearchLink` is the small mutable projection that connects a
Conversation interaction to a long-running Run without copying Run Events. It
contains:

- Conversation, origin message/Turn, tool-call, and Run IDs;
- submitted user-message sequence;
- latest published Result ID;
- task status projected from the Run;
- presentation status: `pending`, `queued`, `presented`, or `skipped`;
- safe skip/failure reason and timestamps.

The Run and Results remain authoritative for research execution and content.
The link answers only how that work appears in the Conversation. A Result first
becomes `pending`. It changes to `queued` only when a serialized presentation
Turn is durably created, to `presented` only when a cited assistant message is
committed, and to `skipped` only after a user-message/scope conflict or explicit
non-presentation decision.

### 4.6 Confirmed Conversation research scope

`ConversationScopeRevision` stores user-confirmed research context separately
from an LLM summary:

- objective/topic text;
- explicit conditions and assumptions;
- requested version policy and selectors;
- source user message IDs;
- revision, creation time, and predecessor ID.

It is contextual input, not evidence. A revision may be created from an
explicit UI scope selection or from an Agent proposal that is echoed in a
visible assistant confirmation. A summary cannot silently update it.

Effective scope is resolved in this order:

1. explicit current-message scope;
2. active confirmed Conversation scope;
3. explicitly referenced historical Result scope;
4. Workspace/system defaults.

Memory can suggest context but cannot select a Release or override user
conditions. Material ambiguity stops the Turn for clarification.

### 4.7 Research input snapshot

Extend `ResearchRequest`/`ResearchRun` compatibly with:

- normalized `objective` and final `question`;
- immutable ordered `conditions`;
- requested `SpecificationScope` supporting Release and exact/mixed selectors;
- `prior_result_refs`, each containing an authorized Result ID and content hash;
- Conversation/origin/Turn IDs and submitted user sequence;
- submission operation key.

`RunHarness._prepare` resolves this request to the existing exact
`corpus_versions` binding before execution. Resume reloads only this frozen
snapshot. Retry and changed conditions create a new Run.

### 4.8 Research outcome and Research Result

`ResearchOutcome` is an internal immutable staging artifact captured when a
workflow attempt terminates. It contains only the state required to build a
Result: synthesis, candidate and validated evidence, sufficiency assessments,
gaps, stop reason, conditions, and exact corpus bindings. It is not an API
object and does not replace a Result.

`ResearchResult` is the durable public research artifact:

| Field | Contract |
| --- | --- |
| `schema_version` | `research-result/v1` |
| `result_id`, `run_id`, `run_attempt` | immutable provenance identity |
| scope | workspace, owner, and optional Conversation |
| `publication_kind` | `final` or `validated_partial` |
| `evidence_status` | `sufficient`, `partial`, `insufficient`, or `conflict` |
| `answer` | safe user-facing result; may explicitly state that no conclusion is supported |
| `evidence` | copied source-valid evidence plus validation relationship/status |
| `evidence_ids` | ordered IDs cited by `answer` |
| applicability | conditions, requested scope, and exact corpus versions |
| unresolved work | gaps and unanswered questions |
| prior context | immutable prior Result references and hashes |
| integrity | source outcome hash, result content hash, and creation time |
| revision | immutable revision within one Run; optional superseded Result ID |

Normally completed workflows publish a Result even when evidence is
insufficient or conflicting. If no validated synthesis exists, the Result
service produces a deterministic safe answer from status and gaps; it never
promotes candidate evidence or model drafts into conclusions.

`run_attempt` is persisted directly on new Run schemas, begins at zero, and is
incremented by Resume. Historical Run payloads read it as zero.

Abnormal Runs publish no Result by default. A future or explicit validated-stage
projector may publish `validated_partial`. If the same Run is later resumed and
completes, it creates a new immutable Result revision and supersedes only the
active Result for that Run; the earlier partial revision remains auditable.

Run and Result states are shown in
[`research_run_result_state.mmd`](research_run_result_state.mmd).

Text summary: execution completion is independent of sufficient, partial,
insufficient, or conflicting evidence, while abnormal termination publishes a
Result only from an explicitly validated stage.

## 5. Durable lifecycle and transaction boundaries

### 5.1 User-message transaction

`ConversationService.append_user_message` executes one `BEGIN IMMEDIATE`
transaction:

1. authorize owner and require `active` status;
2. compare `expected_revision` when supplied;
3. idempotently resolve the client operation key;
4. allocate the next message and Conversation Event sequences;
5. append the canonical `context_messages` user record;
6. create a queued `ConversationTurn` and its durable job;
7. update Conversation revision and user/message watermarks; and
8. append `message_appended` and `turn_queued` events.

The HTTP response is `202 Accepted` with Conversation, message, Turn, revision,
and event cursors. Agent execution is always performed by the Turn Worker, not
inside the request handler.

### 5.2 Agent tool-call boundary

Before an MCP call, the Turn Worker commits `ConversationToolCall.running` and
a `tool_call_started` event. The MCP client injects the persisted operation key
and trusted delegation outside model-controlled arguments.

The network call cannot share a database transaction. Therefore
`start_research` is idempotent by scoped operation key. Replaying the same call
returns the same Run; reusing the key with different canonical arguments is a
conflict. After the response, the Worker stores the full payload through
Context Harness, commits the bounded tool message, updates the tool-call row,
and appends the corresponding event.

### 5.3 Atomic Run finalization and completion outbox

Replace the current separate terminal `put` and `run_stopped` writes with one
Run-store protocol operation. In one SQLite transaction it:

1. persists the terminal Run state;
2. inserts the attempt-aware `run_stopped` event;
3. stores the `ResearchOutcome` staging payload and hash when available; and
4. inserts or preserves a `run_completion_job` keyed by Run and attempt.

This is the transactional outbox. A process crash cannot leave a terminal Run
without a recoverable completion action. The Run Worker completes its own job
only after this transaction commits.

### 5.4 Completion reconciliation transaction

The Lifecycle Worker leases a completion job and asks the pure Result service
to validate/materialize the outcome. The lifecycle store then performs one
transaction that:

- idempotently inserts the Result revision, if any;
- updates the related Conversation research link to `pending`;
- appends `run_status_changed` and optional `result_available` events;
- evaluates the automatic presentation compare-and-set;
- when eligible and no Turn is active, creates exactly one automatic Turn and
  Turn job, marks the link `queued`, and appends `auto_reply_queued`;
- when another Turn is active but the user watermark still matches, leaves the
  link `pending` for terminal-Turn rescheduling; or
- when the watermark/scope is invalid, marks it `skipped` and appends
  `auto_reply_skipped`.

The eligibility update succeeds only when the Conversation is active, its
owner/workspace binding matches the Run, current membership still permits use,
and `last_user_message_sequence` equals the submitted watermark. A unique
operation key `present-result:<conversation_id>:<result_id>` prevents
duplicates.

This two-step eligibility/scheduling rule closes the ten-second boundary race.
If a Result arrives after MCP returned `accepted` but before the originating
Turn commits its acknowledgement, the Result remains pending. The terminal
Turn transaction clears the active slot and immediately retries scheduling. If
several Results are pending, it queues the earliest Result-available event and
repeats after each presentation Turn. A new user message atomically marks older
pending presentations skipped before its own Turn is queued.

Membership removal does not discard an already executed Run or its Result. The
system may persist the owner-private artifact from its frozen binding, but it
skips automatic model execution and denies user reads until authorization is
restored.

### 5.5 Turn terminal transaction

A Turn completes by atomically appending its assistant message, validating
source/citation references, marking any Result it presented, appending
message/Turn events, setting the terminal Turn state, clearing `active_turn_id`,
and selecting the next queued user Turn or eligible pending Result. A failed
Turn records only the safe failure projection and also releases the active slot.

## 6. Synchronous and asynchronous research behavior

The full interaction is shown in
[`conversation_research_sync_async.mmd`](conversation_research_sync_async.mmd).

Text summary: `start_research` always queues one durable Run, then observes it
for at most ten seconds; the Conversation Turn receives either that Run's
Result, an accepted status, or an abnormal terminal status.

The MCP server does not execute the graph inline. After idempotent submission it
polls the durable Run/Result projection at a short bounded interval until one
of three conditions occurs:

- a Result is available: return `completed` with the Result;
- the Run is failed/cancelled/timed out: return `terminal` with a safe status;
- ten seconds elapse: return `accepted` with queued/running status.

The polling deadline is not a Run deadline and does not alter Run budgets. An
MCP transport timeout also does not cancel the Run; a retry with the injected
operation key resolves the original Run.

The asynchronous completion path is shown in
[`conversation_research_completion.mmd`](conversation_research_completion.mmd).

Text summary: the lifecycle worker commits the Result and pending presentation
before a CAS may enqueue a new Turn; active Turns defer scheduling, user-message
conflicts skip it, and replay converges on one logical outcome.

## 7. Conversation Agent design

### 7.1 Bounded loop

The Agent loop is framework-independent and lives under `agents/`. Its explicit
state contains Turn ID, message watermark, reconstructed context reference,
remaining model/tool/research budgets, tool records, draft response, stop
reason, and safe error metadata. It does not import LangGraph.

Supported stop reasons are:

- `response_completed`;
- `clarification_required`;
- `research_submitted`;
- `model_step_budget`;
- `tool_call_budget`;
- `elapsed_time_budget`;
- `cancelled`;
- `agent_error`.

### 7.2 Context assembly order

`ConversationContextAssembler` extends the existing Context Harness and fills a
model budget in this priority order:

1. system policy, response contract, and MCP tool schemas;
2. current trigger message or Result;
3. active confirmed Conversation scope revision;
4. explicitly referenced Research Result summaries and IDs;
5. authorized Core/Topic Memory, marked untrusted;
6. recent complete Conversation turns;
7. active structured history summary.

Full Result evidence and full tool payloads remain durable and are read through
bounded tools/pages only when needed. A compacted summary preserves literal
Conversation, Run, Result, message, and condition references, but never becomes
their authoritative store.

First-phase historical Result selection is conservative: explicit user/Agent
references, the Result attached to the replied-to message, and recent Results
in the same Conversation. Broad semantic search across all Results is deferred.

### 7.3 Response validation

The Agent returns a strict structured proposal containing text, source kind,
Result IDs, Run IDs, and cited evidence IDs. A successful `start_research`
records the exact submitted objective, conditions, scope, and source user
message as the next immutable active scope revision. Before message publication,
the application service verifies:

- every Result is authorized and applicable to the stated scope;
- every cited evidence ID occurs in those Results;
- every declared citation belongs to the referenced Results;
- a pending Run is not presented as completed;
- direct conversation contains no claimed evidence citations; and
- every recorded scope revision references its literal source user message.

Prompt policy and evaluation prohibit unsupported normative answers. The first
slice does not claim that a heuristic classifier can perfectly identify every
3GPP assertion; tool-use routing, unsupported assertions, and citation validity
are explicit evaluation dimensions.

## 8. MCP contract and authentication

### 8.1 Trusted invocation envelope

The internal MCP client sends a short-lived server-issued delegation credential
in the HTTP authentication layer. It binds subject user, Workspace,
Conversation, origin message/Turn, operation key, issued/expiry times, and a
nonce. The MCP server validates it and rechecks current membership before
constructing `ActorContext`.

The user bearer token is never placed in model context. Model-visible tool
arguments contain no user ID, role, Workspace ID, session ID, or permission.
Conversation/Turn binding and the submission operation key are injected by the
client adapter outside the tool schema.

The first implementation uses a stateless HMAC-SHA256 signed bearer delegation
with a shared secret of at least 32 bytes. The Conversation tool executor issues
it for 60 seconds, the verifier rejects lifetimes over five minutes, and every
MCP request rechecks expiry and current membership. The token is never stored in
Conversation audit or model context. This reuses the security posture of local
sessions without adding a JWT dependency.

### 8.2 `start_research`

Model-visible input:

- `question` and optional normalized objective;
- ordered conditions;
- requested Release/exact/mixed version scope;
- authorized prior Result IDs;
- answer language and detail level.

Server-controlled input includes identity, Conversation/Turn/origin IDs,
operation key, budgets, and the fixed ten-second wait. Output is a tagged union:

- `completed`: Run ID, completed execution status, and Research Result;
- `accepted`: Run ID and queued/running status;
- `terminal`: Run ID, failed/cancelled/timed-out status, and safe recovery data.

### 8.3 `get_research_status`

Input is a Run ID. Output contains execution status, safe progress summary,
stop/failure projection, and current Result ID if present. It never exposes a
checkpoint or raw provider exception.

### 8.4 `get_research_result`

Input is a Run or Result ID. Output contains the authorized Result answer,
evidence status, applicability, exact versions, unresolved questions, and
evidence. Context Harness stores the complete MCP payload and returns a bounded
page to the Agent when it exceeds the configured tool-result limit.

## 9. Authorization design

Extend the permission vocabulary rather than encoding exceptions in handlers:

| Permission | Owner | Workspace admin by role alone | Internal worker |
| --- | --- | --- | --- |
| `conversation.create` | allowed | allowed only for own Conversation | denied |
| `conversation.read/write/archive` | allowed | denied for another owner | lifecycle worker only with frozen binding |
| `conversation.events.read` | allowed | denied for another owner | denied |
| `result.read` | allowed | denied for another owner | lifecycle/Turn worker with frozen binding |
| `run.content.read` | allowed | denied for another owner | Run/lifecycle worker with frozen binding |
| `run.lifecycle.manage` | allowed | allowed under existing admin policy | Run Worker |

The legacy content read is split from a `RunMetadataResponse`. An administrator
can inspect status, timing, safe failure kind, and cancel a Run without receiving
the member's question, Context, events, answer, Result, or evidence. Resume and
Retry remain owner-private because their current responses and execution path
use research content. Unknown and inaccessible private resources retain
indistinguishable not-found responses.

MCP authorization first validates delegated scope, then loads the resource and
applies the same owner/workspace policy. Resource IDs never grant access.

## 10. Persistence schema

Migrations 7 through 10 add real tables only with their corresponding services;
older schema versions upgrade monotonically without synthetic Conversations.

### 10.1 New tables

| Table | Important keys and constraints |
| --- | --- |
| `conversations` | PK Conversation; unique Context; indexed workspace/owner/status; revision and user/message watermarks |
| `conversation_scope_revisions` | PK revision ID; unique Conversation/revision; immutable payload and source message references |
| `conversation_turns` | PK Turn; unique operation key; indexed Conversation/status; trigger/result/run references |
| `conversation_tool_calls` | PK call; unique scoped operation key; indexed Turn and Run; canonical argument digest |
| `conversation_research_links` | unique Run; origin message/Turn/call, user watermark, latest Result, task state, and presentation state |
| `conversation_events` | PK event; unique Conversation/sequence and Conversation/operation key |
| `conversation_turn_jobs` | PK Turn; queued/leased state, owner, expiry, attempts, timestamps |
| `run_outcomes` | unique Run/attempt; immutable hash and staging payload |
| `run_completion_jobs` | unique Run/attempt; queued/leased state and recovery fields |
| `research_results` | PK Result; unique Run/attempt; active read is the highest published attempt; indexed Conversation/owner/workspace |

`runs` gains nullable indexed Conversation/origin/Turn IDs, submitted user
sequence, submission operation key, explicit attempt number, and
schema-versioned input snapshot fields. Old rows remain valid, read attempt as
zero, and are not backfilled into synthetic Conversations.

### 10.2 Canonical versus projection data

- `context_messages` is the canonical message body.
- `conversations` stores aggregate watermarks and active pointers.
- Conversation Events are an ordered change log for clients, not the source of
  message or Result content.
- Run Events remain the research execution audit.
- `run_outcomes` is internal staging and may be retained with the Run for audit.
- `research_results` is the long-term read model and must be sufficient without
  loading a LangGraph checkpoint.

All JSON payloads use explicit schema versions. Foreign keys and unique
operation keys enforce scope association and idempotency in addition to
application validation.

## 11. HTTP and Conversation Event surface

Initial owner-private HTTP endpoints:

```text
POST   /api/v1/workspaces/{workspace}/conversations
GET    /api/v1/workspaces/{workspace}/conversations
GET    /api/v1/workspaces/{workspace}/conversations/{conversation_id}
POST   /api/v1/workspaces/{workspace}/conversations/{conversation_id}/messages
POST   /api/v1/workspaces/{workspace}/conversations/{conversation_id}/archive
POST   /api/v1/workspaces/{workspace}/conversations/{conversation_id}/unarchive
GET    /api/v1/workspaces/{workspace}/conversations/{conversation_id}/events
GET    /api/v1/workspaces/{workspace}/conversations/{conversation_id}/results
GET    /api/v1/workspaces/{workspace}/research-results/{result_id}
GET    /api/v1/workspaces/{workspace}/run-metadata
GET    /api/v1/workspaces/{workspace}/runs/{run_id}/metadata
```

Message creation accepts a client operation key and optional expected
Conversation revision. A stale revision returns `409 conversation_conflict`;
replaying an identical operation returns the original accepted record.

The Conversation SSE endpoint accepts `since` and `Last-Event-ID`, emits IDs as
the Conversation sequence, and closes or heartbeats using the current Run SSE
conventions. Reconnection loads a current projection first, then requests all
events after the last applied sequence. Clients deduplicate by event ID and
must not infer canonical content solely from event payloads.

Initial event types:

- `conversation_created`, `conversation_archived`, `conversation_unarchived`;
- `message_appended`;
- `turn_queued`, `turn_started`, `turn_stopped`;
- `tool_call_started`, `tool_call_completed`, `tool_call_failed`;
- `run_linked`, `run_status_changed`;
- `result_available`;
- `auto_reply_deferred`, `auto_reply_queued`, `auto_reply_presented`,
  `auto_reply_skipped`.

Event payloads contain IDs, sequence/revision, safe status, and timestamps. Full
message text, evidence, provider responses, and credentials are fetched only
through authorized resource endpoints.

## 12. Failure and recovery rules

| Failure | Recovery and invariant |
| --- | --- |
| MCP response lost after Run creation | Replay with the injected operation key returns the same Run |
| MCP ten-second wait expires | Return accepted; do not cancel or clone the Run |
| Run Worker dies | Existing lease expiry and checkpoint Resume recover the same Run |
| Process dies after graph return | Atomic terminal outcome/outbox makes completion reconciliation recoverable |
| Lifecycle Worker dies | Completion lease expires; idempotent Result/event/Turn keys make replay safe |
| Conversation Agent dies | Turn lease expires and the same Turn is replayed from durable tool records |
| Duplicate completion event | Unique Run/attempt outcome and presentation operation keys converge |
| User posts while Run executes | User watermark changes; Result is attached but automatic presentation is skipped |
| Multiple Runs complete together | Links remain pending and serialize by Result-available event order; each Result gets at most one presentation Turn |
| Conversation archived | Result persists; auto presentation is skipped and no new user Turn is accepted |
| Membership removed | Internal persistence completes; automatic model execution and reads are denied |
| Context compaction | Original messages/Results remain; summary stores references and recent turns remain whole |

### 12.1 Operational bounds and retention

- A Turn defaults to four model steps, six MCP calls, one research submission,
  and 60 seconds; each limit has a strict model-level upper bound.
- Turn jobs serialize per Conversation and default to at most two active leases
  per Workspace user across Conversations.
- `ResearchToolRequest` caps question/objective text at 20,000 characters,
  conditions and prior Results at 50 entries, and each condition at 4,000
  characters. The MCP client timeout defaults to 30 seconds; the server's
  same-Run observation window remains ten seconds.
- Tool arguments are strict-validated before audit persistence, so model-added
  identity, role, Workspace, or unknown fields are rejected and not recorded.
  Tool failures persist a safe error category rather than exception/provider
  payloads.
- Full canonical messages, Results, evidence, and tool results remain durable;
  the model receives token-bounded projections. Archiving is the supported
  first-release retention action. Automatic hard deletion is deferred.

## 13. Implementation order

Implement in vertical slices, keeping the current Run-first API usable:

1. **Contracts and policy:** Conversation, Turn, Event, scope revision,
   ResearchOutcome/Result, permission split, and pure state-transition tests.
2. **Conversation persistence:** migration, owner-private store/service,
   transactional message append, Turn queue, Context Harness integration, and
   CRUD/event APIs.
3. **Run/Result boundary:** Research input snapshot, corrected normal Run
   terminal mapping, atomic Run outcome/outbox, Result materializer, and
   authorized Result reads.
4. **Lifecycle recovery:** completion worker, Conversation projection/events,
   automatic presentation CAS, duplicate/concurrent completion tests.
5. **MCP:** internal Streamable HTTP server/client, trusted delegation,
   idempotent submit, ten-second observation, and the three tools.
6. **Conversation Agent:** bounded loop, context assembler, response validator,
   tool-result persistence, direct/synchronous/asynchronous Turns.
7. **Frontend:** Conversation list/timeline, status cards, Result/evidence
   reader, SSE gap recovery, history reload, and existing Run controls.
8. **Acceptance and evaluation:** all twelve product scenarios, unsupported
   normative-answer evaluation, existing research regressions, and canonical
   `python scripts/check.py`.

Each slice updates README/current status only when its supported workflow is
real. Replaced checkpoint-only answer projection remains for legacy Runs until
the compatibility window is explicitly closed.

## 14. Acceptance traceability

| Requirement | Primary verification |
| --- | --- |
| Simple research finishes inside ten seconds | one Run ID, direct Result response, cited assistant message |
| Research exceeds ten seconds | accepted response and later completion for the same Run |
| No later user message | exactly one automatic presentation Turn |
| Later user message exists | Result attached to origin and `auto_reply_skipped`, with no inserted answer |
| Concurrent/duplicate completion | pending links plus unique Result/event/Turn operation keys serialize every presentation |
| Insufficient evidence | completed Run plus `insufficient` Result and unresolved gaps |
| Changed conditions | new immutable Run snapshot and preserved prior Result |
| Compacted conversation | full Result/evidence readback by durable IDs |
| Admin privacy | lifecycle metadata/control allowed; private content denied |
| MCP identity spoof | identity fields absent from schema and delegated scope enforced |
| Ordinary versus normative response | direct Turn without MCP versus valid Result/citations |
| Frontend reconnect | projection plus ordered event catch-up reaches database state |

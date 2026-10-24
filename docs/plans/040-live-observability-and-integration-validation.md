# 040 — Live research observability and integration validation

Status: complete

## Goal

Close the observability gaps found by the 2026-10-24 live backend validation,
then verify the actual API → MCP → Worker → Conversation completion path before
returning to browser-level validation.

## Stage 1 — Complete model-call receipt propagation

- Capture safe model receipts around Router, Planner, Sufficiency Judge, and
  Synthesizer graph nodes.
- Preserve Researcher receipts when parallel worker results cross the fan-in
  boundary.
- Emit one durable `model_receipt` Run Event per logical model call without
  storing prompts, endpoints, credentials, or raw provider responses.
- Preserve missing provider usage as `null`; never silently convert missing
  usage to zero.

## Stage 2 — Parallel Researcher timing

- Retain each TaskExecution worker's wall-clock latency in its immutable worker
  result.
- Project task-level Researcher timing after fan-in, where durable event writes
  remain serialized.
- Aggregate concurrent workers by the maximum latency in a dispatch wave, then
  sum waves, so stage latency represents critical-path time rather than total
  parallel CPU time.
- Keep model-call latency separate from graph-node latency to avoid double
  counting.

## Stage 3 — Verification

- Add focused tests for receipt propagation, missing usage, serialized worker
  timing events, and concurrent-wave aggregation.
- Run `python scripts/check.py`.
- Run a live backend case and confirm non-zero/explicit Token telemetry and
  Researcher stage timing.

Completed on 2026-10-24. The focused suite passed 43 tests, the frontend
production build passed, and the full project gate passed all 313 tests in the
project `uv` environment. A live Release 17 Run persisted five model receipts
across Router, Researcher, Sufficiency Judge, and Synthesizer; complete provider
usage aggregated to 21,470 input and 1,518 output Tokens. The Researcher stage
reported 19.37 seconds instead of zero. The Run and Result completed and both
durable queues drained to zero.

## Stage 4 — Service-chain validation

- Start the authenticated API and internal Streamable HTTP MCP service with the
  shared delegation secret and existing Release 17 index.
- Submit a Conversation research request through the API-facing path.
- Verify the ten-second synchronous window hands off the same Run to the
  background Worker without duplication.
- Verify terminal Outcome, Result persistence, completion reconciliation,
  conditional automatic Conversation Turn scheduling, and ordered reconnectable
  Conversation Events.

The first live service-chain attempt on 2026-10-24 verified that the original
Conversation Turn stopped with a background-research acknowledgement after
26.68 seconds and that exactly one Run continued asynchronously. The Run
reached the fourth-wave task terminalization step, but final synthesis received
repeated HTTP 503 responses from the configured model gateway and exhausted its
bounded provider retries. The attempt exposed an observability gap: failed node
calls were captured in process but not emitted as durable model receipts before
the graph exception propagated. Conversation research remains completion-first
without a whole-Run deadline while workflow correctness and efficiency are
being optimized; a production time limit will be introduced only after those
issues are resolved. Stage 4 now includes fixing and regression-testing the
receipt gap before repeating the live chain.

The same attempt persisted 49 successful model receipts before the gateway
failure: 31 Researcher calls and 16 Sufficiency Judge calls for four tasks over
four waves. Completed tasks were being judged again whenever a sibling task
needed corrective retrieval. The remediation therefore also freezes a
completed task's sufficient assessment and validated evidence, and submits
only non-terminal tasks to later Judge waves.

The second live attempt confirmed that incremental judging reduced the second
wave from four Judge calls to three. It then exposed a domain-boundary failure:
one task-level assessment cited an evidence ID outside that task's source-valid
eligible set, and the deterministic validator aborted the whole Run. The
remediation keeps the strict citation boundary but converts that single
assessment to an explicit insufficient status and Gap, so untrusted model
output cannot be promoted and cannot crash unrelated task results.

The third live attempt passed the task join barrier with nine Judge calls
(4 + 3 + 2, versus repeated all-task assessment) and then encountered another
gateway-wide HTTP 503 outage during final synthesis. Because all usable source
material had already passed source validation and task-level selection, the
workflow now publishes a deterministic, claim-limited evidence digest when and
only when the synthesis generator reports a typed model/provider failure. The
digest cites only eligible evidence, explicitly says synthesis was unavailable,
and preserves unresolved task IDs; ordinary programming or state errors still
fail instead of being hidden by the fallback.

The fourth live attempt completed the full authenticated service chain with a
single immutable Run. The originating Conversation Turn returned the
background acknowledgement after 20.87 seconds, while the same Run continued
for 608.14 seconds and published a partial Result with 60 retained evidence
items. Lifecycle reconciliation emitted `result_available`, queued a distinct
internal presentation Turn, and persisted the final Result-linked assistant
message with 22 citations. All 16 Conversation Events were strictly ordered,
unique, and replayable from a midpoint cursor. The Run recorded 765,927 input
and 68,688 output Tokens; critical-path stage latency was available for Router,
Planner, parallel Researcher, Source Validator, Sufficiency Judge,
terminalization, and Synthesizer. No whole-Run deadline was applied. The full
quality gate passed all 316 tests before this live attempt.

## Stage 5 — Browser validation

- Start the existing frontend only after the service-chain test passes.
- Exercise simple conversation, asynchronous complex research, continued chat,
  and reconnect behavior through the browser.
- Record any remaining UI-only defects separately from backend lifecycle
  failures.

Completed on 2026-10-24 with a real headless Chrome session against the Vite
development server and authenticated API proxy. The Chinese login and research
workspace rendered correctly. The completed asynchronous service-chain
Conversation projected all three messages, its completed Run card, and all 22
Result citations. A newly created ordinary Conversation produced a direct
two-message exchange with zero research tool calls and zero Run cards. After a
hard page reload, the selected Conversation and both messages were restored
from the server. The browser recorded no console exception, visible capability
error, API request failure, or conversation error. The production frontend
build also passed.

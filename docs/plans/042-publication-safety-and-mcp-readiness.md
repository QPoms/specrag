# 042 - Publication safety and MCP readiness

Status: complete

## Goal

Make evidence status an application-owned publication decision and make the
internal MCP research service an observable, process-long runtime. The model
may phrase only source-backed content; it does not decide whether a result is
complete, qualified, conflicting, or unpublishable.

## Stage 1 - Publication safety

- Derive `full_synthesis`, `qualified_synthesis`, or `status_only` from task
  assessments and eligible evidence.
- Compose distinct versioned full and qualified prompt blocks from a shared
  grounding contract. Add the conflict restriction only when required.
- Send the Synthesizer the original question, covered task questions, and
  Judge-selected source-valid evidence. Do not send rejected candidates or
  Planner/search execution detail.
- Append unresolved/conflicting task disclosures deterministically from backend
  state. A model cannot omit or rewrite them.
- Do not call the Synthesizer when no publishable supported content exists.
- Preserve the existing `complete`/`partial` persisted synthesis wire values for
  compatibility while recording the internal publication decision in tests and
  events.

## Stage 2 - MCP lifecycle and readiness

- Keep one process-local RetrieverPool for the complete MCP process lifetime.
- Publish `starting`, `warming`, `ready`, or `failed` readiness without exposing
  credentials or internal exception text.
- Reject new research submissions with a stable unavailable response until the
  runtime is ready; status/result readback remains available.
- Record prewarm timing for corpus snapshot binding, local snapshot discovery,
  consistency validation, and retriever/vector/model construction.
- Verify consecutive prewarm/Run preparation reuses the same immutable
  retriever rather than rebuilding it.

## Stage 3 - Operator workflow and verification

- Document one supported long-running local process topology for API, MCP, and
  frontend.
- Add deterministic tests for all evidence-status combinations, prompt
  composition, mandatory qualification disclosure, MCP readiness, timing, and
  retriever reuse.
- Run the canonical quality gate. A live complex model/corpus run follows only
  after these deterministic gates pass.

## Stage 4 - Qualified-synthesis contract hardening

- Treat a partially supported task as both answerable for its supported
  propositions and unresolved for its missing relation or condition. The
  qualified prompt states this overlap explicitly instead of instructing the
  model to omit every unresolved task.
- Continue to require inline citation coverage for every task whose supported
  portion is published.
- Classify ineligible citations, mismatched inline citation declarations, and
  missing covered-task citations as model output-contract violations.
- If that model contract is violated after evidence has already passed source
  validation and Judge selection, publish the deterministic cited evidence
  digest with the same backend-owned qualification metadata. Do not convert a
  publishable qualified Run into an internal-error failure.
- Keep programming errors and inconsistent workflow state fail-closed; the
  fallback applies only to the explicit synthesis-output contract.
- Regress the live failure shape in which a task appears in both
  `covered_task_ids` and `unresolved_task_ids` and the model omits its citation.

## Non-goals

- Letting an LLM reinterpret Judge gaps as specification facts.
- Adding one prompt per evidence status.
- Starting an MCP process for each Research Run.
- Replacing the current embedding or reranking model before measurements show
  which cold-start stage dominates.

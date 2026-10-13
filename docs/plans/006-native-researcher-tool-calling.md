# Native Researcher tool calling

Status: complete

## Objective

Replace the Researcher's custom structured `actions` protocol with native
Responses API function calls while preserving the accepted
`researcher-controller/v1.12` evidence-review and retrieval behavior.

## Invariants

- The Tool Registry is the single source for retrieval tool descriptions,
  argument schemas, validation, and execution.
- Provider-specific translation remains under `infrastructure`.
- One model call controls each Researcher round.
- Retrieval calls alone consume the per-task action budget.
- Evidence selection, trusted specification scope, dependency context,
  finalization, and downstream validation behavior do not change.
- The Responses API remains stateless with `store=False`; function calls and
  outputs are replayed explicitly.
- The old custom action protocol is removed rather than retained behind a
  feature flag.

## Execution sequence

1. Prove the configured endpoint supports strict function tools, parallel
   calls, and explicit `function_call_output` replay.
2. Add provider-neutral per-tool argument contracts and Registry definitions.
3. Add native Responses API call/result transport and redacted receipts.
4. Replace the Researcher decision/action loop with one state-update control
   call plus native retrieval calls, or one terminal submission call.
5. Adapt the frozen prompt semantics to the native protocol and delete the
   superseded action models and tests.
6. Run unit, orchestration, full quality-gate, and staged live acceptance
   checks, including complex questions 039 and 050.
7. Update current-status and supported-workflow documentation.

## Result

The configured endpoint passed the strict-schema, parallel-call, and stateless
output-replay compatibility gate. Production now uses one native protocol: one
state-control function plus retrieval calls, or one submission function. The
Registry owns all seven retrieval schemas and execution; Responses-specific
translation remains in `infrastructure`. Provider call IDs replace generated
action IDs throughout audit records.

The first live batch exposed intermittent `invalid_encrypted_content` errors
when provider reasoning output was replayed. The supported stateless protocol
now replays only minimal function-call items and their matched outputs. A fresh
ten-simple-question run completed 10/10 with sufficient evidence, zero
replans, and no errors. Frozen-plan 039 met its retrieval acceptance condition.
Frozen-plan 050 retained the accepted direct evidence for all three tasks, but
the unchanged downstream Judge twice imposed an exhaustive-set requirement on
T1 and therefore did not reproduce the historical overall `completed` verdict;
that separate Judge boundary is documented in `current_status.md`.

## Acceptance

- All seven retrieval capabilities are invoked through native function calls.
- Tool results return through matched function-call outputs.
- Invalid calls remain bounded and auditable without consuming retrieval
  budget unless executed.
- Question 050 remains complete; question 039 retains its correct T3 evidence
  even if the known over-strict Judge verdict remains partial.
- `python scripts/check.py` passes.

# Judge-gap corrective retrieval budget

Status: complete

## Goal

Give a ResearchTask a bounded corrective retrieval phase after the
Sufficiency Judge reports an unresolved evidence gap. The existing Researcher
loop remains the only model-backed retrieval loop: on re-entry it reviews the
gap, rewrites it into concrete source-search objectives, and emits the next
tool calls. No second query-rewrite model or LangGraph node is introduced.

## Scope

1. Add an independent per-task `max_gap_tool_calls` allowance, defaulting to
   six calls, to the runtime budget and persist separate initial/corrective
   usage for auditability.
2. Make single-task and parallel dispatch select the phase-appropriate
   allowance; only retrieval calls consume the selected allowance.
3. Pass an explicit corrective mode and Judge gaps into the Researcher prompt,
   requiring the order: complete evidence review, answer-unit mapping, active
   evidence update, precise gap statement, then gap-driven query rewriting and
   tool calls.
4. Keep Planner, retrieval algorithms, Source Validator, Sufficiency Judge,
   and dependency semantics unchanged.

## Acceptance

- Initial and Judge-gap allowances are independent for every task.
- A task that exhausts its initial allowance can still receive its corrective
  allowance when an unresolved Judge gap exists; the corrective allowance is
  bounded and cannot be reused after it is exhausted.
- The parallel fan-out/fan-in path charges the same phase-specific budget as
  the single-task path.
- The prompt explicitly treats Judge gaps as a corrective retrieval phase and
  does not add a second model call.
- `python scripts/check.py` passes.

## Verification

The canonical gate was run from the repository `.venv` with an isolated
temporary directory (the repository's historical pytest temp directory has
restricted ACLs): structure check passed, QA dataset check passed, and all
107 tests passed.


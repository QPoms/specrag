# 038 — Partial synthesis and bounded Research Run deadlines

Status: implemented

## Goal

Publish a cited, explicitly qualified answer when a completed research attempt
has useful validated evidence but cannot establish every planned task, and make
the default Research Run time budget a real 20-minute lifecycle with protected
finalization and persistence time.

## Decisions

- Keep one Research Run lifecycle and one `EvidenceSynthesisService`.
- Add `complete` and `partial` modes to that service; do not add a second
  synthesis implementation.
- A partial synthesis may cite only source-valid, target-supporting evidence
  selected by the task sufficiency assessment. Evidence for conflicting or
  wholly insufficient tasks is not eligible for affirmative conclusions.
- Preserve the original budget stop reason after partial synthesis so the
  published Result remains diagnostically honest.
- The default Run window is 1,200 seconds. The research/search phase stops
  accepting another iteration after 1,020 seconds, final model work stops at
  1,170 seconds, and the last 30 seconds are reserved for durable publication.
- The MCP synchronous wait remains 10 seconds and never creates a second Run.
- Existing persisted Run payloads remain readable; new Runs persist all three
  deadline timestamps.

## Implementation slices

1. Extend synthesis contracts, prompt input, evidence eligibility, and graph
   routing for evidence-qualified partial synthesis.
2. Persist research, finalization, and hard deadlines on Research Run creation,
   resume, and retry; update API, CLI, Worker, and frontend defaults.
3. Enforce the research cutoff at graph budget boundaries and use the
   finalization cutoff for synthesis/provider work while retaining the hard
   deadline as the Harness safety boundary.
4. Add unit and integration regressions for citation eligibility, unresolved
   task reporting, stop-reason preservation, deadline calculation, old payload
   compatibility, and unchanged MCP waiting behavior.
5. Update supported-workflow documentation and run `python scripts/check.py`.

## Non-goals

- Changing corpus indexing or embedding ownership.
- Changing the 10-second MCP synchronous wait.
- Publishing unsupported speculation merely because some evidence exists.
- Replacing the current Workflow, Harness, Worker, or Result persistence path.

# 039 — Parallel task barrier and completion-first research

Status: implemented and verified offline

## Goal

Make complex Research Runs complete reliably before optimizing latency. The
Research Workflow must execute independent task attempts concurrently, wait
until every planned task has reached a terminal lifecycle state, and only then
publish a complete, partial, conflicting, or insufficient result. Research
submission must not synchronously initialize the embedding model or vector
retriever.

This change extends the existing Parallel Research Workflow, RunHarness,
Research Result materializer, retriever pool, and Worker. It does not create a
second research implementation.

## Stage 1 — Task lifecycle and join barrier

- Keep task execution lifecycle separate from evidence sufficiency.
- Treat `completed`, `blocked`, and `failed` as terminal task lifecycle states.
- Add one deterministic terminalization step for exhausted logical budgets:
  running attempts become completed with their latest assessment; tasks that
  could not run because of dependencies become blocked; missing assessments
  become explicit insufficient assessments.
- Permit synthesis only after every planned task is terminal.

## Stage 2 — Dependency-aware concurrent TaskExecution

- Record a distinct TaskExecution for every initial or corrective attempt.
- Dispatch all dependency-ready tasks up to the configured concurrency limit
  through the existing LangGraph `Send` fan-out/fan-in path.
- A failed execution terminalizes its task without failing unrelated sibling
  executions. Dependent tasks are blocked at terminalization.
- Replanning may create later attempts, but it must not mutate earlier
  execution records.

## Stage 3 — Robust synthesis

- The synthesis model returns only prose plus cited evidence identifiers.
- Mode, covered/unresolved/conflicting task IDs, and other lifecycle metadata
  are derived by deterministic application code.
- Structured-output validation failures receive bounded corrective retries.
- A normally completed workflow always materializes a durable Research Result,
  including partial, conflict, and insufficient outcomes.

## Stage 4 — Completion-first runtime policy

- New Research Runs default to no whole-Run business wall-clock deadline.
- Logical iteration/tool/replan budgets, per-provider request timeouts,
  cancellation, Worker leases, and crash recovery remain bounded.
- Explicit wall-clock windows remain representable for compatibility and
  operational overrides.

## Stage 5 — Cold-path reduction and prewarming

- Corpus snapshot binding remains part of immutable Run submission.
- Retriever construction, vector-index opening, and embedding model loading are
  removed from the submission transaction.
- The MCP research process prewarms the Release 17 retriever once at startup;
  Worker execution reuses the same immutable-scope retriever pool.
- API and MCP process ownership remains unchanged, preventing both processes
  from rebuilding the same retrieval runtime.

## Verification boundary

This implementation is verified with deterministic unit/integration tests and
the repository quality gate. It deliberately does not start API/MCP processes,
call a live model, or run a full live backend research case. Remaining runtime
and product issues will be audited separately after these five stages land.

Verification completed on 2026-10-24: `python scripts/check.py` passed the
structure check, the 50-question QA dataset check, and all 312 tests. The
frontend TypeScript/Vite production build also passed. No live backend or model
research validation was run.

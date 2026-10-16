# Durable worker service and restart recovery

Status: implemented locally

## Goal

Replace the process-local Run queue with one durable queue contract backed by
the existing SQLite database. Preserve the embedded local-development mode,
while allowing the API and Worker to run as separate processes and recover
queued or interrupted Runs after a process restart.

## Contracts

- API submission, Resume, and Retry persist a durable job before returning.
- Workers atomically claim one job with an expiring lease. A second Worker
  cannot execute the same live lease, and an expired lease becomes recoverable.
- A heartbeat renews the lease while the existing `RunHarness` executes.
- On startup, the Worker repairs missing queue rows for persisted `queued` or
  `running` Runs created by older application versions.
- A recovered `running` Run resumes its LangGraph checkpoint. A queued Resume
  job also uses the checkpoint; a fresh queued Run executes normally.
- The default API mode embeds the durable Worker for local use. Production can
  run `specrag-api --worker-mode external` with a separate `specrag-worker`
  process against the same database.
- No second research workflow, event store, authorization path, or frontend
  protocol is introduced.

## Completion gate

- Queue claim, lease exclusion, lease expiry, completion, and delayed release
  are covered by SQLite adapter tests.
- Worker restart tests recover both a queued Run and an interrupted running
  Run without duplicate terminal events.
- Existing asynchronous API, SSE, cancellation, Resume, and Retry tests pass.
- CLI entry points and deployment instructions document embedded and external
  modes.
- `python scripts/check.py` and the frontend production build pass.

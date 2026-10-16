# Asynchronous Run API and event streaming

Status: implemented

## Goal

Complete the next platform slice after the authenticated API boundary:
submitting a research Run must return immediately, a bounded background worker
must execute the existing `RunHarness`, and authorized clients must be able to
consume durable Run events through Server-Sent Events (SSE).

The existing CLI remains synchronous and continues to call `RunHarness.start`.
This slice does not introduce a second research workflow, a second event store,
or a frontend. The first worker is an in-process durable-state worker: Run and
event state is persisted in SQLite, but queued work is not automatically
recovered by a newly started process until a deployment-level scheduler is
added.

## Contracts

- `POST /api/v1/workspaces/{workspace}/runs` authenticates and authorizes the
  caller, snapshots corpus/SpecFact/Memory context through the existing
  Harness, persists a `queued` Run and `run_started` event, enqueues execution,
  and returns HTTP 202 with the opaque Run ID.
- The API never accepts arbitrary Memory revision IDs or trusted identity fields
  from the client. Configuration and authorized context remain server-owned.
- The worker calls the existing Harness with the initiating trusted context;
  cancellation and elapsed deadlines continue to be checked by the Harness.
- `GET /api/v1/workspaces/{workspace}/runs/{run_id}/events` returns only events
  for an authorized Run. Each SSE record contains the durable event sequence as
  its SSE ID and the existing `RunEvent` JSON as data. A terminal Run closes
  the stream after its final event.
- Cross-workspace, cross-owner, missing, expired, or revoked access is denied
  before the stream is created. No event payload is emitted to an unauthorized
  caller.

## Completion gate

- API Run submission returns before the worker completes the Run.
- A worker failure leaves a terminal failed Run and an auditable stop event.
- Cancellation can reach a queued/running Run through the existing Harness.
- SSE emits ordered durable events and closes after terminal completion.
- Two-workspace tests reject Run submission, inspection, cancellation, and SSE
  access across the tenant boundary.
- CLI behavior remains unchanged and `python scripts/check.py` passes.

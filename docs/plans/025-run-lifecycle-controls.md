# 025 Run lifecycle controls

## Goal

Complete the authenticated Run lifecycle across Harness, Worker, API, and the
browser client: cancellation, safe failure diagnostics, checkpoint resume, and
fresh retry.

## Scope

- keep cancellation cooperative and idempotent for queued/running Runs;
- queue checkpoint resumes in the existing in-process Worker;
- reset terminal timing/cancellation fields only after a checkpoint is proven
  available;
- retry a terminal Run as a new Run with copied research parameters and an
  explicit `retry_of_run_id` link;
- expose a redacted failure summary and operational metrics;
- add contextual Cancel, Resume, and Retry controls to Run results.

## Out of scope

- a durable external broker or automatic queue recovery after process restart;
- raw exception traces, provider payloads, or secrets in API responses;
- automatic retry policies and scheduled retries.

## Acceptance criteria

- repeated cancellation/resume requests do not create duplicate work;
- only terminal Runs with a durable checkpoint can be resumed;
- retry always creates a new Run and preserves the source Run;
- unauthorized Run operations remain indistinguishable from unknown Runs;
- the frontend explains why an action is or is not available;
- frontend build and the canonical quality gate pass.

## Status

Implemented locally.

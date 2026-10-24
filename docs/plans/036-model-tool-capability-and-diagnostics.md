# 036 - Model Tool Capability and Diagnostics

Status: complete

## Goal

Make Conversation Agent model compatibility observable before a user submits a
message, retain actionable but secret-safe provider diagnostics, and validate
the exact Responses API native-tool contract used by SpecRAG.

## Constraints

- Provider SDK calls and compatibility logic remain in `infrastructure`.
- Conversation orchestration consumes provider-neutral calls and diagnostics.
- Raw provider bodies, prompts, credentials, endpoints, and request headers are
  never persisted or returned to the browser.
- Capability probing must use the same configured model and native-tool path as
  Conversation Agent execution.
- A probe must not create a Conversation, Research Run, or domain result.
- The configured API mode remains explicit. A rejected Responses request is
  reported rather than silently switching to a different provider contract.

## Work

- [x] Add a safe provider diagnostic contract with HTTP status, stable failure
  kind, capability category, retryability, and an actionable public summary.
- [x] Preserve that diagnostic on failed Conversation Turns and show it in the
  Conversation UI.
- [x] Replace the default-oriented Conversation research schema with a valid
  strict function schema. Keep the API mode explicit rather than silently
  switching transports after a rejected request.
- [x] Probe native tool capability while composing the Conversation runtime,
  expose the redacted status through health/API state, and log a startup result.
- [x] Show an early frontend compatibility warning before message submission.
- [x] Add focused provider, runtime, API, persistence, and frontend tests.
- [x] Allow ordinary platform introductions that mention 3GPP or Releases while
  retaining the direct-response guard for actual normative assertions.
- [x] Update README and current status, then run the frontend build and
  `python scripts/check.py`.

## Acceptance

1. A Responses-compatible provider is probed and reported ready.
2. A gateway that rejects Responses tools is reported unavailable at startup
   with its safe HTTP/capability diagnostic rather than failing the first Turn.
3. A gateway that cannot satisfy the configured endpoint and tool contract is
   reported unavailable with a safe actionable reason.
4. The startup projection exposes the safe reason, HTTP status, API mode, and
   retryability; failed Turns persist its actionable public explanation without
   provider payloads or secrets.
5. The browser presents readiness and Turn failure details in user-readable
   form instead of only `invalid_request`.
6. Existing Research Workflow model calls and evidence behavior are unchanged.

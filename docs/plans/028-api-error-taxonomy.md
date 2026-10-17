# 028 API error taxonomy and Run failure projection

## Status

Implemented locally; automatic whole-Run retry remains intentionally out of scope.

## Goal

Make outbound model API failures observable as safe, machine-readable Run
metadata without moving retry ownership into the Worker.

## Decisions

- The LLM adapter owns bounded request-level retry and exponential backoff.
- The adapter classifies provider failures as retryable transient, rate limited,
  timeout, provider unavailable, authentication/configuration, invalid request,
  structured output, or internal/unknown.
- Harness persists only the classification, retryable flag, attempt count, and
  safe provider status code. It never persists provider bodies, headers,
  prompts, or credentials.
- A failed Run remains terminal. The UI may offer a manual retry when the
  failure is marked retryable; the Worker does not automatically replay the
  whole Run.
- Structured output/refusal failures remain non-retryable until a separate
  prompt-repair policy is designed.

## Verification

- Cover transient, rate-limit, timeout, authentication, invalid-request, and
  structured-output classification.
- Cover exhausted request retries and safe Run/API failure projection.
- Run `python scripts/check.py`.

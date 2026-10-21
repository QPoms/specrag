# 034 Context Harness

## Goal

Add one public, provider-independent Context Harness that protects every API
LLM request with a model-aware input budget and supports durable tool-result
truncation, message-history compaction, checkpoint activation, and context
reconstruction.

The first implementation deliberately does not add Native Compaction, async or
post-turn compaction, periodic message deletion, Memory integration, or changes
to the Researcher evidence state machine.

## Design

- Put dependency-free records in `models` and the policy/service code in a new
  `context` package. Provider and SQLite adapters remain in `infrastructure`.
- Estimate each regenerated request in full. Reserve the larger of expected
  output, the configured safety ratio, and the large-window minimum reserve.
  Resolve a profile independently for every fallback model.
- Persist the complete tool payload and expose a deterministic, JSON-safe page
  capped by a configurable token budget. Evidence remains owned by the existing
  evidence ledger and is never reconstructed from this display page.
- Compact only canonical message history. Keep recent whole turns, validate the
  structured summary and literal references, require a measurable token saving,
  persist a checkpoint, then atomically move the Context's active pointer.
- Preserve every original message. Reconstruction reads the active summary plus
  messages beginning at the checkpoint's recent-history boundary.
- Classify only explicit provider context-window errors as overflow. The caller
  may perform one emergency compaction and retry of the same LLM step; completed
  tool calls are outside that retry boundary.

## Integration

- Enforce request budgets in `OpenAIStructuredClient` before every provider
  attempt and record the estimate/ceiling in its receipt.
- Persist Researcher tool results through an injected Context Harness without
  changing selection, validation, or evidence lifecycle behavior.
- Construct one SQLite-backed Context Harness in workspace composition.
- Keep the summary generator as an injected protocol with an API-backed adapter
  so tests and future conversation agents use the same production service.

## Verification

- Unit-test budget math, per-model fallback checks, deterministic truncation and
  paging, summary validation, checkpoint activation/reconstruction, insufficient
  savings, and one-shot overflow recovery.
- Test SQLite round trips and migration behavior.
- Run `python scripts/check.py`.

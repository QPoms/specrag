# 043 - Structured evidence requirements and clause navigation

Status: complete

## Outcome

Implemented on 2026-10-27. The active Researcher now carries validated,
checkpointed Evidence Requirements; retrieval snapshots persist a deterministic
Clause Index; clause, parent/child, and source-confirmed cross-reference reads
are explicit and paged; inactive evidence is bounded in controller context;
and corrective attempts no longer spend dependency scheduling waves. The
existing Validator, TaskExecution, fan-out/fan-in, immutable corpus scope,
Context Harness, and Synthesizer paths remain authoritative.

## Goal

Extend the active SpecRAG research path with durable, task-local Evidence
Requirements and deterministic 3GPP clause navigation. Preserve the existing
Router, Planner, Source Validator, Sufficiency Judge, dependency-aware
LangGraph workflow, Run Worker, frozen corpus scope, and publication-safe
Synthesizer.

This is an incremental extension of the current Researcher ledger and
retrieval snapshot. It does not introduce a Query Rewrite Agent, Evidence
Critic Agent, Summary Agent, parallel retrieval implementation, or a second
workflow.

## Invariants

- A Researcher requirement is an internal coverage hypothesis. Only the
  Sufficiency Judge decides task completeness.
- Requirement identifiers are stable within one ResearchTask. A later review
  may refine a description or add requirements but may not silently delete an
  existing identifier.
- Requirement evidence identifiers must resolve to candidates owned by the
  same task. `covered` and `partial` requirements retain direct evidence in the
  active set; `missing` requirements do not claim supporting evidence.
- Clause structure is derived deterministically from frozen Markdown sources.
  Models never create clause identities, hierarchy, locators, or references.
- Clause and cross-reference navigation never leaves the Run's immutable
  `CorpusScope` and never falls back to a newer active version.
- Existing chunk IDs remain the BM25, dense, hybrid, and evidence units.
- Initial and corrective tool allowances remain independent. Researcher
  rounds, TaskExecution attempts, scheduling waves, and Gap replans remain
  distinct bounded concepts.

## Stage 1 - Evidence Requirement contract and state

1. Add dependency-free `EvidenceRequirement` models with task ownership,
   stable ID, description, `missing / partial / covered`, and evidence IDs.
2. Extend the Researcher native control review to return the complete current
   requirement set in the same model decision as evidence selection, gaps, and
   retrieval calls.
3. Validate requirement identity, non-deletion, task ownership, candidate
   references, active-evidence retention, and coverage coherence in trusted
   code.
4. Persist the latest requirement set through Researcher results, parallel
   fan-in, `ResearchState`, LangGraph checkpoints, and Judge-gap restoration.
5. Keep the full control turns as the audit history; do not create a second
   requirement store.

## Stage 2 - Deterministic Clause Index

1. Extract one reusable Markdown section parser from the existing chunker.
2. Build immutable `ClauseRecord` values containing Release, specification,
   exact version, source path/hash, clause ID/title, direct and descendant
   source bounds, parent/children, direct and descendant chunk IDs, and parsed
   references.
3. Persist `clauses.jsonl` beside `chunks.jsonl`. Existing snapshots remain
   readable; build-enabled maintenance backfills missing clause artifacts
   without recomputing already cached vectors.
4. Replace heading-substring clause lookup with exact indexed lookup and
   deterministic page/cursor semantics. Make descendant inclusion explicit.
5. Keep source locators and returned evidence chunks unchanged so Source
   Validator remains authoritative.

## Stage 3 - Parent/child and cross-reference tools

1. Retain `expand_source_context` for bounded local source-order context, but
   resolve section ownership through the Clause Index.
2. Add `cross_reference_lookup` to the existing Tool Registry and strict native
   schemas.
3. Require a previously observed seed evidence ID and a reference that the
   deterministic index can confirm from that source clause.
4. Resolve same-document references, explicit TS references, and reference-list
   indirection only when the target clause and specification are unambiguous.
5. Return an explicit rejected/no-hit result for ambiguous, absent, or
   out-of-scope targets. Never guess a neighboring clause or version and never
   recurse automatically.

## Stage 4 - Researcher context and budget alignment

1. Update initial and corrective prompts to use structured requirements while
   preserving one controller call per round.
2. Supply complete active evidence and new pending candidates. Supply inactive
   candidates as bounded metadata by default while full tool results remain
   available through the existing Context Harness store.
3. Configure the production Researcher for four retrieval rounds and retain
   the existing 12-call initial and six-call corrective allowances.
4. Prevent the outer scheduling-wave guard from exhausting a valid dependency
   chain merely because a predecessor used a corrective attempt. Preserve
   explicit finite bounds rather than removing the guard.

## Compatibility and migration

- Advance the runtime state schema and add explicit defaults when resuming
  older checkpoints that do not contain Evidence Requirements.
- Keep historical retrieval manifests readable. Clause artifacts are additive
  and do not change chunk or vector identities.
- Keep existing public Run, Result, API, MCP, and CLI wire contracts unless a
  new field is required for internal audit. No unreviewed Requirement state is
  promoted to final Evidence or SpecFacts.

## Verification

- Requirements survive multiple Researcher rounds and corrective checkpoint
  recovery without identifier loss.
- Invalid coverage/evidence associations and silent requirement deletion are
  rejected before tool execution.
- Clause lookup returns every page of a long clause in stable source order and
  distinguishes direct content from descendants.
- Parent and child relationships are deterministic for numeric, letter-suffix,
  and annex clause identifiers found in the retained corpus.
- Cross-reference lookup succeeds only from an observed source reference and
  cannot read a specification version outside the Run scope.
- Existing search, rerank, source expansion, source validation, Judge review,
  dependency handoff, terminal barrier, and synthesis tests continue to pass.
- A dependency chain plus one corrective predecessor attempt can still reach
  every eligible downstream task within the documented outer bound.
- `python scripts/check.py` passes and `docs/current_status.md` describes the
  active implementation.

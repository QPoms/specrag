# 015 Specification version sync and immutable corpus history

## Objective

Add an explicit, testable 3GPP specification-version synchronization boundary
without introducing automatic scheduling or destructive cleanup. A newly
discovered version is stored as a separate source document and may be built
incrementally. Activation is atomic from the registry's point of view; an
incomplete or failed build never displaces the current active version.

Historical source documents and their indexes are retained by default. Any
retirement/garbage-collection operation is explicit and out of scope for the
first implementation. Research runs bind an immutable corpus scope at start;
resume uses that binding rather than re-resolving the moving active pointer.

## Scope

- Add dependency-free version/build/status contracts under `models`.
- Add retrieval-side discovery, local-document registration, and
  document-level incremental build planning. Existing corpus chunking and
  index fingerprints remain the implementation used by a build.
- Add an infrastructure SQLite registry for discovered versions, active
  pointers, build records, and run corpus bindings.
- Extend durable `ResearchRun` data so its exact corpus scope is persisted at
  creation and reused on resume.
- Add one explicit `specrag corpus` command family for inspection and a
  dry-run/apply synchronization workflow. No scheduler or network provider is
  required; providers are injected and tests use local fakes.
- Add focused tests for idempotent discovery, old-version retention,
  build-before-activate, failed-build fallback, activation, and run pinning.
- Reuse an append-only vector cache keyed by embedding/chunk configuration and
  filter dense queries by the exact document hashes in their corpus scope, so a
  new active document does not re-embed unchanged documents.

## Non-goals

- No chunk-level diffing, embedding reuse, distributed workers, Kafka,
  background scheduler, or automatic network polling.
- No automatic deletion of source files, chunks, vectors, or retired versions.
- No change to the existing retrieval ranking or chunking algorithm.

## Design constraints

- `retrieval` owns corpus/version mechanics and remains framework-independent.
- SQLite persistence stays in `infrastructure` and does not import retrieval.
- `orchestration` owns only run binding/resume coordination; it does not own
  ingestion or retrieval algorithms.
- The existing `CorpusScope` remains the retrieval contract. A run stores the
  exact selected specification versions needed to reconstruct that scope.
- A build must validate source existence/hash, non-empty chunks, unique chunk
  IDs, and complete lexical/vector artifacts before the active pointer changes.

## Verification

Run `python scripts/check.py`, including structure checks and the complete test
suite. Update `README.md` and `docs/current_status.md` to document the
explicit, non-destructive sync path and historical-version retention.

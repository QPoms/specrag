# 037 - Release 17 index readiness and asynchronous handoff

Status: complete

## Goal

Make Release 17 the only supported research scope for the current product
phase, keep corpus construction out of Research Run execution, reuse one
read-only retriever per process and corpus snapshot, and finish a user Turn as
soon as an asynchronous Research Run has been durably accepted.

## Current environment audit

- The retained Release 17 snapshot contains 34 documents and 27,858 chunks.
- Every Release 17 chunk ID is present in the shared vector cache; zero Release
  17 vectors are missing.
- Every source document still matches the hash frozen in the snapshot.
- The `corpus_versions` and `active_corpus_versions` registry tables are empty.
  Physical artifacts therefore exist, but no Release 17 versions are registered
  as ready/active for runtime admission.
- Recent mixed/latest and Release 18 Runs created different snapshots and
  attempted to extend the shared vector cache while executing a user Run.

Completion note: the existing Chroma HNSW segment proved unreadable by the
current runtime. Its Release 17 vector payloads were validated and imported
into a new current-format index without recomputing embeddings. The registry
now contains 47 Release 17 historical versions, all 34 current versions are
active, the new index contains exactly 27,858 Release 17 vectors, and a
read-only hybrid query completes without errors. The old segment remains in an
ignored recoverable `chroma-corrupt-20261023` directory pending manual cleanup.

## Invariants

- A Conversation research request without an explicit scope is normalized to
  `highest_version_within_release`, Release 17.
- An explicitly requested non-Release-17 scope is rejected clearly; the server
  never silently answers a Release 18 question from Release 17 evidence.
- Corpus synchronization/indexing is the only path allowed to add missing
  embeddings. Research execution opens an already-complete index read-only.
- A Research Run is not admitted unless the exact Release 17 corpus snapshot is
  registered active and its physical chunks/vectors are complete.
- Retriever construction is cached by immutable corpus scope/fingerprint. A
  Run does not rebuild BM25 or reload the embedding adapter when an equivalent
  retriever is already resident in the same process.
- When `start_research` returns an accepted queued/running Run, the originating
  Conversation Turn publishes a status acknowledgement and stops. It never
  polls the Run in the same Turn. The existing lifecycle worker owns Result
  persistence and conditional automatic presentation.
- In the normal API + MCP deployment, the API runs the Conversation Worker and
  the MCP process runs Research and lifecycle workers. They do not both execute
  Research Runs.

## Work

- [x] Add deterministic pending-Run completion to the Conversation Agent and
  regression coverage proving no same-Turn status polling occurs.
- [x] Normalize default scope to Release 17 and reject explicit unsupported
  releases at the research service boundary.
- [x] Add read-only index opening plus a thread-safe retriever cache and inject
  it into Research Application construction.
- [x] Require active/ready corpus registry state before Research Run admission;
  retain explicit `corpus sync --release 17 --apply` as the build/activation
  path.
- [x] Split API worker startup so the default Conversation deployment does not
  also execute Research Runs when MCP owns them.
- [x] Add focused tests for asynchronous acknowledgement, scope enforcement,
  read-only missing-index failure, retriever reuse, and worker ownership.
- [x] Update README and current status, run `python scripts/check.py`, and
  document the one-time Release 17 registry activation procedure.

## Acceptance

1. A complex specification question creates one Release 17 Run, receives an
   immediate queued/running assistant acknowledgement, and consumes no status
   polling calls in that Turn.
2. Completion later creates one Result and at most one automatic presentation
   Turn through the existing lifecycle path.
3. No Research Run calls an embedding write path. A missing vector/index causes
   an explicit `index_not_ready` failure rather than an inline build.
4. Two Runs over the same Release 17 snapshot reuse the same process-local
   retriever.
5. The normal API process owns Conversation Turns only; the MCP process owns
   Research Run and lifecycle execution.
6. Release 17 status reports all 34 active documents and a complete 27,858
   chunk/vector snapshot before the service is declared research-ready.

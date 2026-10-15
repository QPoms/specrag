# Core and Topic Memory

Status: implemented locally; canonical gate pending unrelated researcher module split

## Goal

Evolve the existing private User Memory implementation into one auditable
multi-user Memory system with a small always-available Core, workspace Topic
Memory, bounded BM25 recall, reviewable extraction, and low-frequency
maintenance. Preserve the current authorization, immutable revision, operation
key, and Run snapshot guarantees.

## Domain boundary

- User Core is private to one user in one workspace. It stores typed defaults,
  long-lived preferences, research habits, followed topics, and saved leads.
- Project Core is scoped to the existing Workspace. It stores team decisions,
  evaluation conventions, and durable research practices.
- Topic Memory is Workspace-scoped. A Topic has a description and may be
  pinned. Pinned Topics join Core during bounded context assembly; other Topics
  are recalled with local BM25.
- Memory is untrusted context. It cannot satisfy source validation or the
  sufficiency judge, create Evidence, or replace a SpecFact.
- Claims about what a specification says remain SpecFact candidates. Runtime
  progress remains Run/checkpoint state.
- SQLite remains the authoritative store. Markdown is only a possible rendered
  import/export view; there is no parallel filesystem store.
- The product term Project maps to the existing `workspace_id`; this milestone
  does not add another Project identity layer.

## Read path

1. Resolve typed user configuration using command argument, User Core, then
   `config.yaml`.
2. Load active User Core, Project Core, and active entries under pinned Topics.
3. Rank non-pinned active Topics against the current Turn with in-memory BM25
   and select at most three.
4. Apply deterministic entry and character budgets, mark every item as
   untrusted context, and record the exact Memory revision IDs on the Run.
5. Resume loads only those recorded revisions after Run authorization.

## Write and review path

- Explicit User Core writes are immediately active after owner authorization.
- Reviewer/admin users manage Project Core, Topics, and pinning. Members may
  read them and may submit proposals.
- An Extractor produces pending operations: `add`, `supersede`, `retract`, or
  `create_topic`. It never writes active Memory directly.
- User-scoped proposals require the owner. Workspace-scoped proposals require a
  reviewer/admin. Rejection requires a reason.
- Every accepted operation is idempotent, attributed, and retains source Run or
  Turn references.

## Low-frequency maintenance

- Consolidation becomes eligible only when both 24 hours have elapsed and at
  least 10 accepted changes were recorded for the same scope.
- Consolidation stays within one scope and Topic/Core collection. It may only
  propose `merge`; the merged entry retains every input ID and source reference,
  and inputs become superseded only after approval.
- Capacity maintenance runs after consolidation. It proposes reversible
  `archive` operations for non-pinned Topic entries. Core and pinned Topics are
  never automatically archived.
- Maintenance never physically deletes a revision and never invents a new
  source reference.

## Implementation slices

1. Replace `UserMemoryItem` with the cohesive Memory domain models and migrate
   the SQLite schema and legacy rows in one change.
2. Extend the single Memory service and stores with User/Project Core, Topics,
   immutable mutations, proposals, and authorization.
3. Add bounded Topic recall and connect it to the existing Run snapshot and
   resume path.
4. Add provider-neutral extraction and maintenance services, plus structured
   model adapters and versioned prompts where model output is required.
5. Extend the existing CLI for Core, Topics, proposal review, recall, and
   maintenance. Do not create another CLI or worker framework.
6. Update README, current status, and package boundaries; run the canonical
   quality gate.

## Acceptance

- Alice's User Core is invisible to Bob, including when both share a Workspace.
- Workspace members read approved Project Core and Topics, while another
  Workspace cannot read, recall, review, pin, or maintain them.
- Core and pinned entries are selected by default within the configured budget;
  a non-pinned reranker Topic is recalled for a reranker query and unrelated
  Topics are not.
- A Run resumes with the same revisions after later supersession, retraction,
  pin changes, or archival.
- Extractor output is pending until the correct owner/reviewer accepts it.
- Consolidation is blocked before either threshold, cannot cross scope or
  Topic, and preserves the full merge lineage.
- Pruning archives only eligible non-pinned Topic entries and is reversible.
- No Memory item can enter Evidence, Source Validator, Sufficiency Judge, or
  SpecFact creation as a trusted assertion.

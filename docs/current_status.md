# Current project status

Updated: 2026-10-09

## Current state

The historical generated runs, benchmark tensors, retrieval indexes, staging
models, analysis outputs, and temporary caches have been removed. The workspace
has been reduced from approximately 23.9 GB to 5.5 GB.

The earlier risk-seed flaw-discovery pipeline and the protocol-context/attack-
chain generation pipeline have also been removed. User questions now enter the
RAG project directly; they are not derived from those upstream research flows.

Removed source areas include the old `run.py`/LangGraph entry, `flow/`,
`nodes/`, `kg/`, `rag/`, `seeds/`, their shared schemas and prompts, and the CR
evaluation package that depended on the attack-chain flow. The old sentence-
level JSONL inputs were removed as well. The Markdown specification corpus was
retained.

The standalone AttackTest IR, the attack-chain/CR evaluator, and the unrelated
B-static source-code analysis and benchmark pipeline have also been removed.
Their CLI registration, Joern adapters, hidden-gold benchmark contracts, and
dataset-specific helper script were removed with them.

The historical attacker-capability validation line has now been removed in
full. This includes all versioned capability agents and validators, attacker
profiles, condition gates, deterministic capability methods, Gold/Shadow/Rel-17
evaluations, and the complete `threatmodel/` experiment tree. The retained
retriever was decoupled from the deleted threat model and now indexes only the
3GPP Markdown specification corpus.

The reusable code has been mechanically migrated out of the ambiguous
`screening/` package:

- retrieval mechanics and planning are under `specrag/retrieval/`;
- the bounded researcher loop is under `specrag/agents/`;
- model and structured-agent adapters are under
  `specrag/infrastructure/`.

The repository, distribution, and Python import package now consistently use
the name `specrag`. The retrieval configuration key is `retrieval`, generated
indexes use `.local/indexes/specifications/`, and the old `capability_*`
package and configuration names are no longer active.

The data tree has two responsibilities only: `data/3gpp/` retains every local
historical specification version, while `data/memory/` is reserved for future
validated long-term knowledge. Redundant `source_corpus/v0_1/specifications`
wrappers and obsolete research catalogs were removed.

Repository-wide guardrails now live in `AGENTS.md`, package ownership is
documented in `docs/architecture/package_boundaries.md`, and
`scripts/check_structure.py` enforces the main rules automatically.
`python scripts/check.py` is the canonical local quality gate and runs both the
structure contract and the test suite.

## Verified reusable path

The immediate migration target is:

```text
research task
-> retrieval planning
-> hybrid/agentic retrieval
-> source-verified evidence
-> sufficiency/gap analysis
-> bounded re-retrieval
-> grounded answer
```

All migrated modules import successfully. The structure check passes and the
current focused test suite passes: 5 tests.

## Known limitations

- Local model weights remain under `models/` but are ignored by Git.
- No supported end-to-end product CLI has been selected yet.
- Evidence sufficiency, validation, synthesis, and long-term memory have not yet
  been rebuilt as clean product modules.
- The corpus retains every historical version, but the current retrieval policy
  exposes only `latest_per_spec` and `highest_version_within_release`. Exact or
  mixed per-specification version routing still needs a clean query contract.
- The README and package metadata now describe the transition toward the new
  Research Agent, but the final package layout and public CLI are not in place.

## Next milestone

Add characterization tests around corpus discovery, lexical fallback,
retrieval planning, and bounded agent termination. Then split the oversized
modules behind tested interfaces and build the clean evidence-validation
boundary for one small direct-question 3GPP scenario.

The live eight-stage progress checklist is maintained in
`docs/plans/003-optimization-roadmap.md`.

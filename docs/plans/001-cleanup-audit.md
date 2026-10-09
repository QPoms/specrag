# Repository cleanup audit

Date: 2026-10-09

This document is a chronological cleanup record. Statements under the initial
snapshot and deferred-work sections describe earlier phases; see
`docs/current_status.md` for the current tree.

## Objective

Reduce the research snapshot to a safe starting point for the Technical Research
Agent project. This pass removes generated artifacts and records which code is a
migration candidate. It does not yet reorganize the Python package or delete old
source implementations.

## Current snapshot

- Total workspace size before cleanup: approximately 23,940 MB.
- `src/specagent/screening`: 106 Python files and approximately 49,553 lines.
- Version-suffixed files in `screening`: 49.
- `runs`: approximately 24,960 files and 1,248 MB.
- `model_staging`: approximately 11,599 MB.
- `models`: approximately 4,812 MB.
- Generated retrieval indexes: approximately 4,650 MB.
- Embedding benchmark tensors: approximately 935 MB.
- The directory is not currently a Git repository.

## Mainline assessment

The repository currently has two capability-validation lines:

1. `capability-method/v1.0` is the deterministic, explicitly registered CLI
   baseline. It contains predicate contracts, evidence requirements, replay, and
   deterministic validation gates.
2. `capability-method/v1.1` is the newer model-led evidence and process audit.
   It has better evidence provenance and gap-retrieval concepts, but its own
   documentation says the online validation was blocked and is not a frozen
   production baseline.

Neither capability method should become the new product architecture wholesale.
The reusable mainline for the Technical Research Agent is the shared mechanism:

```text
question/task
-> retrieval planning
-> hybrid/agentic retrieval
-> source-verified candidate evidence
-> sufficiency and gap analysis
-> bounded re-retrieval
-> evidence-grounded synthesis
-> replayable run record
```

## Keep now

These paths remain untouched during the generated-artifact cleanup:

- `src/specagent/`
- `tests/`
- `scripts/`
- `threatmodel/cases/`
- `threatmodel/docs/`
- `threatmodel/prompts/`
- `threatmodel/registry/`
- `threatmodel/specfacts/`
- `data/source_corpus/`
- `data/specification_catalog/`
- `.env` locally and `.env.example` for documentation

## Primary migration candidates

The following modules contain mechanisms worth extracting into the new package
boundaries. The names below identify source material, not the final API:

- Retrieval: `capability_retrieval.py`, `cross_encoder_reranker.py`,
  `discovery_retrieval_planner.py`, `agentic_retrieval.py`, and
  `answer_hypothesis_retrieval.py`.
- Evidence provenance: `retrieval_evidence_input.py`,
  `evidence_bundle_v11.py`, and `evidence_bundle_adapter_v11.py`.
- Model integration: `llm_client.py` and the structured-output patterns in
  `capability_agent_v11.py`.
- Evaluation and replay: the source hashing, bounded budgets, replay records,
  and deterministic checks in `capability_method_v1.py` and
  `capability_method_v11.py`.
- Domain examples: selected 3GPP questions, facts, and expected evidence from
  `threatmodel/`, to be moved under `examples/3gpp/` later.

## Delete in this pass

The following are generated, reproducible, temporary, or historical output:

- `runs/`
- `analysis/`
- `model_staging/`
- `data/capability_retrieval/`
- `data/embedding_benchmark/`
- `tmp/`, `.tmp/`, `.pytest_cache/`, `.pytest_tmp*/`,
  `pytest-cache-files-*/`, and `.torch_cuda/`
- all `__pycache__/` directories and `*.pyc` files
- root smoke-test scratch file `.tmp_smoke_hybrid.py`

Expected reclaimed space: approximately 18,400 MB.

## Execution result

Cleanup completed on 2026-10-09.

- Workspace size after cleanup: approximately 5,499 MB.
- Space reclaimed: approximately 18,441 MB.
- The explicitly listed generated directories are absent.
- `models/`, `data/source_corpus/`, `src/specagent/`, `tests/`, and
  `threatmodel/` remain present.
- Verification command used the existing Python 3.12 environment and ran the
  retrieval/evidence-focused test subset with bytecode and pytest cache writes
  disabled.
- Verification result: 12 passed.

## Deliberately deferred

- `models/` remains until model download names, checksums, and a replacement
  download/cache policy are recorded. It is now ignored by Git.
- No versioned Python module is deleted in this pass. Old versions are still
  entangled with imports and tests; they should be removed only after the
  reusable retrieval/evidence path has been migrated.
- `threatmodel/test/` remains temporarily because it mixes experiments with
  executable pipeline entry points. It will be split or removed after one new
  end-to-end scenario exists.
- Eight existing test modules reference historical `runs/` paths. They are
  legacy/integration tests and will need deletion or fixture replacement during
  the package migration. Generated run directories will not be retained merely
  to satisfy them.

## Exit criteria for the next phase

1. Define one small 3GPP scenario and its expected evidence under
   `examples/3gpp/`.
2. Create the new package skeleton without placeholder implementations.
3. Migrate retrieval and evidence provenance first.
4. Replace historical-run test dependencies with small files under
   `tests/fixtures/`.
5. Register one supported CLI and remove version numbers from active module
   names.

## Follow-up scope decision

On 2026-10-09 the project scope was narrowed further: questions will be direct
RAG inputs and will no longer originate from risk seeds, flaw discovery,
protocol restoration, or attack-chain generation.

The following source areas were therefore removed after dependency auditing:

- the old `specagent.run` and LangGraph pipeline;
- `flow/`, `nodes/`, `kg/`, the old Chroma `rag/`, and `seeds/`;
- their dedicated state, schemas, prompts, structured-output helper, and dual-
  model LLM adapter;
- the CR dossier evaluator, which imported the removed attack-chain flow;
- the three old risk/sentence JSONL inputs.

The retained RAG implementation is the newer specification retrieval and
evidence path under `src/specagent/screening/`, backed by the Markdown corpus in
`data/source_corpus/`.

## Additional non-RAG removal

The project scope was then narrowed to exclude three more independent research
areas:

- the standalone AttackTest intermediate representation under `testgen/`;
- the already removed attack-chain/CR evaluation flow;
- the B-static source/specification joint-analysis pipeline, including its
  dataset contracts, Joern adapters, adjudicator, benchmark bundle, hidden-gold
  evaluation, CLI, and dataset-specific citation checker.

## Historical capability-validation removal

After the reusable RAG dependency boundary was verified, the complete
attacker-capability validation lineage was removed rather than selecting or
merging one of its incompatible versions. The removal included:

- 74 historical modules from `src/specagent/screening/`;
- 21 capability/threat-model preparation and verification scripts;
- 34 historical validation and evaluation tests;
- the complete `threatmodel/` tree, including cases, prompts, registries,
  spec-facts, Gold/Shadow evaluation assets, and benchmark programs;
- the legacy capability CLI registration and the retriever's mandatory
  `threat_model_path` configuration.

Nine independently importable RAG modules remain under `screening/`. The
retriever now uses only the retained 3GPP Markdown corpus. `llm_client.py` and
`codex_agent.py` remain because they are generic model adapters; attacker models
and all capability decision logic were deleted. Verification completed with all
retained modules importing successfully and the focused suite reporting
`2 passed`.

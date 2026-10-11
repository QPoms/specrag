# Specrag optimization roadmap

Updated: 2026-10-09

## Overall objective

Build Specrag into a clean, testable, internship-ready 3GPP Technical Research
Agent. LangGraph is the orchestration runtime for state, routing, loops,
parallel Researcher scheduling, checkpoints, and recovery. Domain, retrieval,
evidence, memory, and evaluation logic remain framework-independent.

## Progress summary

| Milestone | Status | Current evidence | Next completion gate |
| --- | --- | --- | --- |
| 1. Engineering baseline | Complete | Clean package/data layout, structural guardrails, canonical quality gate, Git baseline | Maintain the gate throughout later milestones |
| 2. Core refactor | Complete | All retained Python modules are within the normal 600-line limit | Preserve these boundaries while adding tool interfaces |
| 3. Version-aware retrieval | Complete | Uniform Release, latest, exact-version, mixed-version, and isolated comparison scopes preserve full source identity and have focused tests | Preserve scope identity through later evidence and memory layers |
| 4. Retrieval tool system | Complete | Framework-independent contracts, registry, six independently verified adapters, and the Researcher integration boundary exist | Keep tools independent as new capabilities are added |
| 5. LangGraph Research Agent | Complete | Single and dynamic parallel Researcher graphs, dependency-ready scheduling, Gap re-plan loop, budgets, checkpoint, and recovery are verified | Preserve runtime boundaries while adding observability |
| 6. Evidence system | Complete | Source validation, support assessment, deduplication, conflicts, task sufficiency, Gap re-planning, and claim-to-evidence synthesis are independently implemented and graph-integrated | Preserve validation contracts through Memory and evaluation |
| 7. Long-term Memory | Complete | Validated-only distillation/store handler, read-only Researcher context, search, version supersession, conflict, manual/time invalidation, and JSONL persistence are implemented and tested | Preserve the write boundary in demos and evaluation |
| 8. Usable baseline and demo | In progress | Runtime contracts and 60 focused tests provide a stable implementation base | Real R17 end-to-end path, strategy tuning, CLI demo, architecture diagram, examples, polished README; formal evaluation is deferred |

## 1. Engineering baseline

- [x] Unify the repository and Python package name as `specrag`.
- [x] Reduce `data/` to the 3GPP corpus and future memory data.
- [x] Preserve all 146 local historical specification versions.
- [x] Add `AGENTS.md`, package boundaries, and automated structure checks.
- [x] Keep generated indexes outside project data under `.local/`.
- [x] Create the first recoverable Git baseline commit.
- [x] Define and run one canonical quality command for structure and tests.

## 2. Core refactor

- [x] Add characterization tests for corpus selection, chunk provenance,
  lexical fallback, reranking, and bounded Researcher termination.
- [x] Split `retrieval/planner.py` by models, planning, execution, selection,
  and answer summarization. Contract models, deterministic question planning,
  specification routing, candidate execution, and ranking are separated;
  relation expansion, selection, and summarization are also separated behind
  the public planner facade.
- [x] Split `retrieval/hypothesis.py` into proposal models, retrieval execution,
  and deterministic parallel-channel merging.
- [x] Split `retrieval/hybrid.py` into corpus, index, lexical, semantic, and
  fusion responsibilities.
- [x] Split `agents/researcher.py` into state, policy, and loop execution.
- [x] Remove all temporary line-count exceptions from the structure checker.

## 3. Version-aware retrieval

- [x] Retain every historical version in a deterministic corpus layout.
- [x] Support the highest available version within a selected Release.
- [x] Define one explicit per-run corpus scope and include it in index identity,
  manifests, and execution artifacts.
- [x] Apply one selected Release uniformly to every specification in a run;
  this remains the default production scope.
- [x] Support cross-version comparison as two isolated Release-scoped runs
  (for example R16 versus R17), without a mixed-version index.
- [x] Include specification, version, source path, document hash, and locator in
  every retrieval hit that can become a citation.
- [x] Support the latest available version of every requested specification.
- [x] Support one exact specification version.
- [x] Support an explicit mixed-version scope for advanced research tasks while
  preserving each document's exact version and Release identity.

## 4. Retrieval tool system

- [x] Define one stable agent-tool request/result contract.
- [x] Add a tool registry without provider-specific logic.
- [x] Expose keyword/BM25 retrieval as one tool.
- [x] Expose semantic retrieval as one tool.
- [x] Expose hybrid fusion and reranking as composable stages.
- [x] Expose clause lookup and adjacent-section expansion.
- [x] Route the bounded Researcher through the Tool Registry instead of direct
  retrieval implementation calls.
- [x] Test every tool independently from the agent loop.

## 5. LangGraph Research Agent

- [x] Add LangGraph only under `specrag/orchestration/`; keep all domain
  decisions and retrieval/evidence/memory implementations framework-independent.
- [x] Define explicit research task, plan, unified `ResearchState`, action, and
  observation models.
- [x] Add bounded iteration, tool-call, and re-plan budgets.
- [x] Record explicit finish and stop reasons.
- [x] Implement Router, Planner, Validator, and Sufficiency Judge as structured
  graph nodes whose business logic remains outside LangGraph.
- [x] Implement Observe → Tool Call → Observe → Re-plan with conditional edges.
- [x] Add checkpoint and recovery for the Single Researcher workflow.
- [x] Preserve Researcher candidate evidence, tool calls, budget use, errors,
  and worker stop reason when updating graph state.
- [x] Add Planner-driven fan-out/fan-in for N Researchers after the single
  workflow is stable; preserve task dependencies and multi-hop boundaries.
- [x] Supply dependency-ready Researchers with predecessor findings selected by
  sufficient assessments and validated as source-backed support.

## 6. Evidence system

- [x] Define candidate and validated evidence models.
- [x] Verify source path, document hash, clause locator, and exact excerpt.
- [x] Deduplicate overlapping evidence.
- [x] Detect version and semantic conflicts.
- [x] Evaluate evidence sufficiency against the research question.
- [x] Produce explicit gap analysis and bounded follow-up retrieval.
- [x] Route `missing_relation` and insufficient evidence back to Re-planning
  through LangGraph conditional edges.
- [x] Generate answers that map claims to evidence.

## 7. Long-term Memory

- [x] Define the knowledge and provenance schema.
- [x] Permit writes only from validated evidence and grounded synthesis.
- [x] Preserve specification version, source hash, and validation time.
- [x] Implement search and reuse in later research tasks.
- [x] Handle updates, conflicts, and supersession explicitly.
- [x] Handle time-based expiry explicitly.
- [x] Prevent raw model output from entering memory.

## 8. Usable baseline and demo

- [x] Select a small set of representative R17 development questions; the
  curated set contains 34 simple, 16 complex, and 50 development
  cases. It guides implementation work and is not presented as a formal
  benchmark.
- [ ] Make one real question run end to end through planning, retrieval,
  validation, Gap/re-plan, and grounded synthesis.
- [ ] Inspect and tune Planner decomposition, Researcher tool choice, retrieval
  fusion/reranking, sufficiency decisions, and synthesis one stage at a time.
- [ ] Record enough graph transitions, Researcher allocation, tool calls, gaps,
  re-plan count, checkpoint identity, and recovery events to debug a run.
- [ ] Provide one supported `specrag` CLI demo.
- [ ] Add an architecture diagram and reproducible demo instructions.
- [ ] Polish the README around problem, design, results, and limitations.
- [ ] Defer formal metric and benchmark design to the project owner after the
  end-to-end strategies are stable.

## Global completion constraints

- Do not reintroduce attack-chain, attacker-capability, B-static, CR, or hidden-
  Gold research pipelines.
- Do not introduce versioned Python modules or parallel implementations.
- Do not treat model-generated content as evidence.
- Do not write unvalidated content to long-term memory.
- Keep every retained 3GPP historical version available for retrieval.
- A normal retrieval run uses one uniform Release snapshot. Exact and mixed
  scopes require an explicit request and retain per-document identity.
- Only `orchestration/` may import LangGraph. Core evaluation results remain
  usable without LangSmith or any tracing provider.
- Completion requires passing structure checks, automated tests, a runnable demo,
  and traceable claim-to-evidence output.

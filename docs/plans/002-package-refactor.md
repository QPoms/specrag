# Package refactor plan

Date: 2026-10-09

## Goal

Turn the retained RAG code into a stable base for a future multi-agent research
system without recreating the versioned-module and experiment-directory sprawl
from the research snapshot.

## Phase 1: establish guardrails

- Add permanent repository rules in `AGENTS.md`.
- Document package ownership and allowed dependency direction.
- Add an executable structure check for forbidden legacy paths, versioned
  module names, package-boundary violations, and file-growth ratchets.

## Phase 2: mechanical package migration

- Move retrieval mechanics into `specrag.retrieval`.
- Move the bounded researcher loop into `specrag.agents`.
- Move model/runtime adapters into `specrag.infrastructure`.
- Rename the configuration section from `capability_retrieval` to `retrieval`.
- Remove the ambiguous `screening` package.

## Phase 3: stabilize before splitting

- Add characterization tests for corpus discovery, lexical fallback, retrieval
  planning, and bounded agent stop conditions.
- Define the minimum public interfaces used by the first demo.
- Split oversized modules along those tested interfaces.

## Phase 4: add the missing product path

- Introduce evidence models and source validation.
- Add a small direct-question 3GPP example.
- Add one supported CLI from question to evidence-grounded response.
- Add memory only after validated evidence and synthesis are stable.

## Non-goals

- Reintroducing attack-chain, attacker-capability, B-static, CR, or hidden-Gold
  pipelines.
- Creating all target directories before working implementations exist.
- Keeping compatibility aliases for deleted experimental import paths.

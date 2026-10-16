# 023 Memory and SpecFact management

## Goal

Expose the existing Memory and SpecFact lifecycle through the authenticated
workspace API and give users a focused frontend for inspecting and reviewing
that knowledge.

## Scope

- add workspace-scoped read models for User Core, Project Core, Topic Memory,
  Memory proposals, SpecFacts, and SpecFact proposals;
- expose proposal approve/reject actions through the existing domain services;
- preserve User Memory ownership and reviewer/admin approval rules;
- add a responsive knowledge-management surface to the existing frontend;
- show provenance, proposal state, empty states, and actionable errors.

## Out of scope

- validating Memory quality or retrieval effectiveness;
- free-form editing, deletion, maintenance scheduling, or bulk review;
- changing the proposal extraction pipeline.

## Acceptance criteria

- workspace members only receive their own User Memory and the shared
  knowledge belonging to their current workspace;
- unauthorized review attempts return 403 without leaking cross-workspace data;
- reviewers and admins can approve or reject pending workspace proposals;
- rejection requires a reason and the frontend refreshes after each decision;
- frontend build and the canonical quality gate pass.

## Status

Implemented locally.

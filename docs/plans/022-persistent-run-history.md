# 022 Persistent Run history

## Goal

Make workspace Run history survive browser refreshes and allow authorized users
to reopen historical research from the frontend.

## Scope

- add filtered Run listing to the existing Run stores and RunHarness;
- expose an authenticated workspace Run-list endpoint;
- preserve the existing owner rule: members and reviewers see their own Runs,
  while workspace admins may see every Run in that workspace;
- load history when the frontend enters or switches a Workspace;
- reconnect active persisted Runs to SSE and allow recent/all history views.

## Out of scope

- server-side search, pagination cursors, retention, and deletion;
- Run retry/resume controls;
- Memory and SpecFact management.

## Acceptance criteria

- browser refresh restores authorized Run history from SQLite;
- no Run from another workspace or unauthorized owner appears in the list;
- active restored Runs resume frontend event monitoring;
- frontend build and the canonical quality gate pass.

## Status

Implemented locally.

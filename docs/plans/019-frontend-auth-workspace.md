# 019 Frontend authentication and workspace shell

## Goal

Add the first browser client for the authenticated SpecRAG API: a polished,
responsive login screen and a workspace research shell that uses the existing
HTTP security boundary rather than accessing local state directly.

## Scope

- create a local React + TypeScript + Vite application under `frontend/`;
- implement session-token login, logout, workspace selection, and workspace
  context loading against `/api/v1`;
- provide the first research-workspace visual language and responsive layout;
- keep the frontend transport-only: identity, membership, and authorization
  remain server decisions;
- document local development and verify the Python quality gate plus a frontend
  production build.

## Out of scope

- hosted deployment, public publishing, or a separate frontend server;
- document upload, Memory/SpecFact administration, and full evidence views;
- replacing the API's process-local worker with a broker.

## Acceptance criteria

- unauthenticated users see a login view and receive clear API errors;
- authenticated users can load and switch workspaces without client-supplied
  identity or role claims;
- the UI provides a clear entry point for starting a research run;
- `npm run build` and `python scripts/check.py` pass.

## Status

Implemented locally. The next frontend slice will wire the async Run submit,
SSE event stream, and evidence presentation.

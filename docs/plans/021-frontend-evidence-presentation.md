# 021 Frontend evidence presentation

## Goal

Expose the completed Run's source-backed synthesis through the authenticated
API and render it in the research desk as an auditable answer with provenance-
rich evidence cards.

## Scope

- extend the authorized Run inspection response with the persisted final
  synthesis claims and supported candidate evidence;
- keep only evidence that passed source validation, supports the target, and is
  referenced by the final claims;
- render the answer, claims, source path, exact version, document hash,
  heading, line locator, and excerpt in the browser;
- allow a user to open any recent Run and fetch its current inspection.

## Out of scope

- editing evidence or promoting it directly to SpecFact;
- document upload or a full document viewer;
- Memory and SpecFact management screens.

## Acceptance criteria

- the API never exposes candidate evidence that failed source validation as
  final evidence;
- inaccessible Runs still return the same authorization response as before;
- a completed Run can be selected and its answer/evidence rendered without
  client-side inference;
- frontend build, structure checks, and the full test suite pass.

## Status

Implemented locally. Memory and SpecFact management is the next frontend
slice.

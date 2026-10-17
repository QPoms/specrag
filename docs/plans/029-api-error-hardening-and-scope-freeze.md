# 029 API error hardening and scope freeze

## Status

Implemented locally.

## Goal

Harden the existing HTTP error contract and freeze the current internship
project scope without adding another product capability.

## Scope

- expose a stable machine-readable error code alongside the existing `detail`
  field;
- give request validation failures an explicit `validation_error` code;
- map Run submission conflicts to a stable HTTP error instead of an uncaught
  server exception;
- keep one canonical Run failure-kind set shared by provider classification,
  durable Run metadata, and API projection;
- project model refusal as `model_refusal` rather than generic structured
  output failure;
- render structured validation details clearly in the browser client;
- document the supported feature boundary and explicitly defer scale-out and
  production-only capabilities.

## Non-goals

- no new retrieval, evidence, memory, workspace, or deployment capability;
- no automatic whole-Run replay;
- no external broker or distributed Worker;
- no private-document upload or additional corpus provider.

## Verification

- add focused API, provider, Run projection, and frontend contract tests;
- run `python scripts/check.py`.

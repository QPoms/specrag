# 024 Member management

## Goal

Complete the multi-user administration loop with a protected workspace member
list and a usable frontend for adding, reviewing, changing, and removing
members.

## Scope

- expose an admin-only workspace member list with account status and role;
- retain service-layer authorization for every mutation;
- prevent an administrator from removing or demoting their own membership;
- add a responsive member-management panel with role editing, removal
  confirmation, member creation, and actionable feedback;
- keep non-admin users out of the administration data path and explain the
  role model in the interface.

## Out of scope

- email invitations, password reset, account disabling, and SSO;
- organization-wide user administration;
- audit-log persistence and last-admin transfer workflows.

## Acceptance criteria

- only workspace admins can list or mutate membership;
- member records cannot cross workspace boundaries;
- self-removal and self-demotion are rejected by the backend;
- frontend state refreshes after each successful mutation;
- frontend build and the canonical quality gate pass.

## Status

Implemented locally.

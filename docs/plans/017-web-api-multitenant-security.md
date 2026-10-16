# Web/API multi-tenant security boundary

Status: implemented

## Goal

Expose the existing local identity, workspace, authorization, and Run services
through one HTTP API without creating a second permission system. Every
protected request must authenticate a server-issued session token, resolve the
current membership on the server, construct a trusted `ActorContext`, and call
the existing application services with that context.

This milestone establishes the security boundary needed by a future frontend.
It does not move research execution into a background worker; asynchronous Run
submission and event streaming are the next milestone.

## Security contract

- The API accepts credentials only at login. It never accepts trusted
  `user_id`, `workspace_id`, `role`, or `session_id` fields from a client.
- Protected endpoints require an `Authorization: Bearer <token>` header. Raw
  tokens are returned only at login, are never persisted by the API, and the
  existing identity store continues to persist only their SHA-256 digests.
- A workspace path parameter is resolved together with the authenticated
  session on every request. Current membership and role are therefore checked
  again after role changes or membership removal.
- Existing `AccessPolicy`, `LocalIdentityService`, and `RunHarness` decisions
  remain authoritative. The API maps failures to stable HTTP errors and does
  not expose whether an inaccessible Run exists.
- Browser credential cookies and permissive CORS are not enabled. This first
  boundary uses an explicit bearer header, so cross-site requests cannot rely
  on ambient authentication. Production deployment must terminate TLS.
- Authentication responses and protected API responses carry `no-store` and
  basic browser hardening headers.

## First implementation slice

1. Add dependency-free request/response transport models under `models`.
2. Add an HTTP composition module under `interfaces` that reuses
   `WorkspaceServices`.
3. Expose login/logout, current workspace context, workspace creation, member
   add/role/remove, and authorized Run inspect/cancel endpoints.
4. Keep account bootstrap in the local administrative CLI so an unauthenticated
   network caller cannot initialize a deployment.
5. Add API acceptance tests with two users and two workspaces, including
   missing/invalid credentials, role enforcement, membership revocation, and
   cross-workspace Run denial.
6. Document the supported API command and current limitation that research Run
   creation remains on the CLI until the asynchronous execution milestone.

## Completion gate

- No protected endpoint succeeds without a current server-side session and
  membership.
- A client cannot elevate privileges by submitting identity or role fields.
- Cross-workspace Run inspection and cancellation return the same response as
  an unknown Run.
- Membership removal takes effect on the next API request.
- CLI behavior and the canonical `python scripts/check.py` gate still pass.

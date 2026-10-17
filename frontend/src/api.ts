export type Role = "member" | "reviewer" | "admin";

export type Workspace = {
  workspace_id: string;
  name: string;
  created_by: string;
  created_at: string;
};

export type Membership = {
  workspace_id: string;
  user_id: string;
  role: Role;
  created_at: string;
};

export type WorkspaceMembership = {
  workspace: Workspace;
  membership: Membership;
};

export type WorkspaceMember = {
  user: {
    user_id: string;
    username: string;
    status: "active" | "disabled";
    created_at: string;
  };
  membership: Membership;
};

export type WorkspaceContext = {
  user_id: string;
  workspace_id: string;
  role: Role;
  session_expires_at: string;
};

export type Run = {
  run_id: string;
  workspace_id: string;
  owner_user_id: string;
  question: string;
  release: number;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "timed_out";
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  stop_reason: string | null;
  cancel_requested_at: string | null;
  retry_of_run_id: string | null;
};

export type RunInspection = {
  run: Run;
  events: RunEvent[];
  stage_latency_ms: Record<string, number>;
  input_tokens: number;
  output_tokens: number;
  checkpoint_available: boolean;
  answer: string | null;
  evidence_ids: string[];
  evidence: EvidenceItem[];
  failure: RunFailure | null;
};

export type RunFailure = {
  summary: string;
  stop_reason: string;
  failure_kind: string | null;
  retryable: boolean;
  attempt_count: number | null;
  status_code: number | null;
  last_stage: string | null;
  last_event_type: string | null;
  last_event_at: string | null;
};

export type EvidenceItem = {
  evidence_id: string;
  task_id: string;
  chunk_id: string;
  specification: string;
  release: number;
  version: string;
  source_path: string;
  document_sha256: string;
  heading_path: string[];
  line_start: number;
  line_end: number;
  excerpt: string;
};

export type RunEvent = {
  event_id?: string;
  run_id: string;
  workspace_id: string;
  sequence: number;
  event_type: string;
  stage: string | null;
  occurred_at: string;
  payload?: Record<string, unknown>;
};

export type MemoryEntry = {
  memory_id: string;
  scope: "user" | "workspace";
  layer: "core" | "topic";
  topic_id: string | null;
  kind: string;
  key: string;
  content: string;
  origin: string;
  status: string;
  revision: number;
  updated_at: string;
  source_refs: string[];
};

export type MemoryTopic = {
  topic_id: string;
  slug: string;
  description: string;
  pinned: boolean;
  status: string;
  updated_at: string;
};

export type MemoryProposal = {
  proposal_id: string;
  owner_user_id: string | null;
  draft: {
    operation: string;
    scope: "user" | "workspace";
    kind: string | null;
    key: string | null;
    content: string | null;
    topic_slug: string | null;
    topic_description: string | null;
    source_refs: string[];
  };
  proposed_by: string;
  proposed_at: string;
  status: "pending" | "approved" | "rejected";
  reviewer_user_id: string | null;
  review_reason: string | null;
};

export type MemoryDashboard = {
  user_core: MemoryEntry[];
  project_core: MemoryEntry[];
  topics: { topic: MemoryTopic; entries: MemoryEntry[] }[];
  proposals: MemoryProposal[];
};

export type SpecFactSource = {
  evidence_id: string;
  specification: string;
  release: number;
  version: string;
  source_path: string;
  heading_path: string[];
  line_start: number;
  line_end: number;
};

export type SpecFactProposal = {
  proposal_id: string;
  assertion: string;
  subject: string;
  predicate: string;
  object: string;
  conditions: string[];
  sources: SpecFactSource[];
  proposed_by: string;
  proposed_at: string;
  status: "pending" | "approved" | "rejected";
  reviewer_user_id: string | null;
  review_reason: string | null;
};

export type SpecFact = {
  fact_id: string;
  assertion: string;
  subject: string;
  predicate: string;
  object: string;
  conditions: string[];
  sources: SpecFactSource[];
  approved_at: string;
  status: "approved" | "superseded" | "conflicted" | "invalidated";
};

export type SpecFactDashboard = {
  facts: SpecFact[];
  proposals: SpecFactProposal[];
};

type SessionResponse = { access_token: string; expires_at: string };

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(status: number, message: string, code = "http_error") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

type ApiErrorPayload = { detail?: unknown; code?: unknown };

function formatApiError(payload: ApiErrorPayload, fallback: string): string {
  if (typeof payload.detail === "string" && payload.detail.trim()) {
    return payload.detail;
  }
  if (Array.isArray(payload.detail)) {
    const messages = payload.detail.map((item) => {
      if (!item || typeof item !== "object") return String(item);
      const record = item as { loc?: unknown; msg?: unknown };
      const location = Array.isArray(record.loc)
        ? record.loc.filter((part) => typeof part === "string" || typeof part === "number").join(".")
        : "request";
      return typeof record.msg === "string" ? `${location}: ${record.msg}` : location;
    }).filter(Boolean);
    if (messages.length) return messages.join("; ");
  }
  return fallback;
}

export class SpecRagApi {
  private token: string | null;

  constructor(token: string | null = sessionStorage.getItem("specrag.access_token")) {
    this.token = token;
  }

  get hasToken(): boolean {
    return Boolean(this.token);
  }

  setToken(token: string | null): void {
    this.token = token;
    if (token) sessionStorage.setItem("specrag.access_token", token);
    else sessionStorage.removeItem("specrag.access_token");
  }

  async login(username: string, password: string): Promise<SessionResponse> {
    const response = await this.request<SessionResponse>("/api/v1/auth/login", {
      method: "POST",
      body: { username, password },
      authenticated: false,
    });
    this.setToken(response.access_token);
    return response;
  }

  async logout(): Promise<void> {
    if (this.token) {
      await this.request("/api/v1/auth/logout", { method: "POST" }).catch(() => undefined);
    }
    this.setToken(null);
  }

  workspaces(): Promise<WorkspaceMembership[]> {
    return this.request<WorkspaceMembership[]>("/api/v1/workspaces");
  }

  context(workspaceId: string): Promise<WorkspaceContext> {
    return this.request<WorkspaceContext>(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}/me`);
  }

  createWorkspace(name: string): Promise<Workspace> {
    return this.request<Workspace>("/api/v1/workspaces", { method: "POST", body: { name } });
  }

  members(workspaceId: string): Promise<WorkspaceMember[]> {
    return this.request<WorkspaceMember[]>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/members`,
    );
  }

  addMember(
    workspaceId: string,
    username: string,
    password: string,
    role: Role,
  ): Promise<Membership> {
    return this.request<Membership>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/members`,
      { method: "POST", body: { username, password, role } },
    );
  }

  setMemberRole(
    workspaceId: string,
    username: string,
    role: Role,
  ): Promise<Membership> {
    return this.request<Membership>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(username)}`,
      { method: "PATCH", body: { role } },
    );
  }

  removeMember(workspaceId: string, username: string): Promise<void> {
    return this.request(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(username)}`,
      { method: "DELETE" },
    );
  }

  runs(workspaceId: string, limit = 50): Promise<Run[]> {
    return this.request<Run[]>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/runs?limit=${limit}`,
    );
  }

  submitRun(workspaceId: string, question: string, release: number): Promise<RunInspection> {
    return this.request<RunInspection>(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}/runs`, {
      method: "POST",
      body: { question, release, answer_language: "English", detail_level: "standard" },
    });
  }

  inspectRun(workspaceId: string, runId: string): Promise<RunInspection> {
    return this.request<RunInspection>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/runs/${encodeURIComponent(runId)}`,
    );
  }

  cancelRun(workspaceId: string, runId: string): Promise<Run> {
    return this.request<Run>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/runs/${encodeURIComponent(runId)}/cancel`,
      { method: "POST" },
    );
  }

  resumeRun(workspaceId: string, runId: string): Promise<RunInspection> {
    return this.request<RunInspection>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/runs/${encodeURIComponent(runId)}/resume`,
      { method: "POST", body: { elapsed_time_seconds: 900 } },
    );
  }

  retryRun(workspaceId: string, runId: string): Promise<RunInspection> {
    return this.request<RunInspection>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/runs/${encodeURIComponent(runId)}/retry`,
      { method: "POST", body: { elapsed_time_seconds: 900 } },
    );
  }

  memory(workspaceId: string): Promise<MemoryDashboard> {
    return this.request<MemoryDashboard>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/memory`,
    );
  }

  reviewMemoryProposal(
    workspaceId: string,
    proposalId: string,
    approve: boolean,
    reason?: string,
  ): Promise<MemoryProposal> {
    return this.request<MemoryProposal>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/memory/proposals/${encodeURIComponent(proposalId)}/review`,
      { method: "POST", body: { approve, reason } },
    );
  }

  specFacts(workspaceId: string): Promise<SpecFactDashboard> {
    return this.request<SpecFactDashboard>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/specfacts`,
    );
  }

  reviewSpecFactProposal(
    workspaceId: string,
    proposalId: string,
    approve: boolean,
    reason?: string,
  ): Promise<SpecFactProposal> {
    return this.request<SpecFactProposal>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/specfacts/proposals/${encodeURIComponent(proposalId)}/review`,
      { method: "POST", body: { approve, reason } },
    );
  }

  async streamRunEvents(
    workspaceId: string,
    runId: string,
    onEvent: (event: RunEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    if (!this.token) throw new ApiError(401, "authentication required", "authentication_required");
    const path = `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/runs/${encodeURIComponent(runId)}/events?since=0&wait_seconds=300`;
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { Accept: "text/event-stream", Authorization: `Bearer ${this.token}` },
      signal,
    });
    if (!response.ok) {
      if (response.status === 401) this.setToken(null);
      const fallback = `Event stream failed (${response.status})`;
      let message = fallback;
      let code = "http_error";
      try {
        const payload = (await response.json()) as ApiErrorPayload;
        message = formatApiError(payload, fallback);
        if (typeof payload.code === "string") code = payload.code;
      } catch {
        // Keep the status-based message when the API did not return JSON.
      }
      throw new ApiError(response.status, message, code);
    }
    if (!response.body) throw new Error("Event stream has no response body");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const chunk = await reader.read();
        buffer += decoder.decode(chunk.value ?? new Uint8Array(), { stream: !chunk.done });
        const records = buffer.split("\n\n");
        buffer = records.pop() ?? "";
        for (const record of records) {
          const dataLine = record.split("\n").find((line) => line.startsWith("data:"));
          if (!dataLine) continue;
          try {
            onEvent(JSON.parse(dataLine.slice(5).trim()) as RunEvent);
          } catch {
            // Ignore a malformed event rather than breaking the active stream.
          }
        }
        if (chunk.done) break;
      }
    } finally {
      reader.releaseLock();
    }
  }

  private async request<T>(
    path: string,
    options: { method?: string; body?: unknown; authenticated?: boolean } = {},
  ): Promise<T> {
    const headers = new Headers({ "Content-Type": "application/json" });
    if (options.authenticated !== false && this.token) headers.set("Authorization", `Bearer ${this.token}`);
    const response = await fetch(`${API_BASE}${path}`, {
      method: options.method ?? "GET",
      headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
    });
    if (!response.ok) {
      const fallback = `Request failed (${response.status})`;
      let message = fallback;
      let code = "http_error";
      try {
        const payload = (await response.json()) as ApiErrorPayload;
        message = formatApiError(payload, fallback);
        if (typeof payload.code === "string") code = payload.code;
      } catch {
        // Keep the status-based message when the API did not return JSON.
      }
      if (response.status === 401) this.setToken(null);
      throw new ApiError(response.status, message, code);
    }
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }
}

import { useEffect, useState } from "react";
import {
  MemoryDashboard,
  MemoryEntry,
  MemoryProposal,
  Role,
  SpecFactDashboard,
  SpecFactProposal,
  SpecRagApi,
} from "./api";

type View = "memory" | "specfacts";

type Props = {
  api: SpecRagApi;
  workspaceId: string;
  role: Role;
};

export function KnowledgePanel({ api, workspaceId, role }: Props) {
  const [view, setView] = useState<View>("memory");
  const [memory, setMemory] = useState<MemoryDashboard | null>(null);
  const [specFacts, setSpecFacts] = useState<SpecFactDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [nextMemory, nextSpecFacts] = await Promise.all([
        api.memory(workspaceId),
        api.specFacts(workspaceId),
      ]);
      setMemory(nextMemory);
      setSpecFacts(nextSpecFacts);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load workspace knowledge");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    // Loading is tied to the selected workspace; the API client is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceId]);

  const memoryCount = (memory?.user_core.length ?? 0)
    + (memory?.project_core.length ?? 0)
    + (memory?.topics.reduce((total, item) => total + item.entries.length, 0) ?? 0);
  const pendingMemory = memory?.proposals.filter((item) => item.status === "pending").length ?? 0;
  const pendingFacts = specFacts?.proposals.filter((item) => item.status === "pending").length ?? 0;

  return <section className="knowledge-panel" id="knowledge">
    <div className="knowledge-header">
      <div>
        <p className="eyebrow">Reviewed workspace knowledge</p>
        <h2>Memory &amp; specification facts</h2>
        <p>Inspect what the system remembers, then approve durable facts before they influence future research.</p>
      </div>
      <button className="quiet-button" onClick={() => void load()} disabled={loading}>Refresh</button>
    </div>
    <div className="knowledge-tabs" role="tablist" aria-label="Knowledge type">
      <button className={view === "memory" ? "active" : ""} onClick={() => setView("memory")} role="tab" aria-selected={view === "memory"}>
        <span>Memory</span><small>{memoryCount} active · {pendingMemory} pending</small>
      </button>
      <button className={view === "specfacts" ? "active" : ""} onClick={() => setView("specfacts")} role="tab" aria-selected={view === "specfacts"} id="specfacts">
        <span>SpecFacts</span><small>{specFacts?.facts.length ?? 0} reviewed · {pendingFacts} pending</small>
      </button>
    </div>
    {error && <div className="knowledge-error" role="alert">{error}</div>}
    {loading && !memory ? <KnowledgeLoading /> : view === "memory"
      ? <MemoryView data={memory} role={role} onReview={async (proposal, approve, reason) => {
        await api.reviewMemoryProposal(workspaceId, proposal.proposal_id, approve, reason);
        await load();
      }} />
      : <SpecFactView data={specFacts} canReview={role !== "member"} onReview={async (proposal, approve, reason) => {
        await api.reviewSpecFactProposal(workspaceId, proposal.proposal_id, approve, reason);
        await load();
      }} />}
  </section>;
}

function KnowledgeLoading() {
  return <div className="knowledge-loading"><span className="loader" /> Loading reviewed knowledge…</div>;
}

function MemoryView({ data, role, onReview }: {
  data: MemoryDashboard | null;
  role: Role;
  onReview: (proposal: MemoryProposal, approve: boolean, reason?: string) => Promise<void>;
}) {
  if (!data) return <KnowledgeEmpty title="Memory is unavailable" detail="Refresh to try loading it again." />;
  return <div className="knowledge-body">
    <KnowledgeSection title="Pending proposals" count={data.proposals.length} note="Human review boundary">
      {data.proposals.length === 0
        ? <KnowledgeEmpty title="No proposals waiting" detail="Extracted memories will land here before they become durable." />
        : <div className="proposal-list">{data.proposals.map((proposal) => <article className="proposal-card" key={proposal.proposal_id}>
          <div className="card-topline"><span className={`knowledge-status ${proposal.status}`}>{proposal.status}</span><time>{formatDate(proposal.proposed_at)}</time></div>
          <span className="card-kicker">{proposal.draft.scope} memory · {proposal.draft.operation}</span>
          <h4>{proposal.draft.key ?? proposal.draft.topic_slug ?? "Memory change"}</h4>
          <p>{proposal.draft.content ?? proposal.draft.topic_description ?? "This proposal changes an existing Memory record."}</p>
          <div className="card-meta"><span>Proposed by {proposal.proposed_by}</span>{proposal.draft.kind && <span>{proposal.draft.kind.replaceAll("_", " ")}</span>}</div>
          {proposal.review_reason && <p className="review-note">Review note: {proposal.review_reason}</p>}
          {proposal.status === "pending" && (proposal.draft.scope === "user" || role !== "member")
            ? <ReviewActions onReview={(approve, reason) => onReview(proposal, approve, reason)} />
            : proposal.status === "pending" && <p className="permission-note">Reviewer or admin approval required.</p>}
        </article>)}</div>}
    </KnowledgeSection>
    <div className="knowledge-columns">
      <KnowledgeSection title="Your Core Memory" count={data.user_core.length} note="Private to your account">
        <MemoryEntries entries={data.user_core} empty="No personal preferences have been saved." />
      </KnowledgeSection>
      <KnowledgeSection title="Project Core" count={data.project_core.length} note="Shared across this workspace">
        <MemoryEntries entries={data.project_core} empty="No reviewed project decisions yet." />
      </KnowledgeSection>
    </div>
    <KnowledgeSection title="Topic Memory" count={data.topics.length} note="Long-running research threads">
      {data.topics.length === 0 ? <KnowledgeEmpty title="No research topics yet" detail="Topic Memory appears after a topic is created and reviewed." />
        : <div className="topic-grid">{data.topics.map(({ topic, entries }) => <article className="topic-card" key={topic.topic_id}>
          <div className="card-topline"><span className="topic-slug">{topic.slug}</span>{topic.pinned && <span className="pin-label">Pinned</span>}</div>
          <h4>{topic.description}</h4>
          <MemoryEntries entries={entries} empty="This topic has no active Memory." compact />
        </article>)}</div>}
    </KnowledgeSection>
  </div>;
}

function MemoryEntries({ entries, empty, compact = false }: { entries: MemoryEntry[]; empty: string; compact?: boolean }) {
  if (entries.length === 0) return <p className="inline-empty">{empty}</p>;
  return <div className={compact ? "memory-list compact" : "memory-list"}>{entries.map((entry) => <div className="memory-row" key={entry.memory_id}>
    <div><span>{entry.kind.replaceAll("_", " ")}</span><strong>{entry.key}</strong></div>
    <p>{entry.content}</p>
    <small>rev {entry.revision} · {entry.origin} · {formatDate(entry.updated_at)}</small>
  </div>)}</div>;
}

function SpecFactView({ data, canReview, onReview }: {
  data: SpecFactDashboard | null;
  canReview: boolean;
  onReview: (proposal: SpecFactProposal, approve: boolean, reason?: string) => Promise<void>;
}) {
  if (!data) return <KnowledgeEmpty title="SpecFacts are unavailable" detail="Refresh to try loading them again." />;
  return <div className="knowledge-body">
    <KnowledgeSection title="Fact proposals" count={data.proposals.length} note="Evidence-backed, awaiting review">
      {data.proposals.length === 0 ? <KnowledgeEmpty title="No fact proposals" detail="Independently extracted specification facts will appear here for review." />
        : <div className="proposal-list">{data.proposals.map((proposal) => <article className="proposal-card fact-card" key={proposal.proposal_id}>
          <div className="card-topline"><span className={`knowledge-status ${proposal.status}`}>{proposal.status}</span><time>{formatDate(proposal.proposed_at)}</time></div>
          <span className="card-kicker">{proposal.subject} · {proposal.predicate}</span>
          <h4>{proposal.assertion}</h4>
          <SourceLine sources={proposal.sources} />
          {proposal.review_reason && <p className="review-note">Review note: {proposal.review_reason}</p>}
          {proposal.status === "pending" && canReview
            ? <ReviewActions onReview={(approve, reason) => onReview(proposal, approve, reason)} />
            : proposal.status === "pending" && <p className="permission-note">Reviewer or admin approval required.</p>}
        </article>)}</div>}
    </KnowledgeSection>
    <KnowledgeSection title="Reviewed facts" count={data.facts.length} note="Reusable retrieval hints">
      {data.facts.length === 0 ? <KnowledgeEmpty title="No approved SpecFacts yet" detail="Approved proposals become durable facts with their source trail intact." />
        : <div className="fact-grid">{data.facts.map((fact) => <article className="approved-fact" key={fact.fact_id}>
          <div className="card-topline"><span className={`knowledge-status ${fact.status}`}>{fact.status}</span><time>{formatDate(fact.approved_at)}</time></div>
          <h4>{fact.assertion}</h4>
          <p className="fact-triple"><span>{fact.subject}</span><b>{fact.predicate}</b><span>{fact.object}</span></p>
          <SourceLine sources={fact.sources} />
        </article>)}</div>}
    </KnowledgeSection>
  </div>;
}

function SourceLine({ sources }: { sources: SpecFactProposal["sources"] }) {
  const first = sources[0];
  if (!first) return null;
  return <div className="source-line"><span>TS {first.specification} · Rel-{first.release} · v{first.version}</span><span>{first.source_path} · lines {first.line_start}–{first.line_end}</span></div>;
}

function ReviewActions({ onReview }: { onReview: (approve: boolean, reason?: string) => Promise<void> }) {
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function decide(approve: boolean) {
    if (!approve && !reason.trim()) {
      setError("Add a rejection reason before continuing.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onReview(approve, approve ? undefined : reason.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Review could not be saved");
    } finally {
      setBusy(false);
    }
  }

  return <div className="review-actions">
    {rejecting && <textarea value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Why should this proposal be rejected?" rows={2} autoFocus />}
    {error && <small className="review-error">{error}</small>}
    <div>
      <button className="approve-button" onClick={() => void decide(true)} disabled={busy}>Approve</button>
      {rejecting
        ? <><button className="reject-button" onClick={() => void decide(false)} disabled={busy}>Confirm rejection</button><button className="cancel-button" onClick={() => { setRejecting(false); setError(null); }} disabled={busy}>Cancel</button></>
        : <button className="reject-button" onClick={() => setRejecting(true)} disabled={busy}>Reject</button>}
    </div>
  </div>;
}

function KnowledgeSection({ title, count, note, children }: { title: string; count: number; note: string; children: React.ReactNode }) {
  return <section className="knowledge-section">
    <header><div><h3>{title} <span>{count}</span></h3><p>{note}</p></div></header>
    {children}
  </section>;
}

function KnowledgeEmpty({ title, detail }: { title: string; detail: string }) {
  return <div className="knowledge-empty"><span>○</span><div><strong>{title}</strong><small>{detail}</small></div></div>;
}

function formatDate(value: string) {
  return new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

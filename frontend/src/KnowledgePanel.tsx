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
import { zhDateOnly, zhLabel } from "./locale";

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
      setError(err instanceof Error ? err.message : "无法加载工作空间知识");
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
        <p className="eyebrow">已审核的工作空间知识</p>
        <h2>记忆与规范事实</h2>
        <p>查看系统保存的记忆，并在事实影响后续研究前完成审核。</p>
      </div>
      <button className="quiet-button" onClick={() => void load()} disabled={loading}>刷新</button>
    </div>
    <div className="knowledge-tabs" role="tablist" aria-label="知识类型">
      <button className={view === "memory" ? "active" : ""} onClick={() => setView("memory")} role="tab" aria-selected={view === "memory"}>
        <span>记忆</span><small>{memoryCount} 条有效 · {pendingMemory} 条待审核</small>
      </button>
      <button className={view === "specfacts" ? "active" : ""} onClick={() => setView("specfacts")} role="tab" aria-selected={view === "specfacts"} id="specfacts">
        <span>规范事实</span><small>{specFacts?.facts.length ?? 0} 条已审核 · {pendingFacts} 条待审核</small>
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
  return <div className="knowledge-loading"><span className="loader" /> 正在加载已审核知识…</div>;
}

function MemoryView({ data, role, onReview }: {
  data: MemoryDashboard | null;
  role: Role;
  onReview: (proposal: MemoryProposal, approve: boolean, reason?: string) => Promise<void>;
}) {
  if (!data) return <KnowledgeEmpty title="记忆当前不可用" detail="请刷新后重新尝试加载。" />;
  return <div className="knowledge-body">
    <KnowledgeSection title="待审核建议" count={data.proposals.length} note="人工审核边界">
      {data.proposals.length === 0
        ? <KnowledgeEmpty title="没有待审核建议" detail="提取出的记忆会先显示在这里，审核后才会长期保存。" />
        : <div className="proposal-list">{data.proposals.map((proposal) => <article className="proposal-card" key={proposal.proposal_id}>
          <div className="card-topline"><span className={`knowledge-status ${proposal.status}`}>{zhLabel(proposal.status)}</span><time>{formatDate(proposal.proposed_at)}</time></div>
          <span className="card-kicker">{proposal.draft.scope === "user" ? "用户" : "工作空间"}记忆 · {zhLabel(proposal.draft.operation)}</span>
          <h4>{proposal.draft.key ?? proposal.draft.topic_slug ?? "记忆变更"}</h4>
          <p>{proposal.draft.content ?? proposal.draft.topic_description ?? "此建议将修改一条已有的记忆记录。"}</p>
          <div className="card-meta"><span>提议人：{proposal.proposed_by}</span>{proposal.draft.kind && <span>{zhLabel(proposal.draft.kind)}</span>}</div>
          {proposal.review_reason && <p className="review-note">审核说明：{proposal.review_reason}</p>}
          {proposal.status === "pending" && (proposal.draft.scope === "user" || role !== "member")
            ? <ReviewActions onReview={(approve, reason) => onReview(proposal, approve, reason)} />
            : proposal.status === "pending" && <p className="permission-note">需要审核员或管理员批准。</p>}
        </article>)}</div>}
    </KnowledgeSection>
    <div className="knowledge-columns">
      <KnowledgeSection title="你的核心记忆" count={data.user_core.length} note="仅你的账户可见">
        <MemoryEntries entries={data.user_core} empty="尚未保存个人偏好。" />
      </KnowledgeSection>
      <KnowledgeSection title="项目核心记忆" count={data.project_core.length} note="在此工作空间内共享">
        <MemoryEntries entries={data.project_core} empty="尚无已审核的项目决策。" />
      </KnowledgeSection>
    </div>
    <KnowledgeSection title="主题记忆" count={data.topics.length} note="长期研究主题">
      {data.topics.length === 0 ? <KnowledgeEmpty title="尚无研究主题" detail="创建并审核主题后，主题记忆会显示在这里。" />
        : <div className="topic-grid">{data.topics.map(({ topic, entries }) => <article className="topic-card" key={topic.topic_id}>
          <div className="card-topline"><span className="topic-slug">{topic.slug}</span>{topic.pinned && <span className="pin-label">已置顶</span>}</div>
          <h4>{topic.description}</h4>
          <MemoryEntries entries={entries} empty="此主题没有有效记忆。" compact />
        </article>)}</div>}
    </KnowledgeSection>
  </div>;
}

function MemoryEntries({ entries, empty, compact = false }: { entries: MemoryEntry[]; empty: string; compact?: boolean }) {
  if (entries.length === 0) return <p className="inline-empty">{empty}</p>;
  return <div className={compact ? "memory-list compact" : "memory-list"}>{entries.map((entry) => <div className="memory-row" key={entry.memory_id}>
    <div><span>{zhLabel(entry.kind)}</span><strong>{entry.key}</strong></div>
    <p>{entry.content}</p>
    <small>版本 {entry.revision} · {zhLabel(entry.origin)} · {formatDate(entry.updated_at)}</small>
  </div>)}</div>;
}

function SpecFactView({ data, canReview, onReview }: {
  data: SpecFactDashboard | null;
  canReview: boolean;
  onReview: (proposal: SpecFactProposal, approve: boolean, reason?: string) => Promise<void>;
}) {
  if (!data) return <KnowledgeEmpty title="规范事实当前不可用" detail="请刷新后重新尝试加载。" />;
  return <div className="knowledge-body">
    <KnowledgeSection title="事实建议" count={data.proposals.length} note="具有证据支持，等待审核">
      {data.proposals.length === 0 ? <KnowledgeEmpty title="没有事实建议" detail="独立提取的规范事实会显示在这里等待审核。" />
        : <div className="proposal-list">{data.proposals.map((proposal) => <article className="proposal-card fact-card" key={proposal.proposal_id}>
          <div className="card-topline"><span className={`knowledge-status ${proposal.status}`}>{zhLabel(proposal.status)}</span><time>{formatDate(proposal.proposed_at)}</time></div>
          <span className="card-kicker">{proposal.subject} · {proposal.predicate}</span>
          <h4>{proposal.assertion}</h4>
          <SourceLine sources={proposal.sources} />
          {proposal.review_reason && <p className="review-note">审核说明：{proposal.review_reason}</p>}
          {proposal.status === "pending" && canReview
            ? <ReviewActions onReview={(approve, reason) => onReview(proposal, approve, reason)} />
            : proposal.status === "pending" && <p className="permission-note">需要审核员或管理员批准。</p>}
        </article>)}</div>}
    </KnowledgeSection>
    <KnowledgeSection title="已审核事实" count={data.facts.length} note="可复用的检索提示">
      {data.facts.length === 0 ? <KnowledgeEmpty title="尚无已批准的规范事实" detail="批准后的建议会成为保留完整来源链的长期事实。" />
        : <div className="fact-grid">{data.facts.map((fact) => <article className="approved-fact" key={fact.fact_id}>
          <div className="card-topline"><span className={`knowledge-status ${fact.status}`}>{zhLabel(fact.status)}</span><time>{formatDate(fact.approved_at)}</time></div>
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
  return <div className="source-line"><span>TS {first.specification} · Rel-{first.release} · v{first.version}</span><span>{first.source_path} · 第 {first.line_start}–{first.line_end} 行</span></div>;
}

function ReviewActions({ onReview }: { onReview: (approve: boolean, reason?: string) => Promise<void> }) {
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function decide(approve: boolean) {
    if (!approve && !reason.trim()) {
      setError("请先填写拒绝原因。");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await onReview(approve, approve ? undefined : reason.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法保存审核结果");
    } finally {
      setBusy(false);
    }
  }

  return <div className="review-actions">
    {rejecting && <textarea value={reason} onChange={(event) => setReason(event.target.value)} placeholder="为什么要拒绝这项建议？" rows={2} autoFocus />}
    {error && <small className="review-error">{error}</small>}
    <div>
      <button className="approve-button" onClick={() => void decide(true)} disabled={busy}>批准</button>
      {rejecting
        ? <><button className="reject-button" onClick={() => void decide(false)} disabled={busy}>确认拒绝</button><button className="cancel-button" onClick={() => { setRejecting(false); setError(null); }} disabled={busy}>取消</button></>
        : <button className="reject-button" onClick={() => setRejecting(true)} disabled={busy}>拒绝</button>}
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
  return zhDateOnly(value, { month: "short", day: "numeric", year: "numeric" });
}

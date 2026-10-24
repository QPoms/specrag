import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { ApiError, Run, RunEvent, RunInspection, SpecRagApi, WorkspaceContext, WorkspaceMembership } from "./api";
import { KnowledgePanel } from "./KnowledgePanel";
import { MemberPanel } from "./MemberPanel";
import { RunLifecyclePanel } from "./RunLifecyclePanel";
import { ConversationDesk } from "./ConversationDesk";
import { zhDate, zhLabel } from "./locale";

const api = new SpecRagApi();

function Icon({ name, size = 18 }: { name: "arrow" | "book" | "chevron" | "clock" | "grid" | "logout" | "plus" | "search" | "spark" | "users"; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };
  const paths: Record<string, ReactNode> = {
    arrow: <><path d="M5 12h13" /><path d="m13 6 6 6-6 6" /></>,
    book: <><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 0 4 21.5z" /><path d="M4 5.5v16" /><path d="M8 7h8" /></>,
    chevron: <path d="m7 10 5 5 5-5" />,
    clock: <><circle cx="12" cy="12" r="8.5" /><path d="M12 7v5l3 2" /></>,
    grid: <><rect x="4" y="4" width="6" height="6" rx="1" /><rect x="14" y="4" width="6" height="6" rx="1" /><rect x="4" y="14" width="6" height="6" rx="1" /><rect x="14" y="14" width="6" height="6" rx="1" /></>,
    logout: <><path d="M10 5H5v14h5" /><path d="M13 8l4 4-4 4" /><path d="M17 12H9" /></>,
    plus: <><path d="M12 5v14" /><path d="M5 12h14" /></>,
    search: <><circle cx="10.8" cy="10.8" r="6.2" /><path d="m16 16 4 4" /></>,
    spark: <><path d="m12 3 1.3 5.7L19 10l-5.7 1.3L12 17l-1.3-5.7L5 10l5.7-1.3z" /><path d="m19 16 .6 2.4L22 19l-2.4.6L19 22l-.6-2.4L16 19l2.4-.6z" /></>,
    users: <><path d="M16 20v-1.2a3.8 3.8 0 0 0-3.8-3.8H7.8A3.8 3.8 0 0 0 4 18.8V20" /><circle cx="10" cy="8" r="3.2" /><path d="M16 5.3a3.2 3.2 0 0 1 0 6.2" /><path d="M20 20v-1.2a3.8 3.8 0 0 0-2.7-3.6" /></>,
  };
  return <svg {...common}>{paths[name]}</svg>;
}

function Login({ onLogin }: { onLogin: (username: string, password: string) => Promise<void> }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await onLogin(username.trim(), password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法登录");
    } finally {
      setBusy(false);
    }
  }

  return <main className="login-shell">
    <section className="login-story">
      <div className="brand-mark"><span className="brand-orbit" /><span>SpecRAG</span></div>
      <div className="story-copy">
        <p className="eyebrow">证据智能 / 01</p>
        <h1>让每一项研究<br /><em>都有据可查。</em></h1>
        <p className="story-lede">一个以规范原文为依据的研究工作台，用于探索技术规范、验证证据，并将复杂问题转化为可靠结论。</p>
      </div>
      <div className="story-footer"><span className="status-dot" /> 本地 API 边界已保护 <span className="footer-rule" /> <span>v0.1 工作台</span></div>
    </section>
    <section className="login-panel">
      <div className="login-card">
        <div className="mobile-brand"><div className="brand-mark"><span className="brand-orbit" /><span>SpecRAG</span></div></div>
        <p className="eyebrow">欢迎回来</p>
        <h2>进入你的工作空间</h2>
        <p className="muted">登录后继续进行规范研究。</p>
        <form onSubmit={submit} className="auth-form">
          <label>用户名<input autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="请输入用户名" required /></label>
          <label>密码<input autoComplete="current-password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="请输入密码" required /></label>
          {error && <div className="form-error" role="alert">{error}</div>}
          <button className="primary-button full-button" disabled={busy}>{busy ? "登录中…" : <>登录 <Icon name="arrow" size={17} /></>}</button>
        </form>
        <p className="security-note"><span className="mini-lock">⊙</span> 当前会话仅能访问你所属的工作空间。</p>
      </div>
    </section>
  </main>;
}

function App() {
  const [authenticated, setAuthenticated] = useState(api.hasToken);
  const [workspaces, setWorkspaces] = useState<WorkspaceMembership[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [context, setContext] = useState<WorkspaceContext | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(api.hasToken);
  const [fatalError, setFatalError] = useState<string | null>(null);

  useEffect(() => {
    if (!authenticated) return;
    api.workspaces().then((items) => {
      setWorkspaces(items);
      setSelectedId((current) => current ?? items[0]?.workspace.workspace_id ?? null);
    }).catch((err) => {
      setFatalError(err instanceof Error ? err.message : "无法加载工作空间");
      setAuthenticated(false);
    }).finally(() => setLoading(false));
  }, [authenticated]);

  useEffect(() => {
    if (!selectedId) {
      setContext(null);
      setRuns([]);
      return;
    }
    let active = true;
    setContext(null);
    setRuns([]);
    Promise.all([api.context(selectedId), api.runs(selectedId)]).then(([nextContext, nextRuns]) => {
      if (!active) return;
      setContext(nextContext);
      setRuns(nextRuns);
    }).catch((err) => {
      if (!active) return;
      if (err instanceof ApiError && err.status === 401) setAuthenticated(false);
      setFatalError(err instanceof Error ? err.message : "无法加载工作空间");
    });
    return () => { active = false; };
  }, [selectedId]);

  const selected = useMemo(() => workspaces.find((item) => item.workspace.workspace_id === selectedId), [selectedId, workspaces]);

  async function login(username: string, password: string) {
    await api.login(username, password);
    setFatalError(null);
    setAuthenticated(true);
    setLoading(true);
  }

  async function logout() {
    await api.logout();
    setAuthenticated(false);
    setWorkspaces([]);
    setSelectedId(null);
    setContext(null);
    setRuns([]);
  }

  function selectWorkspace(workspaceId: string) {
    setRuns([]);
    setSelectedId(workspaceId);
  }

  if (!authenticated) return <Login onLogin={login} />;
  if (loading) return <div className="loading-screen"><span className="loader" /> 正在加载工作空间…</div>;
  if (!workspaces.length) return <EmptyWorkspace onCreated={(item) => { setWorkspaces([{ workspace: item, membership: { workspace_id: item.workspace_id, user_id: "", role: "admin", created_at: item.created_at } }]); setSelectedId(item.workspace_id); }} />;

  return <Dashboard context={context} selected={selected} workspaces={workspaces} selectedId={selectedId} runs={runs} onSelect={selectWorkspace} onRuns={setRuns} onLogout={logout} error={fatalError} />;
}

function EmptyWorkspace({ onCreated }: { onCreated: (workspace: WorkspaceMembership["workspace"]) => void }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function create(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(null);
    try { onCreated(await api.createWorkspace(name.trim())); } catch (err) { setError(err instanceof Error ? err.message : "无法创建工作空间"); } finally { setBusy(false); }
  }
  return <main className="empty-shell"><div className="empty-card"><div className="brand-mark"><span className="brand-orbit" /><span>SpecRAG</span></div><p className="eyebrow">先创建一个研究空间</p><h1>创建你的第一个工作空间</h1><p className="muted">工作空间集中管理研究、证据和已审核知识。</p><form onSubmit={create} className="auth-form"><label>工作空间名称<input value={name} onChange={(e) => setName(e.target.value)} placeholder="例如：协议智能研究" required /></label>{error && <div className="form-error">{error}</div>}<button className="primary-button full-button" disabled={busy}>{busy ? "创建中…" : <>创建工作空间 <Icon name="arrow" size={17} /></>}</button></form></div></main>;
}

function Dashboard({ context, selected, workspaces, selectedId, runs, onSelect, onRuns, onLogout, error }: { context: WorkspaceContext | null; selected?: WorkspaceMembership; workspaces: WorkspaceMembership[]; selectedId: string | null; runs: Run[]; onSelect: (id: string) => void; onRuns: Dispatch<SetStateAction<Run[]>>; onLogout: () => Promise<void>; error: string | null }) {
  const [question, setQuestion] = useState("");
  const [release, setRelease] = useState(17);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [workspaceMenu, setWorkspaceMenu] = useState(false);
  const [liveEvent, setLiveEvent] = useState<RunEvent | null>(null);
  const [selectedInspection, setSelectedInspection] = useState<RunInspection | null>(null);
  const [showAllRuns, setShowAllRuns] = useState(false);
  const [runAction, setRunAction] = useState<"cancel" | "resume" | "retry" | null>(null);
  const [runActionNotice, setRunActionNotice] = useState<string | null>(null);
  const watchedRuns = useRef(new Map<string, AbortController>());

  function watchRun(runId: string) {
    if (!selectedId || watchedRuns.current.has(runId)) return;
    const controller = new AbortController();
    watchedRuns.current.set(runId, controller);
    void api.streamRunEvents(selectedId, runId, (event) => {
      setLiveEvent(event);
      if (event.event_type === "run_started") {
        onRuns((current) => current.map((run) => run.run_id === runId ? { ...run, status: "running" } : run));
      }
    }, controller.signal).then(async () => {
      const inspection = await api.inspectRun(selectedId, runId);
      setSelectedInspection(inspection);
      onRuns((current) => current.map((run) => run.run_id === runId ? inspection.run : run));
    }).catch((err) => {
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        setSubmitError(err instanceof Error ? err.message : "无法持续接收研究事件");
      }
    }).finally(() => {
      watchedRuns.current.delete(runId);
    });
  }

  useEffect(() => {
    setSelectedInspection(null);
    setLiveEvent(null);
    setShowAllRuns(false);
    setRunAction(null);
    setRunActionNotice(null);
    return () => {
      watchedRuns.current.forEach((controller) => controller.abort());
      watchedRuns.current.clear();
    };
  }, [selectedId]);

  useEffect(() => {
    runs
      .filter((run) => run.status === "queued" || run.status === "running")
      .forEach((run) => watchRun(run.run_id));
    // watchRun is guarded by watchedRuns and uses the current workspace.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runs, selectedId]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!selectedId || !question.trim()) return;
    setSubmitting(true); setSubmitError(null);
    try {
      const inspection = await api.submitRun(selectedId, question.trim(), release);
      setSelectedInspection(inspection);
      onRuns((current) => [inspection.run, ...current.filter((run) => run.run_id !== inspection.run.run_id)]);
      setQuestion("");
      watchRun(inspection.run.run_id);
    } catch (err) { setSubmitError(err instanceof Error ? err.message : "无法启动研究任务"); }
    finally { setSubmitting(false); }
  }

  async function loadInspection(runId: string) {
    if (!selectedId) return;
    try {
      setSelectedInspection(await api.inspectRun(selectedId, runId));
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "无法加载研究结果");
    }
  }

  async function cancelSelectedRun() {
    if (!selectedId || !selectedInspection) return;
    setRunAction("cancel"); setSubmitError(null); setRunActionNotice(null);
    try {
      const run = await api.cancelRun(selectedId, selectedInspection.run.run_id);
      setSelectedInspection((current) => current ? { ...current, run } : current);
      onRuns((current) => current.map((item) => item.run_id === run.run_id ? run : item));
      setRunActionNotice("取消请求已记录，工作进程将在下一个安全边界停止。");
    } catch (err) { setSubmitError(err instanceof Error ? err.message : "无法取消研究任务"); }
    finally { setRunAction(null); }
  }

  async function resumeSelectedRun() {
    if (!selectedId || !selectedInspection) return;
    setRunAction("resume"); setSubmitError(null); setRunActionNotice(null);
    try {
      const inspection = await api.resumeRun(selectedId, selectedInspection.run.run_id);
      setSelectedInspection(inspection);
      onRuns((current) => current.map((item) => item.run_id === inspection.run.run_id ? inspection.run : item));
      setRunActionNotice("断点恢复已进入队列，将继续执行同一个研究任务。");
      watchRun(inspection.run.run_id);
    } catch (err) { setSubmitError(err instanceof Error ? err.message : "无法恢复研究任务"); }
    finally { setRunAction(null); }
  }

  async function retrySelectedRun() {
    if (!selectedId || !selectedInspection) return;
    setRunAction("retry"); setSubmitError(null); setRunActionNotice(null);
    try {
      const inspection = await api.retryRun(selectedId, selectedInspection.run.run_id);
      setSelectedInspection(inspection);
      onRuns((current) => [inspection.run, ...current.filter((item) => item.run_id !== inspection.run.run_id)]);
      setRunActionNotice(`已基于 ${inspection.run.retry_of_run_id} 创建新的研究任务。`);
      watchRun(inspection.run.run_id);
    } catch (err) { setSubmitError(err instanceof Error ? err.message : "无法重试研究任务"); }
    finally { setRunAction(null); }
  }

  const initials = context?.user_id.slice(0, 2).toUpperCase() ?? "ME";
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand-mark"><span className="brand-orbit" /><span>SpecRAG</span></div>
      <div className="workspace-switcher">
        <button className="workspace-button" onClick={() => setWorkspaceMenu(!workspaceMenu)} aria-expanded={workspaceMenu}><span className="workspace-avatar">{selected?.workspace.name.slice(0, 1).toUpperCase()}</span><span className="workspace-label"><small>工作空间</small><strong>{selected?.workspace.name ?? "加载中…"}</strong></span><Icon name="chevron" size={16} /></button>
        {workspaceMenu && <div className="workspace-menu">{workspaces.map((item) => <button key={item.workspace.workspace_id} onClick={() => { onSelect(item.workspace.workspace_id); setWorkspaceMenu(false); }} className={item.workspace.workspace_id === selectedId ? "active" : ""}><span>{item.workspace.name}</span><small>{zhLabel(item.membership.role)}</small></button>)}</div>}
      </div>
      <nav className="main-nav"><span className="nav-section">工作空间</span><a className="active" href="#research"><Icon name="spark" />研究工作台</a><a href="#runs"><Icon name="clock" />任务历史</a><a href="#evidence"><Icon name="book" />证据库</a><span className="nav-section second">知识</span><a href="#knowledge"><Icon name="grid" />记忆与规范事实</a><span className="nav-section second">管理</span><a href="#members"><Icon name="users" />成员</a></nav>
      <div className="sidebar-bottom"><div className="api-status"><span className="status-dot" /><div><strong>API 已连接</strong><small>本地安全边界</small></div></div><button className="user-row" onClick={() => void onLogout()}><span className="user-avatar">{initials}</span><span><strong>{context?.user_id ?? "已登录"}</strong><small>{zhLabel(context?.role ?? "member")}</small></span><Icon name="logout" size={17} /></button></div>
    </aside>
    <main className="content">
      <header className="topbar"><div><p className="eyebrow">{zhDate(new Date(), { weekday: "long", month: "long", day: "numeric" })}</p><h1>你好，研究者。</h1></div><div className="topbar-actions"><button className="icon-button" aria-label="搜索"><Icon name="search" /></button><button className="avatar-button">{initials}</button></div></header>
      {error && <div className="notice error"><span>!</span>{error}</div>}
      {selectedId && <ConversationDesk api={api} workspaceId={selectedId} onOpenRun={(runId) => void loadInspection(runId)} />}
      <section className="hero-grid" id="research"><div className="hero-card"><div className="hero-glow" /><div className="hero-content"><span className="pill"><span className="status-dot" /> 工作空间已就绪</span><h2>你想研究和理解<br /><em>什么问题？</em></h2><p>提出问题，SpecRAG 会将答案追溯到规范原文。</p></div><div className="hero-index">01 <span>/</span> 04</div></div><div className="stat-card"><span className="stat-icon"><Icon name="clock" /></span><span className="stat-label">正在执行</span><strong>{runs.filter((run) => run.status === "queued" || run.status === "running").length.toString().padStart(2, "0")}</strong><small>当前工作空间</small><div className="stat-line"><span style={{ width: runs.length ? "42%" : "8%" }} /></div></div><div className="stat-card accent-card"><span className="stat-icon"><Icon name="book" /></span><span className="stat-label">规范语料库</span><strong>3GPP</strong><small>版本固定且可追溯</small><div className="corpus-dots"><i /><i /><i /><i /><i /><i /><i /></div></div></section>
      <section className="research-panel"><div className="section-heading"><div><p className="eyebrow">新建研究任务</p><h2>从一个明确的问题开始</h2></div><span className="shortcut">⌘ ↵ <small>开始研究</small></span></div><form onSubmit={submit}><div className="question-wrap"><span className="question-number">01</span><textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="例如：5G 注册流程如何处理鉴权失败？" rows={3} /><span className="question-hint">问题越具体，基于规范证据的回答越准确。</span></div><div className="form-toolbar"><label className="select-label">版本范围<select value={release} onChange={(e) => setRelease(Number(e.target.value))}><option value={17}>Release 17</option><option value={18}>Release 18</option><option value={19}>Release 19</option></select><Icon name="chevron" size={14} /></label><span className="toolbar-divider" /><span className="budget-hint"><span className="status-dot" /> 标准深度 · 4 次迭代</span><button className="primary-button run-button" disabled={submitting || !question.trim()}>{submitting ? "正在提交…" : <>开始研究 <Icon name="arrow" size={17} /></>}</button></div>{submitError && <div className="form-error inline-error">{submitError}</div>}</form></section>
      <section className="lower-grid" id="runs"><div className="activity-panel"><div className="section-heading compact"><div><p className="eyebrow">你的研究台</p><h2>{showAllRuns ? "任务历史" : "最近研究"}</h2></div>{runs.length > 4 && <button className="text-button" onClick={() => setShowAllRuns(!showAllRuns)}>{showAllRuns ? "仅看最近" : "查看全部"} <Icon name="arrow" size={15} /></button>}</div>{liveEvent && <div className="live-event"><span className="live-pulse" /><span><strong>实时研究进度</strong><small>{zhLabel(liveEvent.stage ?? liveEvent.event_type)}</small></span><time>#{liveEvent.sequence}</time></div>}{runs.length === 0 ? <div className="empty-activity"><div className="empty-icon"><Icon name="spark" /></div><p>还没有研究任务。</p><small>提交第一个问题后，它会显示在这里。</small></div> : <div className={`run-list ${showAllRuns ? "expanded" : ""}`}>{runs.slice(0, showAllRuns ? runs.length : 4).map((run) => <div className="run-row" key={run.run_id} role="button" tabIndex={0} onClick={() => void loadInspection(run.run_id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") void loadInspection(run.run_id); }}><span className={`run-status ${run.status}`} /><div className="run-info"><strong>{run.question}</strong><small>{zhDate(run.created_at)} · Release {run.release}</small></div><span className={`run-badge ${run.status}`}>{zhLabel(run.status)}</span><Icon name="arrow" size={15} /></div>)}</div>}</div><aside className="tip-card"><span className="tip-kicker"><Icon name="spark" size={15} /> 怎样提出更好的问题</span><h3>从要做的判断出发，而不是从文档名称出发。</h3><p>清楚说明你需要判断什么，检索计划会寻找相关条款。</p><span className="tip-rule" /><small>研究提示 / 001</small></aside></section>
      {selectedInspection && <RunLifecyclePanel inspection={selectedInspection} busy={runAction} notice={runActionNotice} onCancel={cancelSelectedRun} onResume={resumeSelectedRun} onRetry={retrySelectedRun} />}
      {selectedInspection && <EvidencePanel inspection={selectedInspection} />}
      {selectedId && context && <KnowledgePanel api={api} workspaceId={selectedId} role={context.role} />}
      {selectedId && context && <MemberPanel api={api} workspaceId={selectedId} currentUserId={context.user_id} role={context.role} />}
      <footer className="content-footer"><span>SpecRAG 规范智能研究工作台</span><span><span className="status-dot" /> 所有服务均在本地运行</span></footer>
    </main>
  </div>;
}

function EvidencePanel({ inspection }: { inspection: RunInspection }) {
  const completed = inspection.run.status === "completed";
  const active = inspection.run.status === "queued" || inspection.run.status === "running";
  const answer = inspection.answer?.trim();
  return <section className="evidence-panel" id="evidence">
    <div className="section-heading evidence-heading"><div><p className="eyebrow">规范证据支持的结果</p><h2>{completed ? "有据可查的研究答案" : "研究结果"}</h2><p className="result-meta">{inspection.run.run_id} · Release {inspection.run.release} · {zhLabel(inspection.run.status)}</p></div><span className={`result-status ${inspection.run.status}`}><span className="status-dot" /> {zhLabel(inspection.run.status)}</span></div>
    {!answer ? <div className="result-empty"><div className="empty-icon"><Icon name="clock" /></div><p>{completed ? "本次研究没有返回最终综合答案。" : active ? "此任务仍在收集和验证证据。" : "此任务在生成最终综合答案前已经停止。"}</p><small>{active ? "任务结束后再次打开即可刷新结果。" : "请查看上方的生命周期诊断，然后恢复任务或创建一次全新重试。"}</small></div> : <>
      <div className="answer-block"><span className="answer-label">综合结论</span><p>{answer}</p></div>
      <div className="evidence-heading-row"><span className="answer-label">支持证据</span><span>{inspection.evidence.length} 条已验证来源</span></div>
      {inspection.evidence.length === 0 ? <div className="result-empty compact-empty"><p>本次综合答案没有返回证据卡片。</p></div> : <div className="evidence-list">{inspection.evidence.map((item, index) => <article className="evidence-card" key={item.evidence_id}><div className="evidence-card-top"><span className="evidence-number">E{String(index + 1).padStart(2, "0")}</span><span className="evidence-spec">TS {item.specification} · Rel-{item.release}</span><span className="evidence-version">v{item.version}</span></div><div className="evidence-id"><code>[{item.evidence_id}]</code></div><h3>{item.heading_path.length ? item.heading_path.join(" / ") : "规范摘录"}</h3><blockquote>{item.excerpt}</blockquote><div className="evidence-source"><span>{item.source_path}</span><span>第 {item.line_start}–{item.line_end} 行</span></div><div className="evidence-hash">SHA-256 <code>{item.document_sha256.slice(0, 16)}…</code></div></article>)}</div>}
    </>}
  </section>;
}

export default App;

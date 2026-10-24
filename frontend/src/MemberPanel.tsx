import { FormEvent, useEffect, useMemo, useState } from "react";
import { Role, SpecRagApi, WorkspaceMember } from "./api";
import { zhDateOnly, zhLabel } from "./locale";

type Props = {
  api: SpecRagApi;
  workspaceId: string;
  currentUserId: string;
  role: Role;
};

const roleCopy: Record<Role, string> = {
  member: "执行研究、读取共享知识并提交新的事实建议。",
  reviewer: "还可审核知识，并管理项目记忆和主题记忆。",
  admin: "还可管理成员，并查看工作空间中的全部研究任务。",
};

export function MemberPanel({ api, workspaceId, currentUserId, role }: Props) {
  const [members, setMembers] = useState<WorkspaceMember[]>([]);
  const [loading, setLoading] = useState(role === "admin");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function load() {
    if (role !== "admin") return;
    setLoading(true);
    setError(null);
    try {
      setMembers(await api.members(workspaceId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法加载成员列表");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setMembers([]);
    setNotice(null);
    void load();
    // Membership is reloaded whenever the trusted workspace context changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceId, role]);

  const counts = useMemo(() => ({
    member: members.filter((item) => item.membership.role === "member").length,
    reviewer: members.filter((item) => item.membership.role === "reviewer").length,
    admin: members.filter((item) => item.membership.role === "admin").length,
  }), [members]);

  if (role !== "admin") return <section className="member-panel member-locked" id="members">
    <div className="member-lock-mark">◇</div>
    <div><p className="eyebrow">工作空间管理</p><h2>仅管理员可以管理成员</h2><p>你当前的<strong>{zhLabel(role)}</strong>角色可以使用研究和共享知识，但无权查看工作空间账户目录。</p></div>
  </section>;

  async function mutate(action: () => Promise<unknown>, message: string) {
    setError(null);
    setNotice(null);
    try {
      await action();
      setNotice(message);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "成员信息更新失败");
      throw err;
    }
  }

  return <section className="member-panel" id="members">
    <div className="member-heading">
      <div><p className="eyebrow">工作空间管理</p><h2>成员与权限</h2><p>谨慎授予访问权限。角色会影响研究可见性、知识审核和工作空间管理。</p></div>
      <div className="member-total"><strong>{members.length.toString().padStart(2, "0")}</strong><span>位成员</span></div>
    </div>
    <div className="role-summary">{(["member", "reviewer", "admin"] as Role[]).map((item) => <div key={item}><span>{zhLabel(item)}</span><strong>{counts[item]}</strong><small>{roleCopy[item]}</small></div>)}</div>
    {notice && <div className="member-notice success">{notice}</div>}
    {error && <div className="member-notice error" role="alert">{error}</div>}
    <div className="member-layout">
      <div className="member-directory">
        <div className="member-subheading"><div><h3>工作空间成员目录</h3><p>角色变更将在下一次授权请求时生效。</p></div><button className="quiet-button" onClick={() => void load()} disabled={loading}>刷新</button></div>
        {loading && members.length === 0 ? <div className="member-loading"><span className="loader" /> 正在加载成员…</div>
          : <div className="member-list">{members.map((item) => <MemberRow
            key={item.user.user_id}
            item={item}
            isCurrent={item.user.user_id === currentUserId}
            onRole={async (nextRole) => {
              await mutate(
                () => api.setMemberRole(workspaceId, item.user.username, nextRole),
                `${item.user.username} 现在是${zhLabel(nextRole)}。`,
              );
            }}
            onRemove={async () => {
              await mutate(
                () => api.removeMember(workspaceId, item.user.username),
                `已将 ${item.user.username} 移出此工作空间。`,
              );
            }}
          />)}</div>}
      </div>
      <AddMemberForm onAdd={async (username, password, nextRole) => {
        await mutate(
          () => api.addMember(workspaceId, username, password, nextRole),
          `${username} 现在可以访问此工作空间。`,
        );
      }} />
    </div>
  </section>;
}

function MemberRow({ item, isCurrent, onRole, onRemove }: {
  item: WorkspaceMember;
  isCurrent: boolean;
  onRole: (role: Role) => Promise<void>;
  onRemove: () => Promise<void>;
}) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const initials = item.user.username.slice(0, 2).toUpperCase();

  async function changeRole(nextRole: Role) {
    setBusy(true);
    try { await onRole(nextRole); } catch { /* The parent renders the API error. */ }
    finally { setBusy(false); }
  }

  async function remove() {
    setBusy(true);
    try { await onRemove(); } catch { setBusy(false); }
  }

  return <article className="member-row">
    <span className="member-avatar">{initials}</span>
    <div className="member-identity"><strong>{item.user.username}{isCurrent && <em>你</em>}</strong><small>加入于 {formatDate(item.membership.created_at)} · {zhLabel(item.user.status)}</small></div>
    <label className="role-select"><span className="sr-only">{item.user.username} 的角色</span><select value={item.membership.role} disabled={busy || isCurrent} onChange={(event) => void changeRole(event.target.value as Role)}><option value="member">成员</option><option value="reviewer">审核员</option><option value="admin">管理员</option></select></label>
    {isCurrent ? <span className="self-protected">受保护</span> : confirming
      ? <div className="remove-confirm"><button onClick={() => void remove()} disabled={busy}>移除</button><button onClick={() => setConfirming(false)} disabled={busy}>保留</button></div>
      : <button className="remove-member" onClick={() => setConfirming(true)} aria-label={`移除 ${item.user.username}`}>×</button>}
  </article>;
}

function AddMemberForm({ onAdd }: { onAdd: (username: string, password: string, role: Role) => Promise<void> }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>("member");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await onAdd(username.trim(), password, role);
      setUsername("");
      setPassword("");
      setRole("member");
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法添加成员");
    } finally {
      setBusy(false);
    }
  }

  return <aside className="add-member-card">
    <span className="add-member-index">＋</span>
    <p className="eyebrow">授予访问权限</p>
    <h3>添加工作空间成员</h3>
    <p>创建本地账户，或通过用户名添加已有账户。此操作不会修改已有账户的密码。</p>
    <form onSubmit={submit}>
      <label>用户名<input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="off" required /></label>
      <label>初始密码<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="new-password" minLength={10} required /></label>
      <label>角色<select value={role} onChange={(event) => setRole(event.target.value as Role)}><option value="member">成员</option><option value="reviewer">审核员</option><option value="admin">管理员</option></select></label>
      {error && <small className="review-error">{error}</small>}
      <button className="primary-button" disabled={busy}>{busy ? "添加中…" : "添加成员"}</button>
    </form>
  </aside>;
}

function formatDate(value: string) {
  return zhDateOnly(value, { month: "short", day: "numeric", year: "numeric" });
}

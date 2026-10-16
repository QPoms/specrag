import { FormEvent, useEffect, useMemo, useState } from "react";
import { Role, SpecRagApi, WorkspaceMember } from "./api";

type Props = {
  api: SpecRagApi;
  workspaceId: string;
  currentUserId: string;
  role: Role;
};

const roleCopy: Record<Role, string> = {
  member: "Runs research, reads shared knowledge, and proposes new facts.",
  reviewer: "Also reviews knowledge and manages Project and Topic Memory.",
  admin: "Also manages members and can inspect every Run in the workspace.",
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
      setError(err instanceof Error ? err.message : "Unable to load members");
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
    <div><p className="eyebrow">Workspace administration</p><h2>Member management is admin-only</h2><p>Your <strong>{role}</strong> role can use research and shared knowledge without receiving the workspace account directory.</p></div>
  </section>;

  async function mutate(action: () => Promise<unknown>, message: string) {
    setError(null);
    setNotice(null);
    try {
      await action();
      setNotice(message);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Membership update failed");
      throw err;
    }
  }

  return <section className="member-panel" id="members">
    <div className="member-heading">
      <div><p className="eyebrow">Workspace administration</p><h2>People &amp; permissions</h2><p>Keep access intentional. Roles affect research visibility, knowledge review, and workspace administration.</p></div>
      <div className="member-total"><strong>{members.length.toString().padStart(2, "0")}</strong><span>members</span></div>
    </div>
    <div className="role-summary">{(["member", "reviewer", "admin"] as Role[]).map((item) => <div key={item}><span>{item}</span><strong>{counts[item]}</strong><small>{roleCopy[item]}</small></div>)}</div>
    {notice && <div className="member-notice success">{notice}</div>}
    {error && <div className="member-notice error" role="alert">{error}</div>}
    <div className="member-layout">
      <div className="member-directory">
        <div className="member-subheading"><div><h3>Workspace directory</h3><p>Roles take effect on the next authorized request.</p></div><button className="quiet-button" onClick={() => void load()} disabled={loading}>Refresh</button></div>
        {loading && members.length === 0 ? <div className="member-loading"><span className="loader" /> Loading members…</div>
          : <div className="member-list">{members.map((item) => <MemberRow
            key={item.user.user_id}
            item={item}
            isCurrent={item.user.user_id === currentUserId}
            onRole={async (nextRole) => {
              await mutate(
                () => api.setMemberRole(workspaceId, item.user.username, nextRole),
                `${item.user.username} is now ${nextRole}.`,
              );
            }}
            onRemove={async () => {
              await mutate(
                () => api.removeMember(workspaceId, item.user.username),
                `${item.user.username} was removed from this workspace.`,
              );
            }}
          />)}</div>}
      </div>
      <AddMemberForm onAdd={async (username, password, nextRole) => {
        await mutate(
          () => api.addMember(workspaceId, username, password, nextRole),
          `${username} can now access this workspace.`,
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
    <div className="member-identity"><strong>{item.user.username}{isCurrent && <em>You</em>}</strong><small>Joined {formatDate(item.membership.created_at)} · {item.user.status}</small></div>
    <label className="role-select"><span className="sr-only">Role for {item.user.username}</span><select value={item.membership.role} disabled={busy || isCurrent} onChange={(event) => void changeRole(event.target.value as Role)}><option value="member">Member</option><option value="reviewer">Reviewer</option><option value="admin">Admin</option></select></label>
    {isCurrent ? <span className="self-protected">Protected</span> : confirming
      ? <div className="remove-confirm"><button onClick={() => void remove()} disabled={busy}>Remove</button><button onClick={() => setConfirming(false)} disabled={busy}>Keep</button></div>
      : <button className="remove-member" onClick={() => setConfirming(true)} aria-label={`Remove ${item.user.username}`}>×</button>}
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
      setError(err instanceof Error ? err.message : "Unable to add member");
    } finally {
      setBusy(false);
    }
  }

  return <aside className="add-member-card">
    <span className="add-member-index">＋</span>
    <p className="eyebrow">Grant access</p>
    <h3>Add a workspace member</h3>
    <p>Create a local account or add an existing account by username. Existing account passwords are never changed here.</p>
    <form onSubmit={submit}>
      <label>Username<input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="off" required /></label>
      <label>Initial password<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="new-password" minLength={10} required /></label>
      <label>Role<select value={role} onChange={(event) => setRole(event.target.value as Role)}><option value="member">Member</option><option value="reviewer">Reviewer</option><option value="admin">Admin</option></select></label>
      {error && <small className="review-error">{error}</small>}
      <button className="primary-button" disabled={busy}>{busy ? "Adding…" : "Add member"}</button>
    </form>
  </aside>;
}

function formatDate(value: string) {
  return new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

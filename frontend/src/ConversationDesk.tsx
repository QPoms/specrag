import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Conversation,
  ConversationDetail,
  ConversationMessage,
  ConversationMessageContent,
  NativeToolCapability,
  SpecRagApi,
} from "./api";
import "./ConversationDesk.css";
import { zhDate, zhDateOnly, zhLabel } from "./locale";

type Props = {
  api: SpecRagApi;
  workspaceId: string;
  onOpenRun: (runId: string) => void;
};

function contentOf(message: ConversationMessage): ConversationMessageContent {
  const content = message.content as Partial<ConversationMessageContent>;
  return {
    text: typeof content.text === "string" ? content.text : JSON.stringify(message.content),
    source_kind: content.source_kind ?? null,
    research_result_ids: content.research_result_ids ?? [],
    run_ids: content.run_ids ?? [],
    citations: content.citations ?? [],
    status_kind: content.status_kind ?? null,
  };
}

function conversationLabel(item: Conversation): string {
  return item.title || `会话 ${item.conversation_id.slice(-6)}`;
}

export function ConversationDesk({ api, workspaceId, onOpenRun }: Props) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ConversationDetail | null>(null);
  const [draft, setDraft] = useState("");
  const [newTitle, setNewTitle] = useState("");
  const [creating, setCreating] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [capability, setCapability] = useState<NativeToolCapability | null>(null);
  const cursor = useRef(0);
  const appliedCursor = useRef(0);

  async function refreshList(preferred?: string) {
    const items = await api.conversations(workspaceId);
    setConversations(items);
    setSelectedId((current) => preferred ?? current ?? items[0]?.conversation_id ?? null);
  }

  useEffect(() => {
    setDetail(null);
    setSelectedId(null);
    setError(null);
    void refreshList().catch((err) => setError(err instanceof Error ? err.message : "无法加载会话列表"));
    // Workspace changes reset this private conversation projection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceId]);

  useEffect(() => {
    void api.health()
      .then((health) => setCapability(health.conversation_agent))
      .catch(() => setCapability({
        status: "unavailable",
        api_mode: null,
        failure_kind: "health_check_failed",
        status_code: null,
        retryable: true,
        message: "无法读取对话智能体的启动能力检测结果。",
      }));
  }, [api]);

  useEffect(() => {
    if (!selectedId) { setDetail(null); return; }
    const controller = new AbortController();
    let alive = true;
    async function load() {
      const snapshot = await api.conversation(workspaceId, selectedId!);
      if (!alive) return;
      cursor.current = snapshot.event_cursor;
      appliedCursor.current = snapshot.event_cursor;
      setDetail(snapshot);
    }
    async function follow() {
      try {
        await load();
        while (alive && !controller.signal.aborted) {
          await api.streamConversationEvents(
            workspaceId,
            selectedId!,
            cursor.current,
            (event) => {
              if (event.sequence <= cursor.current) return;
              cursor.current = event.sequence;
              void api.conversation(workspaceId, selectedId!).then((snapshot) => {
                if (!alive || snapshot.event_cursor < appliedCursor.current) return;
                appliedCursor.current = snapshot.event_cursor;
                cursor.current = Math.max(cursor.current, snapshot.event_cursor);
                setDetail(snapshot);
              });
            },
            controller.signal,
          );
          if (!alive) break;
          const snapshot = await api.conversation(workspaceId, selectedId!);
          if (!alive) break;
          cursor.current = snapshot.event_cursor;
          appliedCursor.current = snapshot.event_cursor;
          setDetail(snapshot);
          setConversations((current) => current.map((item) =>
            item.conversation_id === snapshot.conversation.conversation_id
              ? snapshot.conversation : item,
          ));
          setError(null); // The authoritative snapshot also repairs any cursor gap.
        }
      } catch (err) {
        if (!(err instanceof DOMException && err.name === "AbortError") && alive) {
          setError(err instanceof Error ? err.message : "会话连接已中断");
        }
      }
    }
    void follow();
    return () => { alive = false; controller.abort(); };
  }, [api, selectedId, workspaceId]);

  async function create(event: FormEvent) {
    event.preventDefault();
    setCreating(true); setError(null);
    try {
      const created = await api.createConversation(workspaceId, newTitle);
      setNewTitle("");
      setDetail(created);
      cursor.current = created.event_cursor;
      appliedCursor.current = created.event_cursor;
      await refreshList(created.conversation.conversation_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法创建会话");
    } finally { setCreating(false); }
  }

  async function send(event: FormEvent) {
    event.preventDefault();
    if (!detail || !draft.trim() || capability?.status !== "ready") return;
    const text = draft.trim();
    setSending(true); setError(null);
    try {
      await api.appendConversationMessage(
        workspaceId,
        detail.conversation.conversation_id,
        text,
        `web-${crypto.randomUUID()}`,
        detail.conversation.revision,
      );
      setDraft("");
      const snapshot = await api.conversation(workspaceId, detail.conversation.conversation_id);
      cursor.current = snapshot.event_cursor;
      appliedCursor.current = snapshot.event_cursor;
      setDetail(snapshot);
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法发送消息");
    } finally { setSending(false); }
  }

  async function toggleArchive() {
    if (!detail) return;
    const archived = detail.conversation.status === "active";
    setError(null);
    try {
      await api.setConversationArchived(
        workspaceId,
        detail.conversation.conversation_id,
        archived,
        detail.conversation.revision,
      );
      const snapshot = await api.conversation(workspaceId, detail.conversation.conversation_id);
      setDetail(snapshot);
      await refreshList(snapshot.conversation.conversation_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法更新会话");
    }
  }

  const turns = useMemo(
    () => new Map((detail?.turns ?? []).map((turn) => [turn.turn_id, turn])),
    [detail],
  );
  const linksByMessage = useMemo(() => {
    const links = new Map<string, NonNullable<ConversationDetail>["research_links"]>();
    for (const link of detail?.research_links ?? []) {
      links.set(link.origin_message_id, [...(links.get(link.origin_message_id) ?? []), link]);
    }
    return links;
  }, [detail]);
  const toolsByTurn = useMemo(() => {
    const tools = new Map<string, NonNullable<ConversationDetail>["tool_calls"]>();
    for (const call of detail?.tool_calls ?? []) {
      tools.set(call.turn_id, [...(tools.get(call.turn_id) ?? []), call]);
    }
    return tools;
  }, [detail]);

  return <section className="conversation-desk" id="research">
    <aside className="conversation-index">
      <div className="conversation-index-heading">
        <div><p className="eyebrow">研究主题</p><h2>会话</h2></div>
        <span>{conversations.length}</span>
      </div>
      <form className="new-conversation" onSubmit={create}>
        <input value={newTitle} onChange={(event) => setNewTitle(event.target.value)} placeholder="可选的主题标题" />
        <button disabled={creating}>{creating ? "…" : "+"}<span className="sr-only">创建会话</span></button>
      </form>
      <div className="conversation-list">
        {conversations.map((item) => <button
          key={item.conversation_id}
          className={item.conversation_id === selectedId ? "selected" : ""}
          onClick={() => setSelectedId(item.conversation_id)}
        >
          <strong>{conversationLabel(item)}</strong>
          <small>{zhLabel(item.status)} · {item.last_message_sequence} 条消息</small>
          <time>{zhDateOnly(item.updated_at)}</time>
        </button>)}
        {!conversations.length && <div className="conversation-list-empty">创建一个会话，开始持续研究。</div>}
      </div>
    </aside>
    <div className="conversation-thread">
      <header className="thread-heading">
        <div>
          <p className="eyebrow">持久研究会话</p>
          <h2>{detail ? conversationLabel(detail.conversation) : "请选择一个会话"}</h2>
          {detail && <small>版本 {detail.conversation.revision} · 事件 #{detail.event_cursor}</small>}
        </div>
        {detail && <button className="quiet-button" onClick={() => void toggleArchive()} disabled={Boolean(detail.conversation.active_turn_id)}>
          {detail.conversation.status === "active" ? "归档" : "恢复"}
        </button>}
      </header>
      {capability && capability.status !== "ready" && <div className="conversation-capability-error">
        <strong>对话智能体当前不可用</strong>
        <span>{capability.message}</span>
        <small>{[
          capability.failure_kind,
          capability.status_code ? `HTTP ${capability.status_code}` : null,
          capability.api_mode,
        ].filter(Boolean).join(" · ")}</small>
      </div>}
      {error && <div className="conversation-error">{error}</div>}
      <div className="message-thread" aria-live="polite">
        {!detail && <div className="thread-empty">请选择已有会话，或创建一个新会话。</div>}
        {detail && !detail.messages.length && <div className="thread-empty">可以从任何问题开始；涉及规范的问题会自动发起证据驱动的研究。</div>}
        {detail?.messages.map((message) => {
          const content = contentOf(message);
          const turn = message.turn_id ? turns.get(message.turn_id) : undefined;
          const links = linksByMessage.get(message.message_id) ?? [];
          const calls = message.role === "user" && message.turn_id
            ? toolsByTurn.get(message.turn_id) ?? [] : [];
          return <article className={`conversation-message ${message.role}`} key={message.message_id}>
            <div className="message-meta"><span>{message.role === "user" ? "你" : "SpecRAG"}</span><time>{zhDate(message.created_at)}</time></div>
            <p>{content.text}</p>
            {content.source_kind && <div className="source-attribution">
              <span>{zhLabel(content.source_kind)}</span>
              {content.citations.map((citation) => <code key={citation}>[{citation}]</code>)}
            </div>}
            {turn && <div className={`turn-status ${turn.status}`}><i /> 本轮{zhLabel(turn.status)}{turn.failure_kind ? ` · ${turn.failure_kind}` : ""}{turn.failure_message && <small>{turn.failure_message}</small>}</div>}
            {calls.map((call) => <div className={`tool-audit ${call.status}`} key={call.call_id}>
              <span>{zhLabel(call.tool_name)}</span>
              <strong>{zhLabel(call.status)}</strong>
              {call.safe_error && <small>{call.safe_error}</small>}
            </div>)}
            {links.map((link) => <button className="research-link-card" key={link.link_id} onClick={() => onOpenRun(link.run_id)}>
              <span><i className={`run-status ${link.task_status}`} /><strong>研究{zhLabel(link.task_status)}</strong></span>
              <small>{link.latest_result_id ? `结果 ${link.latest_result_id.slice(-8)}` : `任务 ${link.run_id.slice(-8)}`}</small>
              <em>{zhLabel(link.presentation_status ?? "executing")}{link.status_reason ? ` · ${link.status_reason}` : ""}</em>
            </button>)}
          </article>;
        })}
      </div>
      {detail && <form className="conversation-composer" onSubmit={send}>
        <textarea
          rows={3}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={detail.conversation.status === "archived" ? "恢复此会话后才能继续。" : "继续讨论，或提出一个规范问题…"}
          disabled={detail.conversation.status === "archived" || sending || capability?.status !== "ready"}
        />
        <div><small>会话历史、约束条件和先前的研究结果会持续关联到此会话。</small><button className="primary-button" disabled={sending || !draft.trim() || detail.conversation.status === "archived" || capability?.status !== "ready"}>{sending ? "发送中…" : "发送"}</button></div>
      </form>}
    </div>
  </section>;
}

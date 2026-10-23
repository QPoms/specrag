import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Conversation,
  ConversationDetail,
  ConversationMessage,
  ConversationMessageContent,
  SpecRagApi,
} from "./api";
import "./ConversationDesk.css";

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
  return item.title || `Conversation ${item.conversation_id.slice(-6)}`;
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
    void refreshList().catch((err) => setError(err instanceof Error ? err.message : "Unable to load conversations"));
    // Workspace changes reset this private conversation projection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceId]);

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
          setError(err instanceof Error ? err.message : "Conversation connection lost");
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
      setError(err instanceof Error ? err.message : "Unable to create conversation");
    } finally { setCreating(false); }
  }

  async function send(event: FormEvent) {
    event.preventDefault();
    if (!detail || !draft.trim()) return;
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
      setError(err instanceof Error ? err.message : "Unable to send message");
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
      setError(err instanceof Error ? err.message : "Unable to update conversation");
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
        <div><p className="eyebrow">Research threads</p><h2>Conversations</h2></div>
        <span>{conversations.length}</span>
      </div>
      <form className="new-conversation" onSubmit={create}>
        <input value={newTitle} onChange={(event) => setNewTitle(event.target.value)} placeholder="Optional topic title" />
        <button disabled={creating}>{creating ? "…" : "+"}<span className="sr-only">Create conversation</span></button>
      </form>
      <div className="conversation-list">
        {conversations.map((item) => <button
          key={item.conversation_id}
          className={item.conversation_id === selectedId ? "selected" : ""}
          onClick={() => setSelectedId(item.conversation_id)}
        >
          <strong>{conversationLabel(item)}</strong>
          <small>{item.status} · {item.last_message_sequence} messages</small>
          <time>{new Date(item.updated_at).toLocaleDateString()}</time>
        </button>)}
        {!conversations.length && <div className="conversation-list-empty">Create a thread to begin a persistent study.</div>}
      </div>
    </aside>
    <div className="conversation-thread">
      <header className="thread-heading">
        <div>
          <p className="eyebrow">Persistent research conversation</p>
          <h2>{detail ? conversationLabel(detail.conversation) : "Select a conversation"}</h2>
          {detail && <small>Revision {detail.conversation.revision} · event #{detail.event_cursor}</small>}
        </div>
        {detail && <button className="quiet-button" onClick={() => void toggleArchive()} disabled={Boolean(detail.conversation.active_turn_id)}>
          {detail.conversation.status === "active" ? "Archive" : "Restore"}
        </button>}
      </header>
      {error && <div className="conversation-error">{error}</div>}
      <div className="message-thread" aria-live="polite">
        {!detail && <div className="thread-empty">Choose an existing conversation or create a new one.</div>}
        {detail && !detail.messages.length && <div className="thread-empty">Ask anything. Specification questions will automatically start evidence-backed research.</div>}
        {detail?.messages.map((message) => {
          const content = contentOf(message);
          const turn = message.turn_id ? turns.get(message.turn_id) : undefined;
          const links = linksByMessage.get(message.message_id) ?? [];
          const calls = message.role === "user" && message.turn_id
            ? toolsByTurn.get(message.turn_id) ?? [] : [];
          return <article className={`conversation-message ${message.role}`} key={message.message_id}>
            <div className="message-meta"><span>{message.role === "user" ? "You" : "SpecRAG"}</span><time>{new Date(message.created_at).toLocaleString()}</time></div>
            <p>{content.text}</p>
            {content.source_kind && <div className="source-attribution">
              <span>{content.source_kind.replaceAll("_", " ")}</span>
              {content.citations.map((citation) => <code key={citation}>[{citation}]</code>)}
            </div>}
            {turn && <div className={`turn-status ${turn.status}`}><i /> Turn {turn.status}{turn.failure_kind ? ` · ${turn.failure_kind}` : ""}</div>}
            {calls.map((call) => <div className={`tool-audit ${call.status}`} key={call.call_id}>
              <span>{call.tool_name.replaceAll("_", " ")}</span>
              <strong>{call.status}</strong>
              {call.safe_error && <small>{call.safe_error}</small>}
            </div>)}
            {links.map((link) => <button className="research-link-card" key={link.link_id} onClick={() => onOpenRun(link.run_id)}>
              <span><i className={`run-status ${link.task_status}`} /><strong>Research {link.task_status}</strong></span>
              <small>{link.latest_result_id ? `Result ${link.latest_result_id.slice(-8)}` : `Run ${link.run_id.slice(-8)}`}</small>
              <em>{link.presentation_status ?? "executing"}{link.status_reason ? ` · ${link.status_reason}` : ""}</em>
            </button>)}
          </article>;
        })}
      </div>
      {detail && <form className="conversation-composer" onSubmit={send}>
        <textarea
          rows={3}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={detail.conversation.status === "archived" ? "Restore this conversation to continue." : "Continue the discussion or ask a specification question…"}
          disabled={detail.conversation.status === "archived" || sending}
        />
        <div><small>Conversation history, constraints, and prior Results stay attached to this thread.</small><button className="primary-button" disabled={sending || !draft.trim() || detail.conversation.status === "archived"}>{sending ? "Sending…" : "Send"}</button></div>
      </form>}
    </div>
  </section>;
}

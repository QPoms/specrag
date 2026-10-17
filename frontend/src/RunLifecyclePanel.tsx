import { useState } from "react";
import { RunInspection } from "./api";

type Action = "cancel" | "resume" | "retry";

type Props = {
  inspection: RunInspection;
  busy: Action | null;
  notice: string | null;
  onCancel: () => Promise<void>;
  onResume: () => Promise<void>;
  onRetry: () => Promise<void>;
};

const terminal = new Set(["completed", "failed", "cancelled", "timed_out"]);

export function RunLifecyclePanel({ inspection, busy, notice, onCancel, onResume, onRetry }: Props) {
  const [confirming, setConfirming] = useState<Action | null>(null);
  const run = inspection.run;
  const active = run.status === "queued" || run.status === "running";
  const canResume = ["failed", "cancelled", "timed_out"].includes(run.status)
    && inspection.checkpoint_available;
  const canRetry = terminal.has(run.status);
  const latency = Object.values(inspection.stage_latency_ms).reduce((sum, item) => sum + item, 0);
  const latestEvents = inspection.events.slice(-4).reverse();

  async function act(action: Action) {
    if ((action === "cancel" || action === "retry") && confirming !== action) {
      setConfirming(action);
      return;
    }
    setConfirming(null);
    if (action === "cancel") await onCancel();
    if (action === "resume") await onResume();
    if (action === "retry") await onRetry();
  }

  return <section className={`run-control-panel ${inspection.failure ? "has-failure" : ""}`}>
    <div className="run-control-heading">
      <div><p className="eyebrow">Run lifecycle</p><h2>Execution controls &amp; diagnostics</h2><p>{run.run_id}{run.retry_of_run_id && <> · retry of {run.retry_of_run_id}</>}</p></div>
      <span className={`run-state-large ${run.status}`}><i />{run.status.replaceAll("_", " ")}</span>
    </div>
    {notice && <div className="run-action-notice">{notice}</div>}
    <div className="run-control-grid">
      <div className="run-metrics">
        <Metric label="Elapsed stages" value={latency ? `${Math.round(latency)} ms` : "—"} />
        <Metric label="Input tokens" value={inspection.input_tokens.toLocaleString()} />
        <Metric label="Output tokens" value={inspection.output_tokens.toLocaleString()} />
        <Metric label="Checkpoint" value={inspection.checkpoint_available ? "Available" : "Unavailable"} />
      </div>
      <div className="run-actions">
        {active && <button className="danger-action" disabled={busy !== null || Boolean(run.cancel_requested_at)} onClick={() => void act("cancel")}>{run.cancel_requested_at ? "Cancellation requested" : confirming === "cancel" ? "Confirm cancellation" : "Cancel Run"}</button>}
        {canResume && <button className="resume-action" disabled={busy !== null} onClick={() => void act("resume")}>{busy === "resume" ? "Queueing resume…" : "Resume checkpoint"}</button>}
        {canRetry && <button className="retry-action" disabled={busy !== null} onClick={() => void act("retry")}>{busy === "retry" ? "Creating retry…" : confirming === "retry" ? "Create a fresh Run" : "Retry as new Run"}</button>}
        {confirming && <button className="cancel-confirm" onClick={() => setConfirming(null)} disabled={busy !== null}>Keep current state</button>}
        {inspection.failure && !canResume && <small>Resume is unavailable because this Run has no durable checkpoint. Retry remains safe.</small>}
        {!active && !canResume && !canRetry && <small>No lifecycle action is available for this state.</small>}
      </div>
    </div>
    {inspection.failure && <div className="failure-diagnostic">
      <div><span className="failure-mark">!</span><div><p className="eyebrow">Safe failure summary</p><h3>{inspection.failure.summary}</h3><p>Reason: <strong>{inspection.failure.stop_reason.replaceAll("_", " ")}</strong>{inspection.failure.last_stage && <> · Last stage: <strong>{inspection.failure.last_stage.replaceAll("_", " ")}</strong></>}</p></div></div>
      <p className="failure-retryability">Manual retry: <strong>{inspection.failure.retryable ? "recommended when ready" : "check configuration first"}</strong>{inspection.failure.attempt_count && <> · Attempts: <strong>{inspection.failure.attempt_count}</strong></>}</p>
      <p className="failure-safety">Provider payloads, stack traces, credentials, and raw exception text are intentionally excluded.</p>
    </div>}
    {latestEvents.length > 0 && <div className="recent-events"><span>Recent durable events</span>{latestEvents.map((event) => <div key={event.event_id ?? `${event.sequence}-${event.event_type}`}><b>#{event.sequence}</b><p>{event.event_type.replaceAll("_", " ")}</p><small>{event.stage?.replaceAll("_", " ") ?? "run"} · {formatTime(event.occurred_at)}</small></div>)}</div>}
  </section>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}

function formatTime(value: string) {
  return new Date(value).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

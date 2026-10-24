import { useState } from "react";
import { RunInspection } from "./api";
import { zhDate, zhLabel } from "./locale";

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
      <div><p className="eyebrow">研究任务生命周期</p><h2>执行控制与诊断</h2><p>{run.run_id}{run.retry_of_run_id && <> · 重试来源 {run.retry_of_run_id}</>}</p></div>
      <span className={`run-state-large ${run.status}`}><i />{zhLabel(run.status)}</span>
    </div>
    {notice && <div className="run-action-notice">{notice}</div>}
    <div className="run-control-grid">
      <div className="run-metrics">
        <Metric label="各阶段累计耗时" value={latency ? `${Math.round(latency)} 毫秒` : "—"} />
        <Metric
          label="输入 Token"
          value={inspection.input_tokens?.toLocaleString("zh-CN") ?? "未提供"}
        />
        <Metric
          label="输出 Token"
          value={inspection.output_tokens?.toLocaleString("zh-CN") ?? "未提供"}
        />
        <Metric label="运行断点" value={inspection.checkpoint_available ? "可用" : "不可用"} />
      </div>
      <div className="run-actions">
        {active && <button className="danger-action" disabled={busy !== null || Boolean(run.cancel_requested_at)} onClick={() => void act("cancel")}>{run.cancel_requested_at ? "已请求取消" : confirming === "cancel" ? "确认取消" : "取消任务"}</button>}
        {canResume && <button className="resume-action" disabled={busy !== null} onClick={() => void act("resume")}>{busy === "resume" ? "正在提交恢复…" : "从断点恢复"}</button>}
        {canRetry && <button className="retry-action" disabled={busy !== null} onClick={() => void act("retry")}>{busy === "retry" ? "正在创建重试…" : confirming === "retry" ? "创建全新任务" : "新建任务并重试"}</button>}
        {confirming && <button className="cancel-confirm" onClick={() => setConfirming(null)} disabled={busy !== null}>保持当前状态</button>}
        {inspection.failure && !canResume && <small>此任务没有持久化断点，因此无法恢复；仍可安全地新建任务重试。</small>}
        {!active && !canResume && !canRetry && <small>当前状态没有可用的生命周期操作。</small>}
      </div>
    </div>
    {inspection.failure && <div className="failure-diagnostic">
      <div><span className="failure-mark">!</span><div><p className="eyebrow">安全错误摘要</p><h3>{inspection.failure.summary}</h3><p>原因：<strong>{zhLabel(inspection.failure.stop_reason)}</strong>{inspection.failure.last_stage && <> · 最后阶段：<strong>{zhLabel(inspection.failure.last_stage)}</strong></>}</p></div></div>
      <p className="failure-retryability">手动重试：<strong>{inspection.failure.retryable ? "准备好后建议重试" : "请先检查配置"}</strong>{inspection.failure.attempt_count && <> · 尝试次数：<strong>{inspection.failure.attempt_count}</strong></>}</p>
      <p className="failure-safety">出于安全考虑，此处不会展示供应商原始响应、堆栈、凭证或未经处理的异常文本。</p>
    </div>}
    {latestEvents.length > 0 && <div className="recent-events"><span>最近的持久化事件</span>{latestEvents.map((event) => <div key={event.event_id ?? `${event.sequence}-${event.event_type}`}><b>#{event.sequence}</b><p>{zhLabel(event.event_type)}</p><small>{zhLabel(event.stage ?? "run")} · {formatTime(event.occurred_at)}</small></div>)}</div>}
  </section>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}

function formatTime(value: string) {
  return zhDate(value, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

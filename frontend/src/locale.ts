const labels: Record<string, string> = {
  active: "活跃",
  archived: "已归档",
  queued: "排队中",
  running: "执行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  timed_out: "已超时",
  pending: "待处理",
  presented: "已呈现",
  skipped: "已跳过",
  proposed: "已提议",
  approved: "已批准",
  rejected: "已拒绝",
  superseded: "已取代",
  conflicted: "存在冲突",
  invalidated: "已失效",
  available: "可用",
  unavailable: "不可用",
  enabled: "已启用",
  disabled: "已禁用",
  member: "成员",
  reviewer: "审核员",
  admin: "管理员",
  direct_conversation: "直接对话",
  research_result: "研究结果",
  historical_result_readback: "历史结果回读",
  start_research: "发起规范研究",
  get_research_status: "查询研究状态",
  get_research_result: "获取研究结果",
  read_research_result: "回读研究结果",
  executing: "执行中",
  run: "研究任务",
  router: "问题路由",
  planner: "研究规划",
  researcher_worker: "证据检索",
  source_validator: "来源验证",
  sufficiency_judge: "充分性判断",
  replanner: "补充规划",
  synthesizer: "答案综合",
  run_started: "研究已开始",
  run_stopped: "研究已结束",
};

export function zhLabel(value: string | null | undefined): string {
  if (!value) return "—";
  return labels[value] ?? value.replaceAll("_", " ");
}

export function zhDate(value: string | Date, options?: Intl.DateTimeFormatOptions): string {
  return new Date(value).toLocaleString("zh-CN", options);
}

export function zhDateOnly(value: string | Date, options?: Intl.DateTimeFormatOptions): string {
  return new Date(value).toLocaleDateString("zh-CN", options);
}

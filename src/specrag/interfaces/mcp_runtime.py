"""Thread-safe, redacted readiness state for the long-running MCP process."""

from __future__ import annotations

from threading import Lock
from time import perf_counter

from specrag.models import MCPRuntimeReadiness, MCPRuntimeStatus, MCPWarmupStage


class MCPRuntimeState:
    def __init__(self, release: int = 17) -> None:
        self.release = release
        self._lock = Lock()
        self._status: MCPRuntimeStatus = "starting"
        self._message = "The research runtime is starting."
        self._stages: list[MCPWarmupStage] = []
        self._started = perf_counter()

    def begin_warmup(self) -> None:
        with self._lock:
            self._status = "warming"
            self._message = "The Release 17 research runtime is warming."
            self._stages = []
            self._started = perf_counter()

    def record_stage(self, name: str, duration_ms: float) -> None:
        stage = MCPWarmupStage(name=name, duration_ms=max(0, duration_ms))
        with self._lock:
            self._stages.append(stage)

    def mark_ready(self) -> None:
        with self._lock:
            self._status = "ready"
            self._message = "The Release 17 research runtime is ready."

    def mark_failed(self) -> None:
        with self._lock:
            self._status = "failed"
            self._message = (
                "The research runtime failed to warm. Check the MCP service logs."
            )

    def snapshot(self) -> MCPRuntimeReadiness:
        with self._lock:
            stages = tuple(self._stages)
            status = self._status
            message = self._message
            total_ms = (
                sum(item.duration_ms for item in stages)
                if status in {"ready", "failed"}
                else (perf_counter() - self._started) * 1000
            )
        return MCPRuntimeReadiness(
            status=status,
            release=self.release,
            message=message,
            stages=stages,
            total_ms=max(0, total_ms),
        )

    def require_ready(self) -> None:
        snapshot = self.snapshot()
        if snapshot.status != "ready":
            raise RuntimeError(
                f"research runtime unavailable: {snapshot.status}"
            )


__all__ = ["MCPRuntimeState"]

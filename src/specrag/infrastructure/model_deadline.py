"""Deadline-aware provider request and retry timing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .api_errors import StructuredLLMError


@dataclass(frozen=True)
class ModelCallDeadline:
    timeout_seconds: float
    deadline_at: datetime | None
    api_mode: str

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.deadline_at is not None and self.deadline_at.tzinfo is None:
            raise ValueError("deadline_at must be timezone-aware")

    def request_timeout(self) -> float:
        if self.deadline_at is None:
            return self.timeout_seconds
        remaining = (self.deadline_at - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            self._raise_expired("structured LLM request deadline expired")
        return min(self.timeout_seconds, remaining)

    def sleep_before_retry(self, delay: float) -> None:
        if not delay:
            return
        if self.deadline_at is not None:
            remaining = (
                self.deadline_at - datetime.now(timezone.utc)
            ).total_seconds()
            if remaining <= delay:
                self._raise_expired("structured LLM retry would exceed its deadline")
        time.sleep(delay)

    def _raise_expired(self, message: str) -> None:
        raise StructuredLLMError(
            message,
            failure_kind="timeout",
            retryable=True,
            public_message="The research model deadline was reached.",
            api_mode=self.api_mode,
        )


__all__ = ["ModelCallDeadline"]

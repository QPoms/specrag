from __future__ import annotations

from specrag.interfaces.conversation_runtime import ApplicationWorkers


class _Worker:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1


def test_conversation_only_worker_mode_does_not_start_research_workers() -> None:
    run = _Worker()
    lifecycle = _Worker()
    conversation = _Worker()
    runtime = ApplicationWorkers(
        run=run,  # type: ignore[arg-type]
        lifecycle=lifecycle,  # type: ignore[arg-type]
        conversation=conversation,  # type: ignore[arg-type]
        owned=(True, True, True),
    )

    runtime.start(research=False, lifecycle=False, conversation=True)
    runtime.stop(research=False, lifecycle=False, conversation=True)

    assert (run.starts, run.stops) == (0, 0)
    assert (lifecycle.starts, lifecycle.stops) == (0, 0)
    assert (conversation.starts, conversation.stops) == (1, 1)

"""Standalone durable Run Worker entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from specrag.infrastructure import DEFAULT_DATABASE_PATH
from specrag.interfaces.workspace_services import WorkspaceServices
from specrag.orchestration import ResearchLifecycleWorker, RunWorker
from specrag.interfaces.conversation_runtime import (
    DELEGATION_SECRET_ENV,
    build_conversation_worker,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="specrag-worker")
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="SQLite state database shared with specrag-api",
    )
    arguments = parser.parse_args()
    services = WorkspaceServices.create(arguments.database)
    worker = RunWorker(services.harness, services.run_queue)
    lifecycle = ResearchLifecycleWorker(
        services.runs,
        services.research_results,
        services.conversation_lifecycle_store,
        services.completion_queue,
    )
    conversation = (
        build_conversation_worker(services)
        if os.environ.get(DELEGATION_SECRET_ENV)
        else None
    )
    try:
        lifecycle.start()
        if conversation is not None:
            conversation.start()
        worker.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        lifecycle.stop()
        if conversation is not None:
            conversation.stop()
        services.close()


__all__ = ["main"]


if __name__ == "__main__":
    main()

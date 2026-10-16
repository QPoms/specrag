"""Standalone durable Run Worker entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from specrag.infrastructure import DEFAULT_DATABASE_PATH
from specrag.interfaces.workspace_services import WorkspaceServices
from specrag.orchestration import RunWorker


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
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        services.close()


__all__ = ["main"]


if __name__ == "__main__":
    main()

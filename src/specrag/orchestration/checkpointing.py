"""Durable LangGraph checkpoint construction."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from .workflow import _CHECKPOINT_MODEL_ALLOWLIST


class SQLiteCheckpointManager:
    """Own the connection used by one SQLite LangGraph saver."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode = WAL")
        serializer = JsonPlusSerializer(
            allowed_msgpack_modules=_CHECKPOINT_MODEL_ALLOWLIST,
        )
        self.saver = SqliteSaver(self.connection, serde=serializer)

    def close(self) -> None:
        self.connection.close()


__all__ = ["SQLiteCheckpointManager"]

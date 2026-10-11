"""Persistence adapters for validated memory records."""

from __future__ import annotations

import json
from pathlib import Path

from specrag.models import ValidatedKnowledge


class InMemoryKnowledgeStore:
    def __init__(self, initial: tuple[ValidatedKnowledge, ...] = ()) -> None:
        self._records = {item.knowledge_id: item for item in initial}

    def all(self) -> tuple[ValidatedKnowledge, ...]:
        return tuple(
            self._records[key] for key in sorted(self._records)
        )

    def put(self, knowledge: ValidatedKnowledge) -> None:
        self._records[knowledge.knowledge_id] = knowledge


class JsonlKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.is_file():
            records = tuple(
                ValidatedKnowledge.model_validate_json(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        else:
            records = ()
        super().__init__(records)

    def put(self, knowledge: ValidatedKnowledge) -> None:
        super().put(knowledge)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "".join(
                json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
                + "\n"
                for item in self.all()
            ),
            encoding="utf-8",
        )


__all__ = ["InMemoryKnowledgeStore", "JsonlKnowledgeStore"]

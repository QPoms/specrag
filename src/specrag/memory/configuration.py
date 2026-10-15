"""Resolve presentation and Release settings with explicit precedence."""

from __future__ import annotations

from dataclasses import dataclass

from specrag.models import MemoryEntry


@dataclass(frozen=True)
class ResolvedUserConfiguration:
    release: int
    answer_language: str
    detail_level: str
    memory_ids: tuple[str, ...] = ()


def resolve_user_configuration(
    memories: tuple[MemoryEntry, ...],
    *,
    config_release: int,
    explicit_release: int | None = None,
    explicit_language: str | None = None,
    explicit_detail: str | None = None,
) -> ResolvedUserConfiguration:
    values = {
        item.kind: item
        for item in memories
        if item.status == "active" and item.key.casefold() == "default"
    }
    release_item = values.get("default_release")
    language_item = values.get("answer_language")
    detail_item = values.get("detail_level")
    memory_release = int(release_item.content) if release_item else None
    used = tuple(sorted(
        item.memory_id for item, explicit in (
            (release_item, explicit_release),
            (language_item, explicit_language),
            (detail_item, explicit_detail),
        ) if item is not None and explicit is None
    ))
    return ResolvedUserConfiguration(
        release=explicit_release or memory_release or config_release,
        answer_language=(
            explicit_language or (language_item.content if language_item else None) or "English"
        ),
        detail_level=(
            explicit_detail or (detail_item.content if detail_item else None) or "standard"
        ),
        memory_ids=used,
    )


__all__ = ["ResolvedUserConfiguration", "resolve_user_configuration"]

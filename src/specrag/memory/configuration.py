"""Resolve presentation and Release settings with explicit precedence."""

from __future__ import annotations

from dataclasses import dataclass

from specrag.models import UserMemoryItem


@dataclass(frozen=True)
class ResolvedUserConfiguration:
    release: int
    answer_language: str
    detail_level: str


def resolve_user_configuration(
    memories: tuple[UserMemoryItem, ...],
    *,
    config_release: int,
    explicit_release: int | None = None,
    explicit_language: str | None = None,
    explicit_detail: str | None = None,
) -> ResolvedUserConfiguration:
    values = {item.kind: item.value for item in memories if item.status == "active"}
    release_text = values.get("default_release")
    memory_release = int(release_text) if release_text else None
    return ResolvedUserConfiguration(
        release=explicit_release or memory_release or config_release,
        answer_language=(
            explicit_language or values.get("answer_language") or "English"
        ),
        detail_level=explicit_detail or values.get("detail_level") or "standard",
    )


__all__ = ["ResolvedUserConfiguration", "resolve_user_configuration"]

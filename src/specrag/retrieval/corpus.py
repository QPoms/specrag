"""3GPP Markdown discovery, version selection, and source-faithful chunking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable

from ..config import RetrievalCfg
from .models import (
    CorpusScope,
    MarkdownChunk,
    SourceDocument,
)


ROOT = Path(__file__).resolve().parents[3]
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SPECIFICATION = re.compile(r"^\d{2}\.\d{3}$")
_VERSION = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_MARKDOWN_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+.*$")


@dataclass(frozen=True)
class MarkdownSection:
    """One Markdown heading and its direct (non-descendant) source lines."""

    heading_path: tuple[str, ...]
    heading_level: int
    first_line: int
    lines: tuple[str, ...]


def repository_path(raw_path: str, label: str) -> Path:
    path = (ROOT / raw_path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository: {raw_path}") from exc
    return path


def relative_repository_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _version_key(path: Path) -> tuple[int, int, int]:
    match = _VERSION.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid specification version directory: {path}")
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def _select_version_source(
    version_sources: list[tuple[Path, Path]],
    scope: CorpusScope,
    specification: str,
) -> tuple[Path, Path] | None:
    if scope.version_policy == "latest_per_spec":
        eligible = version_sources
    elif scope.version_policy == "highest_version_within_release":
        eligible = [
            item
            for item in version_sources
            if _version_key(item[0])[0] == scope.release
        ]
    else:
        requested = {
            item.specification: item.version for item in scope.versions
        }.get(specification)
        eligible = [
            item
            for item in version_sources
            if requested is not None and item[0].name == f"v{requested}"
        ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: _version_key(item[0]))


def corpus_scope_from_config(cfg: RetrievalCfg) -> CorpusScope:
    return CorpusScope(
        version_policy=cfg.version_policy,
        release=cfg.target_release,
    )


def discover_source_documents(
    cfg: RetrievalCfg,
    scope: CorpusScope | None = None,
) -> tuple[SourceDocument, ...]:
    """Resolve the immutable source snapshot selected by an explicit scope."""

    active_scope = scope or corpus_scope_from_config(cfg)
    documents: list[SourceDocument] = []
    spec_root = repository_path(cfg.specification_root, "specification root")
    if not spec_root.is_dir():
        raise FileNotFoundError(f"3GPP Markdown corpus not found: {spec_root}")
    for spec_dir in sorted(path for path in spec_root.iterdir() if path.is_dir()):
        if _SPECIFICATION.fullmatch(spec_dir.name) is None:
            continue
        version_sources: list[tuple[Path, Path]] = []
        for version_dir in spec_dir.iterdir():
            if not version_dir.is_dir() or _VERSION.fullmatch(version_dir.name) is None:
                continue
            markdown = sorted(version_dir.glob("*.md"))
            if len(markdown) > 1:
                raise ValueError(
                    f"expected at most one Markdown file in {version_dir}, "
                    f"found {len(markdown)}"
                )
            if markdown:
                version_sources.append((version_dir, markdown[0]))
        if not version_sources:
            continue
        selected = _select_version_source(
            version_sources,
            active_scope,
            spec_dir.name,
        )
        if selected is None:
            continue
        version_dir, source = selected
        documents.append(SourceDocument(
            source_collection="3gpp_markdown",
            specification=spec_dir.name,
            version=version_dir.name[1:],
            source_path=relative_repository_path(source),
            document_sha256=hashlib.sha256(source.read_bytes()).hexdigest().upper(),
        ))
    if active_scope.versions:
        requested = {
            (item.specification, item.version)
            for item in active_scope.versions
        }
        selected = {
            (item.specification, item.version)
            for item in documents
        }
        missing = sorted(requested - selected)
        if missing:
            raise ValueError(
                "requested specification versions are unavailable: "
                + ", ".join(
                    f"{specification} v{version}"
                    for specification, version in missing
                )
            )
    if not documents:
        release_detail = (
            f" for Release {active_scope.release}"
            if active_scope.release is not None
            else ""
        )
        raise ValueError("the 3GPP Markdown source snapshot is empty" + release_detail)
    return tuple(documents)


def discover_all_source_documents(
    cfg: RetrievalCfg,
    *,
    release: int | None = None,
) -> tuple[SourceDocument, ...]:
    """Discover every locally retained version without selecting an active one.

    This is intentionally separate from :func:`discover_source_documents`:
    retrieval scopes select one immutable snapshot, while synchronization must
    see historical and newly added versions together.
    """

    documents: list[SourceDocument] = []
    spec_root = repository_path(cfg.specification_root, "specification root")
    if not spec_root.is_dir():
        raise FileNotFoundError(f"3GPP Markdown corpus not found: {spec_root}")
    for spec_dir in sorted(path for path in spec_root.iterdir() if path.is_dir()):
        if _SPECIFICATION.fullmatch(spec_dir.name) is None:
            continue
        version_dirs = sorted(
            (
                path for path in spec_dir.iterdir()
                if path.is_dir() and _VERSION.fullmatch(path.name) is not None
            ),
            key=_version_key,
        )
        for version_dir in version_dirs:
            if release is not None and _version_key(version_dir)[0] != release:
                continue
            markdown = sorted(version_dir.glob("*.md"))
            if len(markdown) > 1:
                raise ValueError(
                    f"expected at most one Markdown file in {version_dir}, "
                    f"found {len(markdown)}"
                )
            if not markdown:
                continue
            source = markdown[0]
            documents.append(SourceDocument(
                source_collection="3gpp_markdown",
                specification=spec_dir.name,
                version=version_dir.name[1:],
                source_path=relative_repository_path(source),
                document_sha256=hashlib.sha256(source.read_bytes()).hexdigest().upper(),
            ))
    if not documents:
        detail = f" for Release {release}" if release is not None else ""
        raise ValueError("the 3GPP Markdown source snapshot is empty" + detail)
    return tuple(documents)


def parse_markdown_sections(
    lines: tuple[str, ...], *, skip_preamble: bool
) -> tuple[MarkdownSection, ...]:
    starts = [index for index, line in enumerate(lines) if _HEADING.match(line)]
    if not starts:
        return (MarkdownSection((), 0, 0, lines),) if lines else ()
    sections: list[MarkdownSection] = []
    hierarchy: list[str] = []
    if not skip_preamble and starts[0] > 0:
        sections.append(MarkdownSection((), 0, 0, lines[:starts[0]]))
    for position, start in enumerate(starts):
        match = _HEADING.match(lines[start])
        assert match is not None
        level = len(match.group(1))
        hierarchy = hierarchy[: level - 1]
        hierarchy.append(match.group(2).strip())
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        sections.append(MarkdownSection(
            tuple(hierarchy), level, start, lines[start:end]
        ))
    return tuple(sections)


def _window_section(
    section: MarkdownSection,
    *,
    max_characters: int,
    overlap_characters: int,
) -> Iterable[tuple[int, int, int | None, int | None, str]]:
    """Split on source lines and preserve exact inclusive line bounds."""

    lines = section.lines
    start = 0
    while start < len(lines):
        if len(lines[start]) > max_characters:
            source_line = lines[start]
            offset = 0
            stride = max_characters - overlap_characters
            while offset < len(source_line):
                character_end = min(len(source_line), offset + max_characters)
                text = source_line[offset:character_end]
                if text.strip():
                    source_line_number = section.first_line + start + 1
                    yield (
                        source_line_number,
                        source_line_number,
                        offset,
                        character_end,
                        text,
                    )
                if character_end == len(source_line):
                    break
                offset += stride
            start += 1
            continue
        end = start
        size = 0
        while end < len(lines):
            added = len(lines[end]) + (1 if end > start else 0)
            if end > start and size + added > max_characters:
                break
            size += added
            end += 1
            if size >= max_characters:
                break
        if end == start:
            end += 1
        text = "\n".join(lines[start:end])
        if text.strip():
            yield (
                section.first_line + start + 1,
                section.first_line + end,
                None,
                None,
                text,
            )
        if end >= len(lines):
            break
        retained = 0
        next_start = end
        while next_start > start and retained < overlap_characters:
            next_start -= 1
            retained += len(lines[next_start]) + 1
        start = max(start + 1, next_start)


def chunk_source_documents(
    documents: tuple[SourceDocument, ...],
    cfg: RetrievalCfg,
) -> tuple[MarkdownChunk, ...]:
    chunks: list[MarkdownChunk] = []
    for document in documents:
        source = ROOT / document.source_path
        lines = tuple(source.read_text(encoding="utf-8").splitlines())
        sections = parse_markdown_sections(
            lines,
            skip_preamble=document.source_collection == "3gpp_markdown",
        )
        for section in sections:
            for segment, (
                line_start,
                line_end,
                character_start,
                character_end,
                text,
            ) in enumerate(
                _window_section(
                    section,
                    max_characters=cfg.chunk_characters,
                    overlap_characters=cfg.chunk_overlap_characters,
                ),
                start=1,
            ):
                identity = "|".join((
                    document.source_path,
                    document.document_sha256,
                    str(line_start),
                    str(line_end),
                    str(segment),
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                ))
                chunks.append(MarkdownChunk(
                    chunk_id="CH-" + hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest()[:24].upper(),
                    source_collection=document.source_collection,
                    specification=document.specification,
                    version=document.version,
                    source_path=document.source_path,
                    document_sha256=document.document_sha256,
                    heading_path=section.heading_path,
                    line_start=line_start,
                    line_end=line_end,
                    character_start=character_start,
                    character_end=character_end,
                    text=text,
                ))
    return tuple(chunks)


def evidence_body_text(text: str) -> str:
    """Return evidentiary body text without Markdown section headings."""

    return "\n".join(
        line
        for line in text.splitlines()
        if not _MARKDOWN_HEADING_LINE.match(line)
    ).strip()


__all__ = [
    "chunk_source_documents",
    "corpus_scope_from_config",
    "discover_all_source_documents",
    "discover_source_documents",
    "evidence_body_text",
    "MarkdownSection",
    "parse_markdown_sections",
    "relative_repository_path",
    "repository_path",
]

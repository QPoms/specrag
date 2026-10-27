"""Deterministic structural index for 3GPP Markdown clauses."""

from __future__ import annotations

from collections import Counter, defaultdict
import re

from .corpus import parse_markdown_sections, repository_path
from .models import ClauseRecord, ClauseReference, MarkdownChunk, SourceDocument


_FINAL_SUFFIX = r"[A-Za-z](?=\s|\d|[A-Z]|[a-z]+[A-Z]|$)"
_NUMERIC_SEGMENT = r"\d+(?:[A-Za-z](?=\.))?"
_NUMERIC_ID = (
    rf"{_NUMERIC_SEGMENT}(?:\.{_NUMERIC_SEGMENT})*"
    rf"(?:{_FINAL_SUFFIX})?(?:\.{_FINAL_SUFFIX})?"
)
_ANNEX_ID = rf"[A-Z](?:\.{_NUMERIC_SEGMENT})+(?:{_FINAL_SUFFIX})?"
_CLAUSE_HEADING = re.compile(
    rf"^(?P<clause>{_ANNEX_ID}|{_NUMERIC_ID})(?P<title>.*)$"
)
_CLAUSE_REFERENCE = re.compile(
    rf"(?i)\bclauses?\s+(?P<clause>{_ANNEX_ID}|{_NUMERIC_ID})"
)
_SPECIFICATION_REFERENCE = re.compile(r"(?i)\b(?:3GPP\s+)?TS\s+(\d{2}\.\d{3})\b")
_VERSION_REFERENCE = re.compile(r"(?i)\bV(\d+\.\d+\.\d+)\b")
_BIBLIOGRAPHY_REFERENCE = re.compile(
    r"^\s*\[(?P<label>\d+)\].*?\b(?:3GPP\s+)?TS\s+"
    r"(?P<specification>\d{2}\.\d{3})\b",
    re.IGNORECASE,
)
_REFERENCE_LABEL = re.compile(r"\[(\d+)\]")


def split_clause_heading(heading: str) -> tuple[str, str] | None:
    """Split a converted 3GPP heading into its stable clause ID and title."""

    match = _CLAUSE_HEADING.fullmatch(heading.strip())
    if match is None:
        return None
    clause_id = match.group("clause")
    title = match.group("title").strip(" \t:-")
    # Converted headings contain no delimiter. Do not steal the first letter
    # from an acronym title such as ``5GMM`` or ``2APN`` as an uppercase
    # clause suffix. Mixed-case titles such as ``1AGeneral`` retain the suffix.
    if (
        clause_id[-1:].isupper()
        and len(title) >= 2
        and title[0].isupper()
        and (title[1].isupper() or title[1] in "-_/")
    ):
        title = clause_id[-1] + title
        clause_id = clause_id[:-1]
    return clause_id, title


def _disambiguate_duplicate_ids(
    values: list[tuple[int, str, str, str]],
) -> list[tuple[int, str, str, str]]:
    """Resolve converter-joined IDs only when the initial parse collides.

    3GPP headings omit the separator between an identifier and title. A title
    beginning with ``5G`` can therefore make clause ``...3`` look like
    ``...35``. Lettered subclauses have the inverse ambiguity (``...3a1``).
    A collision supplies deterministic evidence for trying the alternate
    boundary; otherwise the conservative initial parse is retained.
    """

    counts = Counter(clause_id.casefold() for _, clause_id, _, _ in values)
    resolved: list[tuple[int, str, str, str]] = []
    for section_index, clause_id, title, heading in values:
        key = clause_id.casefold()
        if counts[key] < 2:
            resolved.append((section_index, clause_id, title, heading))
            continue
        alternatives: list[str] = []
        leading_digits = re.match(r"\d+", title)
        if clause_id[-1:].isalpha() and leading_digits is not None:
            alternatives.append(clause_id + leading_digits.group(0))
        if re.search(r"\d[A-Za-z]$", clause_id):
            alternatives.append(clause_id[:-1])
        trailing_digits = re.search(r"\d{2,}$", clause_id)
        if trailing_digits is not None:
            alternatives.append(clause_id[:-1])
        replacement = next(
            (
                candidate
                for candidate in alternatives
                if counts[candidate.casefold()] == 0
            ),
            None,
        )
        if replacement is None:
            resolved.append((section_index, clause_id, title, heading))
            continue
        counts[key] -= 1
        counts[replacement.casefold()] += 1
        resolved.append((
            section_index,
            replacement,
            heading[len(replacement):].strip(" \t:-"),
            heading,
        ))
    return resolved


def _reference_catalog(lines: tuple[str, ...]) -> dict[str, str]:
    catalog: dict[str, str] = {}
    for line in lines:
        match = _BIBLIOGRAPHY_REFERENCE.search(line)
        if match is not None:
            catalog[match.group("label")] = match.group("specification")
    return catalog


def _references(
    lines: tuple[str, ...],
    *,
    first_line: int,
    source_specification: str,
    catalog: dict[str, str],
) -> tuple[ClauseReference, ...]:
    found: list[ClauseReference] = []
    seen: set[tuple[int, str, str | None, str | None, str | None]] = set()
    for offset, line in enumerate(lines):
        line_number = first_line + offset
        for match in _CLAUSE_REFERENCE.finditer(line):
            vicinity = line[max(0, match.start() - 100):match.end() + 100]
            explicit_specs = set(_SPECIFICATION_REFERENCE.findall(vicinity))
            versions = set(_VERSION_REFERENCE.findall(vicinity))
            labels = set(_REFERENCE_LABEL.findall(vicinity))
            if len(explicit_specs) > 1 or len(versions) > 1 or len(labels) > 1:
                continue
            explicit = next(iter(explicit_specs), None)
            target_version = next(iter(versions), None)
            label = next(iter(labels), None)
            target_specification = (
                explicit
                if explicit is not None
                else catalog.get(label or "", source_specification)
            )
            key = (
                line_number,
                match.group("clause").casefold(),
                target_specification,
                target_version,
                label,
            )
            if key in seen:
                continue
            seen.add(key)
            found.append(ClauseReference(
                raw_text=match.group(0),
                line_number=line_number,
                target_clause_id=match.group("clause"),
                target_specification=target_specification,
                target_version=target_version,
                reference_label=label,
            ))
    return tuple(found)


def build_clause_records(
    documents: tuple[SourceDocument, ...],
    chunks: tuple[MarkdownChunk, ...],
) -> tuple[ClauseRecord, ...]:
    """Build source-ordered clause records without model or vector calls."""

    chunks_by_path: dict[str, list[MarkdownChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_path[chunk.source_path].append(chunk)

    records: list[ClauseRecord] = []
    for document in documents:
        source = repository_path(document.source_path, "clause source")
        lines = tuple(source.read_text(encoding="utf-8").splitlines())
        sections = parse_markdown_sections(lines, skip_preamble=True)
        parsed: list[tuple[int, str, str, str]] = []
        for index, section in enumerate(sections):
            if not section.heading_path:
                continue
            identity = split_clause_heading(section.heading_path[-1])
            if identity is not None:
                parsed.append((index, *identity, section.heading_path[-1]))
        parsed = _disambiguate_duplicate_ids(parsed)

        catalog = _reference_catalog(lines)
        provisional: list[dict[str, object]] = []
        path_to_clause: dict[tuple[str, ...], str] = {}
        for section_index, clause_id, title, _heading in parsed:
            section = sections[section_index]
            next_section_start = (
                sections[section_index + 1].first_line
                if section_index + 1 < len(sections)
                else len(lines)
            )
            direct_line_end = max(section.first_line + 1, next_section_start)
            descendant_end = len(lines)
            for later in sections[section_index + 1:]:
                if later.heading_level <= section.heading_level:
                    descendant_end = later.first_line
                    break
            parent_clause_id = None
            for depth in range(len(section.heading_path) - 1, 0, -1):
                parent_clause_id = path_to_clause.get(section.heading_path[:depth])
                if parent_clause_id is not None:
                    break
            document_chunks = chunks_by_path[document.source_path]
            direct_chunks = tuple(
                chunk.chunk_id
                for chunk in document_chunks
                if chunk.heading_path == section.heading_path
            )
            descendant_chunks = tuple(
                chunk.chunk_id
                for chunk in document_chunks
                if chunk.heading_path[:len(section.heading_path)]
                == section.heading_path
            )
            provisional.append({
                "specification": document.specification,
                "version": document.version,
                "release": int(document.version.split(".", 1)[0]),
                "source_path": document.source_path,
                "document_sha256": document.document_sha256,
                "clause_id": clause_id,
                "title": title,
                "heading_path": section.heading_path,
                "heading_level": section.heading_level,
                "line_start": section.first_line + 1,
                "direct_line_end": direct_line_end,
                "line_end": descendant_end,
                "parent_clause_id": parent_clause_id,
                "direct_chunk_ids": direct_chunks,
                "chunk_ids": descendant_chunks,
                "cross_references": _references(
                    lines[section.first_line:next_section_start],
                    first_line=section.first_line + 1,
                    source_specification=document.specification,
                    catalog=catalog,
                ),
            })
            path_to_clause[section.heading_path] = clause_id

        merged: dict[str, dict[str, object]] = {}
        for item in provisional:
            clause_id = str(item["clause_id"])
            existing = merged.get(clause_id)
            if existing is None:
                merged[clause_id] = item
                continue
            if existing["parent_clause_id"] != item["parent_clause_id"]:
                raise ValueError(
                    f"duplicate clause ID has different parents: {clause_id}"
                )
            titles = tuple(dict.fromkeys((
                str(existing["title"]), str(item["title"]),
            )))
            existing.update({
                "title": " / ".join(value for value in titles if value),
                "line_start": min(
                    int(existing["line_start"]), int(item["line_start"])
                ),
                "direct_line_end": max(
                    int(existing["direct_line_end"]),
                    int(item["direct_line_end"]),
                ),
                "line_end": max(int(existing["line_end"]), int(item["line_end"])),
                "direct_chunk_ids": tuple(dict.fromkeys((
                    *existing["direct_chunk_ids"], *item["direct_chunk_ids"],
                ))),
                "chunk_ids": tuple(dict.fromkeys((
                    *existing["chunk_ids"], *item["chunk_ids"],
                ))),
                "cross_references": tuple(dict.fromkeys((
                    *existing["cross_references"], *item["cross_references"],
                ))),
            })

        children: dict[str, list[str]] = defaultdict(list)
        for item in merged.values():
            parent = item["parent_clause_id"]
            if isinstance(parent, str):
                children[parent].append(str(item["clause_id"]))
        for item in merged.values():
            item["child_clause_ids"] = tuple(dict.fromkeys(
                children[str(item["clause_id"])]
            ))
            records.append(ClauseRecord.model_validate(item))
    return tuple(records)


class ClauseIndex:
    """Read-only lookup and navigation over a persisted clause snapshot."""

    def __init__(
        self,
        records: tuple[ClauseRecord, ...],
        chunks: tuple[MarkdownChunk, ...],
    ) -> None:
        self.records = records
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._by_key = {
            (record.specification, record.version, record.clause_id.casefold()): record
            for record in records
        }
        if len(self._by_key) != len(records):
            raise ValueError(
                "clause records contain duplicate specification/version IDs"
            )
        self._owner_by_chunk = {
            chunk_id: record
            for record in records
            for chunk_id in record.direct_chunk_ids
        }

    def lookup(
        self,
        query: str,
        *,
        specifications: tuple[str, ...] = (),
        limit: int = 10,
    ) -> tuple[ClauseRecord, ...]:
        if not query.strip():
            raise ValueError("clause query must not be blank")
        if limit < 1:
            raise ValueError("limit must be positive")
        allowed = frozenset(specifications)
        folded = re.sub(r"\s+", "", query).casefold()
        matches: list[ClauseRecord] = []
        for record in self.records:
            if allowed and record.specification not in allowed:
                continue
            clause = record.clause_id.casefold()
            title = re.sub(r"\s+", "", record.title).casefold()
            combined = clause + title
            if folded in {clause, title, combined}:
                matches.append(record)
        return tuple(matches[:limit])

    def get(
        self, specification: str, version: str, clause_id: str
    ) -> ClauseRecord | None:
        return self._by_key.get((specification, version, clause_id.casefold()))

    def owner(self, chunk_id: str) -> ClauseRecord | None:
        return self._owner_by_chunk.get(chunk_id)

    def chunks_for(
        self, record: ClauseRecord, *, include_descendants: bool = False
    ) -> tuple[MarkdownChunk, ...]:
        chunk_ids = record.chunk_ids if include_descendants else record.direct_chunk_ids
        return tuple(self._chunks[item] for item in chunk_ids if item in self._chunks)

    def related(
        self,
        record: ClauseRecord,
        relation: str,
    ) -> tuple[ClauseRecord, ...]:
        """Resolve a structural edge inside the active immutable corpus scope."""

        keys: list[tuple[str, str, str]] = []
        if relation == "parent" and record.parent_clause_id is not None:
            keys.append((
                record.specification,
                record.version,
                record.parent_clause_id.casefold(),
            ))
        elif relation == "children":
            keys.extend(
                (record.specification, record.version, item.casefold())
                for item in record.child_clause_ids
            )
        elif relation == "references":
            for reference in record.cross_references:
                target_specification = (
                    reference.target_specification or record.specification
                )
                candidates = (
                    item
                    for item in self.records
                    if item.specification == target_specification
                    and item.clause_id.casefold()
                    == reference.target_clause_id.casefold()
                )
                target = next(candidates, None)
                if target is not None:
                    keys.append((
                        target.specification,
                        target.version,
                        target.clause_id.casefold(),
                    ))
        elif relation not in {"parent", "children", "references"}:
            raise ValueError(f"unknown clause relation: {relation}")
        return tuple(dict.fromkeys(
            target
            for key in keys
            if (target := self._by_key.get(key)) is not None
        ))

    def resolve_reference(
        self,
        source: ClauseRecord,
        *,
        target_specification: str | None,
        target_clause_id: str,
        target_version: str | None = None,
    ) -> ClauseRecord | None:
        """Resolve only an edge explicitly parsed from the source clause."""

        requested_specification = target_specification or source.specification
        matching_edges = tuple(
            reference
            for reference in source.cross_references
            if (reference.target_specification or source.specification)
            == requested_specification
            and reference.target_clause_id.casefold()
            == target_clause_id.casefold()
            and reference.target_version == target_version
        )
        if not matching_edges:
            return None
        candidates = tuple(
            record
            for record in self.records
            if record.specification == requested_specification
            and record.clause_id.casefold() == target_clause_id.casefold()
            and (target_version is None or record.version == target_version)
        )
        return candidates[0] if len(candidates) == 1 else None


__all__ = ["ClauseIndex", "build_clause_records", "split_clause_heading"]

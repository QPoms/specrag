"""Corpus-grounded relation discovery for retrieval expansion."""

from __future__ import annotations

import re
from typing import Any

from .execution import (
    exact_match_priority as _exact_match_priority,
    fold_protocol_text as _fold_protocol_text,
)
from .execution_models import GroundedRelationEvidence
from .hybrid import HybridMarkdownRetriever
from .index import searchable_text
from .models import MarkdownChunk
from .normalization import protocol_identifier_in_text
from .plan_models import DiscoveryRetrievalPlanV3, GroundedTerm
from .planning import GENERIC_FACET_TERMS
_INTENT_QUERY_PHRASES: dict[str, str] = {
    "definition": "definition and normative meaning",
    "representation": "normative representation",
    "field_structure": "fields and information elements",
    "encoding": "encoding and value construction",
    "carrier_usage": "messages or containers that carry the term",
    "procedure_usage": "procedure usage and message occurrence",
    "transport_path": "transport path and encapsulation",
    "endpoint_interface": "endpoints and interfaces",
    "direction": "protocol direction",
    "authority": "authorized entity or role",
    "security_protection": "security protection conditions",
    "state_condition": "protocol state and conditions",
    "timing_order": "timing and ordering",
    "prerequisites": "required inputs and prerequisites",
    "relationship": "normative relationship and mapping",
    "resource_constraint": "limits and resource constraints",
    "explicit_reference": "referenced normative clause",
}


_RELATION_MESSAGE = re.compile(
    r"\b(?:[A-Z][A-Z0-9_/-]*\s+){1,6}"
    r"(?:REQUEST|RESPONSE|REJECT|ACCEPT|COMMAND|COMPLETE|INDICATION|"
    r"NOTIFICATION)\b"
)
_RELATION_IE = re.compile(
    r"\b(?:[A-Z0-9][A-Za-z0-9'()/+\-]*\s+){1,6}"
    r"(?:IE|information element)\b"
)
_TABLE_REFERENCE_NAME = re.compile(
    r"^(?P<name>.+?)(?=\d+(?:\.\d+)+[A-Za-z]?$)"
)
_RELATION_TERM_STOPWORDS = GENERIC_FACET_TERMS | frozenset({
    "format",
    "information element",
    "information element type/reference",
    "message type",
    "presence",
    "spare half octet",
    "type/reference",
})


def _useful_relation_term(surface: str) -> bool:
    folded = surface.strip(" .,:;()[]").casefold()
    if folded in _RELATION_TERM_STOPWORDS:
        return False
    if len(folded) < 3 or len(folded) > 100:
        return False
    if re.match(r"^\d+\s", folded):
        return False
    if re.match(r"^\d+\)", folded) or re.match(r"^\d{2,}[a-z]", folded):
        return False
    if folded.startswith(("this ", "the ", "a ", "an ")):
        return False
    if folded.endswith("message identity") or "conditional ie" in folded:
        return False
    return bool(re.search(r"[a-z]", folded, re.I))


def _grounded_terms_from_chunk(
    chunk: MarkdownChunk,
    focus_terms: tuple[str, ...],
) -> tuple[str, ...]:
    """Extract message/IE/table bridge terms without adding protocol facts."""

    text = chunk.text
    folded_focus = tuple(_fold_protocol_text(term) for term in focus_terms)
    candidates: list[tuple[int, str]] = []

    for match in _RELATION_MESSAGE.finditer(text):
        candidates.append((match.start(), match.group(0)))
    for match in _RELATION_IE.finditer(text):
        surface = match.group(0)
        candidates.append((match.start(), surface))
        if surface.casefold().endswith(" ie"):
            candidates.append((match.start(), surface[:-3]))

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        folded_line = _fold_protocol_text(stripped)
        if stripped.startswith("|") and any(
            term in folded_line for term in folded_focus
        ):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            for cell_index, cell in enumerate(cells):
                if not cell or cell in {"---", "M", "O", "C", "V", "TV", "TLV", "TLV-E", "LV", "LV-E"}:
                    continue
                surface = cell
                reference_match = _TABLE_REFERENCE_NAME.match(cell)
                if reference_match is not None:
                    surface = reference_match.group("name").strip()
                # The Information Element and Type/Reference columns carry
                # useful reverse-lookup terms. Other columns are identifiers,
                # presence flags, formats, or lengths.
                if cell_index not in {1, 2}:
                    continue
                position = text.find(surface, offset, offset + len(line))
                if position >= 0:
                    candidates.append((position, surface))
        offset += len(line)

    ordered: list[str] = []
    seen: set[str] = set()
    for _, surface in sorted(candidates, key=lambda item: (item[0], -len(item[1]))):
        cleaned = surface.strip(" .,:;()[]")
        key = cleaned.casefold()
        if key in seen or not _useful_relation_term(cleaned):
            continue
        if cleaned not in text:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return tuple(ordered[:24])


def _grounded_term_catalog(
    source_evidence_id: str,
    chunk: MarkdownChunk,
    focus_terms: tuple[str, ...],
) -> tuple[GroundedTerm, ...]:
    return tuple(
        GroundedTerm(
            term_id=f"{source_evidence_id}:T{index}",
            source_evidence_id=source_evidence_id,
            surface=surface,
        )
        for index, surface in enumerate(
            _grounded_terms_from_chunk(chunk, focus_terms),
            1,
        )
    )


def _relation_term_priority(term: str) -> tuple[int, int, str]:
    folded = term.casefold()
    score = 0
    if any(
        folded.endswith(suffix)
        for suffix in (
            " request", " response", " reject", " accept", " command",
            " complete", " indication", " notification",
        )
    ):
        score += 8
    if folded.endswith(" ie"):
        score += 6
    return (-score, len(term), folded)


def _build_grounded_relation_evidence(
    retriever: HybridMarkdownRetriever,
    resolved_plan: DiscoveryRetrievalPlanV3,
    *,
    max_per_question: int = 64,
    existing_chunk_ids_by_query: dict[str, frozenset[str]] | None = None,
) -> tuple[GroundedRelationEvidence, ...]:
    """Build a small corpus-grounded entity-to-message/IE relation frontier."""

    all_evidence: list[GroundedRelationEvidence] = []
    corpus_chunks = tuple(
        chunk
        for chunk in retriever.chunks
        if chunk.source_collection == "3gpp_markdown"
    )
    chunks_by_source: dict[str, list[MarkdownChunk]] = {}
    for chunk in corpus_chunks:
        chunks_by_source.setdefault(chunk.source_path, []).append(chunk)
    for chunks in chunks_by_source.values():
        chunks.sort(key=lambda value: (value.line_start, value.line_end, value.chunk_id))
    source_chunk_indexes = {
        (source_path, chunk.chunk_id): index
        for source_path, chunks in chunks_by_source.items()
        for index, chunk in enumerate(chunks)
    }

    for item in resolved_plan.retrievals:
        existing_chunk_ids = (
            existing_chunk_ids_by_query or {}
        ).get(item.query_id, frozenset())
        original_anchors = tuple(
            anchor.canonical
            for anchor in item.anchors
            if anchor.entity_type != "specification"
            and anchor.canonical.casefold() not in GENERIC_FACET_TERMS
        )
        if not original_anchors:
            continue
        folded_anchors = tuple(original_anchors)
        route_weights: dict[str, float] = {}
        for probe in item.retrieval_probes:
            for route in probe.routes:
                route_weights[route.specification] = max(
                    route_weights.get(route.specification, 0.0),
                    route.weight,
                )
        routed = {
            specification
            for specification in route_weights
        }
        seed_scores: list[tuple[float, MarkdownChunk]] = []
        for chunk in corpus_chunks:
            searchable = _fold_protocol_text(searchable_text(chunk))
            if not any(protocol_identifier_in_text(anchor, searchable)
                       for anchor in folded_anchors):
                continue
            priority = max(
                _exact_match_priority(chunk, probe)
                for probe in item.retrieval_probes
            )
            priority += 5.0 * route_weights.get(chunk.specification or "", 0.0)
            seed_scores.append((priority, chunk))
        seed_scores.sort(key=lambda value: (-value[0], value[1].chunk_id))
        # Process every exact occurrence when discovering bridge terms. The
        # final relation frontier remains bounded, but a low-ranked exact
        # occurrence can still contribute a message/IE relation.
        seeds = [chunk for _, chunk in seed_scores]
        seed_priority = {chunk.chunk_id: score for score, chunk in seed_scores}

        selected: dict[str, MarkdownChunk] = {}
        kinds: dict[str, set[str]] = {}
        focus_by_chunk: dict[str, set[str]] = {}
        priority_by_chunk: dict[str, float] = {}

        def add(
            chunk: MarkdownChunk,
            kind: str,
            focus: tuple[str, ...],
            priority: float,
        ) -> None:
            selected.setdefault(chunk.chunk_id, chunk)
            kinds.setdefault(chunk.chunk_id, set()).add(kind)
            focus_by_chunk.setdefault(chunk.chunk_id, set()).update(focus)
            priority_by_chunk[chunk.chunk_id] = max(
                priority_by_chunk.get(chunk.chunk_id, 0.0),
                priority,
            )

        for chunk in seeds:
            add(
                chunk,
                "exact_entity_occurrence",
                original_anchors,
                seed_priority[chunk.chunk_id],
            )

        bridge_terms: list[str] = []
        seen_terms: set[str] = set()
        for chunk in seeds:
            for term in _grounded_terms_from_chunk(chunk, original_anchors):
                key = term.casefold()
                if key in seen_terms or key in {value.casefold() for value in original_anchors}:
                    continue
                seen_terms.add(key)
                bridge_terms.append(term)
        bridge_terms.sort(key=_relation_term_priority)

        # Reverse lookup discovers message tables and IE definitions through
        # terms that were observed next to the original entity in the corpus.
        active_bridge_terms = bridge_terms[:24]
        per_term_limit = max(
            3,
            min(12, max_per_question // max(1, len(active_bridge_terms))),
        )
        for term in active_bridge_terms:
            folded_term = _fold_protocol_text(term)
            matches: list[tuple[float, MarkdownChunk]] = []
            for chunk in corpus_chunks:
                searchable = _fold_protocol_text(searchable_text(chunk))
                if not protocol_identifier_in_text(folded_term, searchable):
                    continue
                score = float(min(3, searchable.count(folded_term)))
                if chunk.specification in routed:
                    score += 3.0
                if protocol_identifier_in_text(
                    folded_term,
                    _fold_protocol_text(" ".join(chunk.heading_path)),
                ):
                    score += 8.0
                if "message content" in searchable:
                    score += 5.0
                if "type/reference" in searchable:
                    score += 3.0
                if "purpose of the" in searchable:
                    score += 7.0
                if any(protocol_identifier_in_text(anchor, searchable)
                       for anchor in folded_anchors):
                    score += 4.0
                matches.append((score, chunk))
            matches.sort(key=lambda value: (-value[0], value[1].chunk_id))
            # Allocate the reverse-lookup budget uniformly from the available
            # question budget; no protocol term receives a special quota.
            for score, chunk in matches[:per_term_limit]:
                searchable = _fold_protocol_text(searchable_text(chunk))
                kind = (
                    "reverse_message_table"
                    if "message content" in searchable
                    else "reverse_grounded_term"
                )
                add(chunk, kind, (term,), score)

        # Preserve immediate source neighbours for split tables and adjacent
        # message definitions. This follows document layout, not model memory.
        frontier = list(selected.values())
        for chunk in frontier:
            siblings = chunks_by_source[chunk.source_path]
            index = source_chunk_indexes[(chunk.source_path, chunk.chunk_id)]
            for neighbor_index in (index - 1, index + 1):
                if 0 <= neighbor_index < len(siblings):
                    neighbor = siblings[neighbor_index]
                    if abs(neighbor.line_start - chunk.line_start) <= 120:
                        add(neighbor, "adjacent_source_chunk", (), 0.0)

        base_ordered = sorted(
            selected.values(),
            key=lambda chunk: (
                0 if chunk.chunk_id not in existing_chunk_ids else 1,
                -priority_by_chunk[chunk.chunk_id],
                0 if chunk.specification in routed else 1,
                chunk.source_path,
                chunk.line_start,
                chunk.chunk_id,
            ),
        )
        ordered_chunks: list[MarkdownChunk] = []
        ordered_ids: set[str] = set()

        def take_bucket(
            predicate: Any,
            limit: int,
        ) -> None:
            if limit <= 0:
                return
            added = 0
            for chunk in base_ordered:
                if chunk.chunk_id in ordered_ids or not predicate(chunk):
                    continue
                ordered_chunks.append(chunk)
                ordered_ids.add(chunk.chunk_id)
                added += 1
                if added >= limit:
                    return

        # Keep independent recall channels. Exact-anchor occurrences are seed
        # evidence; they must not consume the entire relation frontier and
        # hide message tables that omit the original surface form.
        message_table_budget = max(1, (max_per_question * 2) // 5)
        reverse_relation_budget = max(1, max_per_question // 5)
        exact_budget = max(
            1,
            max_per_question - message_table_budget - reverse_relation_budget,
        )
        take_bucket(
            lambda chunk: "reverse_message_table" in kinds[chunk.chunk_id],
            message_table_budget,
        )
        take_bucket(
            lambda chunk: (
                "reverse_grounded_term" in kinds[chunk.chunk_id]
                and "reverse_message_table" not in kinds[chunk.chunk_id]
                and "exact_entity_occurrence" not in kinds[chunk.chunk_id]
            ),
            reverse_relation_budget,
        )
        take_bucket(
            lambda chunk: "exact_entity_occurrence" in kinds[chunk.chunk_id],
            exact_budget,
        )
        take_bucket(
            lambda chunk: kinds[chunk.chunk_id] != {"adjacent_source_chunk"},
            max_per_question - len(ordered_chunks),
        )
        take_bucket(
            lambda chunk: True,
            max_per_question - len(ordered_chunks),
        )
        for index, chunk in enumerate(ordered_chunks, 1):
            evidence_id = f"R-{item.query_id}:{index}"
            searchable = _fold_protocol_text(searchable_text(chunk))
            matched = tuple(
                anchor
                for anchor in original_anchors
                if protocol_identifier_in_text(anchor, searchable)
            )
            focus = tuple(sorted(focus_by_chunk[chunk.chunk_id], key=str.casefold))
            terms = _grounded_term_catalog(
                evidence_id,
                chunk,
                focus or original_anchors,
            )
            all_evidence.append(GroundedRelationEvidence(
                evidence_id=evidence_id,
                query_id=item.query_id,
                chunk_id=chunk.chunk_id,
                specification=chunk.specification or "",
                version=chunk.version or "",
                source_path=chunk.source_path,
                document_sha256=chunk.document_sha256,
                heading_path=chunk.heading_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                excerpt=chunk.text,
                matched_original_anchors=matched,
                bridge_terms=tuple(
                    term
                    for term in focus
                    if term.casefold()
                    not in {anchor.casefold() for anchor in original_anchors}
                ),
                grounded_terms=terms,
                relation_kinds=tuple(sorted(kinds[chunk.chunk_id])),
            ))
    return tuple(all_evidence)



__all__ = ['build_grounded_relation_evidence']

build_grounded_relation_evidence = _build_grounded_relation_evidence

"""Deterministic curation of first-pass and relation evidence candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .execution import (
    chunk_from_planned_hit as _chunk_from_planned_hit,
    fold_protocol_text as _fold_protocol_text,
)
from .execution_models import (
    CuratedEvidenceItem,
    CuratedEvidenceTask,
    EvidenceRelationClass,
    GroundedRelationEvidence,
    PlannedRetrievalTaskResult,
)
from .index import searchable_text
from .models import MarkdownChunk
from .plan_models import DiscoveryRetrievalPlanV3
from .planning import GENERIC_FACET_TERMS
from .ranking import ordered_unique as _ordered_unique
from .relation_evidence import _RELATION_MESSAGE, _grounded_terms_from_chunk
@dataclass
class _CuratedCandidate:
    chunk: MarkdownChunk
    source_evidence_ids: set[str] = field(default_factory=set)
    candidate_sources: set[str] = field(default_factory=set)
    matched_anchors: set[str] = field(default_factory=set)
    grounded_terms: set[str] = field(default_factory=set)
    relation_kinds: set[str] = field(default_factory=set)
    first_stage_score: float = 0.0


_ANGLE_DIRECT_CUES: dict[str, re.Pattern[str]] = {
    "carrier": re.compile(
        r"\b(?:include(?:s|d)?|contain(?:s|ed)?|convey(?:s|ed)?|"
        r"carr(?:y|ies|ied)|indicat(?:e|es|ed)|provide(?:s|d)?|"
        r"encapsulat(?:e|es|ed))\b",
        re.I,
    ),
    "format": re.compile(
        r"\b(?:represent(?:s|ed)?|consist(?:s|ed)?|compris(?:e|es|ed)|"
        r"contain(?:s|ed)?|cod(?:e|ed|ing)|encod(?:e|ed|ing)|"
        r"format|field|octet|bits?)\b",
        re.I,
    ),
    "reachability": re.compile(
        r"\b(?:send|sent|transmit(?:s|ted)?|transport(?:s|ed)?|via|over|"
        r"between|uplink|downlink|interface|encapsulat(?:e|es|ed))\b",
        re.I,
    ),
    "authority": re.compile(
        r"\b(?:authori[sz](?:e|es|ed|ation)|allow(?:s|ed)?|permit(?:s|ted)?|"
        r"shall|may|role|entity)\b",
        re.I,
    ),
    "protection": re.compile(
        r"\b(?:protect(?:s|ed|ion)?|integrity|confidentiality|authenticat(?:e|ed|ion)|"
        r"cipher(?:ed|ing)?|security context|replay|freshness)\b",
        re.I,
    ),
    "state_timing": re.compile(
        r"\b(?:before|after|during|when|while|state|phase|timer|expire(?:s|d)?|"
        r"start(?:s|ed)?|stop(?:s|ped)?|condition)\b",
        re.I,
    ),
    "required_inputs": re.compile(
        r"\b(?:require(?:s|d)?|mandatory|conditional|input|prerequisite|"
        r"shall include|available)\b",
        re.I,
    ),
    "relation": re.compile(
        r"\b(?:map(?:s|ped|ping)?|relat(?:e|es|ed|ionship)|"
        r"associat(?:e|es|ed|ion)|link(?:s|ed)?)\b",
        re.I,
    ),
    "resource_constraints": re.compile(
        r"\b(?:maximum|minimum|at most|at least|times?|attempts?|repeat(?:s|ed)?|"
        r"rate|duration|limit(?:s|ed)?|successive)\b",
        re.I,
    ),
}


def _candidate_relation_class(
    chunk: MarkdownChunk,
    *,
    angle: str,
    original_anchors: tuple[str, ...],
    relation_kinds: frozenset[str],
    grounded_terms: tuple[str, ...] = (),
) -> tuple[EvidenceRelationClass, tuple[str, ...], float]:
    """Conservatively distinguish explicit relations from useful bridges."""

    searchable = searchable_text(chunk)
    folded_searchable = _fold_protocol_text(searchable)
    folded_anchors = tuple(_fold_protocol_text(value) for value in original_anchors)
    matched = tuple(value for value in folded_anchors if value in folded_searchable)
    anchor_rows = tuple(
        line
        for line in chunk.text.splitlines()
        if line.strip().startswith("|")
        and any(value in _fold_protocol_text(line) for value in folded_anchors)
    )
    structured_table_relation = bool(anchor_rows) and (
        "message content" in folded_searchable
        or "type/reference" in folded_searchable
        or "| iei |" in folded_searchable
    )
    cue = _ANGLE_DIRECT_CUES.get(angle)
    relation_blocks = tuple(
        block.strip()
        for block in re.split(r"(?<=[.!?])\s+|\n\s*\n", chunk.text)
        if block.strip()
    )
    normative_relation = bool(cue) and any(
        any(value in _fold_protocol_text(block) for value in folded_anchors)
        and cue.search(block) is not None
        for block in relation_blocks
    )

    reasons: list[str] = []
    score = 0.0
    if structured_table_relation and angle in {"carrier", "format"}:
        reasons.append("structured table row contains the original anchor")
        score += 60.0
    if normative_relation:
        reasons.append("same normative text block links the anchor to the requested angle")
        score += 50.0
    if reasons:
        if "exact_entity_occurrence" in relation_kinds:
            score += 15.0
        if "reverse_message_table" in relation_kinds:
            score += 10.0
        return "direct_relation", tuple(reasons), score

    heading = _fold_protocol_text(" ".join(chunk.heading_path))
    non_message_bridge_terms = tuple(
        term
        for term in grounded_terms
        if _RELATION_MESSAGE.fullmatch(term.strip()) is None
    )
    if "reverse_message_table" in relation_kinds and non_message_bridge_terms:
        reasons.append(
            "message table contains a non-message bridge term useful for an evidence chain"
        )
        score += 35.0
    if matched and (
        "information element" in folded_searchable
        or "coded as" in folded_searchable
        or "message content" in folded_searchable
    ):
        reasons.append("anchor occurs in a message/IE definition useful for an evidence chain")
        score += 30.0
    if "reverse_grounded_term" in relation_kinds:
        reasons.append("chunk was found by reversing a term extracted from anchor evidence")
        score += 15.0
    if matched and any(value in heading for value in folded_anchors):
        reasons.append("original anchor occurs in the section heading")
        score += 10.0
    if reasons:
        if relation_kinds == frozenset({"adjacent_source_chunk"}):
            return (
                "context_only",
                ("adjacent chunk has no independently established relation",),
                0.0,
            )
        return "supporting_relation", tuple(dict.fromkeys(reasons)), score

    return (
        "context_only",
        ("no explicit or chain-supporting relation to the original anchor was established",),
        0.0,
    )


def curate_discovery_evidence(
    resolved_plan: DiscoveryRetrievalPlanV3,
    first_stage_tasks: tuple[PlannedRetrievalTaskResult, ...],
    relation_evidence: tuple[GroundedRelationEvidence, ...],
    *,
    final_evidence_hits: int = 20,
) -> tuple[CuratedEvidenceTask, ...]:
    """Merge first-pass and relation candidates, then select compact evidence."""

    if final_evidence_hits < 1:
        raise ValueError("final_evidence_hits must be positive")
    first_by_query = {task.query_id: task for task in first_stage_tasks}
    curated_tasks: list[CuratedEvidenceTask] = []
    for item in resolved_plan.retrievals:
        original_anchors = tuple(
            anchor.canonical
            for anchor in item.anchors
            if anchor.entity_type != "specification"
            and anchor.canonical.casefold() not in GENERIC_FACET_TERMS
        )
        candidates: dict[str, _CuratedCandidate] = {}

        def candidate_for(chunk: MarkdownChunk) -> _CuratedCandidate:
            return candidates.setdefault(chunk.chunk_id, _CuratedCandidate(chunk=chunk))

        for hit in first_by_query[item.query_id].hits:
            candidate = candidate_for(_chunk_from_planned_hit(hit))
            candidate.source_evidence_ids.add(hit.hit_id)
            candidate.candidate_sources.add("first_stage")
            candidate.matched_anchors.update(hit.matched_anchors)
            candidate.grounded_terms.update(
                _grounded_terms_from_chunk(candidate.chunk, original_anchors)
            )
            candidate.first_stage_score = max(candidate.first_stage_score, hit.final_score)
        for evidence in relation_evidence:
            if evidence.query_id != item.query_id:
                continue
            candidate = candidate_for(MarkdownChunk(
                chunk_id=evidence.chunk_id,
                source_collection="3gpp_markdown",
                specification=evidence.specification,
                version=evidence.version,
                source_path=evidence.source_path,
                document_sha256=evidence.document_sha256,
                heading_path=evidence.heading_path,
                line_start=evidence.line_start,
                line_end=evidence.line_end,
                text=evidence.excerpt,
            ))
            candidate.source_evidence_ids.add(evidence.evidence_id)
            candidate.candidate_sources.update(evidence.relation_kinds)
            candidate.relation_kinds.update(evidence.relation_kinds)
            candidate.matched_anchors.update(evidence.matched_original_anchors)
            candidate.grounded_terms.update(
                term.surface for term in evidence.grounded_terms
            )

        classified: list[CuratedEvidenceItem] = []
        for candidate in candidates.values():
            relation_class, reasons, score = _candidate_relation_class(
                candidate.chunk,
                angle=item.angle,
                original_anchors=original_anchors,
                relation_kinds=frozenset(candidate.relation_kinds),
                grounded_terms=_ordered_unique(candidate.grounded_terms),
            )
            if "first_stage" in candidate.candidate_sources:
                score += min(5.0, max(0.0, candidate.first_stage_score) * 20.0)
            classified.append(CuratedEvidenceItem(
                evidence_id="pending",
                query_id=item.query_id,
                chunk_id=candidate.chunk.chunk_id,
                specification=candidate.chunk.specification or "",
                version=candidate.chunk.version or "",
                source_path=candidate.chunk.source_path,
                document_sha256=candidate.chunk.document_sha256,
                heading_path=candidate.chunk.heading_path,
                line_start=candidate.chunk.line_start,
                line_end=candidate.chunk.line_end,
                excerpt=candidate.chunk.text,
                relation_class=relation_class,
                classification_reasons=reasons,
                candidate_sources=_ordered_unique(candidate.candidate_sources),
                source_evidence_ids=_ordered_unique(candidate.source_evidence_ids),
                matched_original_anchors=_ordered_unique(candidate.matched_anchors),
                grounded_terms=_ordered_unique(candidate.grounded_terms),
                curation_score=score,
            ))

        class_order = {
            "direct_relation": 0,
            "supporting_relation": 1,
            "context_only": 2,
        }
        classified.sort(key=lambda evidence: (
            class_order[evidence.relation_class],
            -evidence.curation_score,
            evidence.specification,
            evidence.line_start,
            evidence.chunk_id,
        ))
        selected = tuple(
            evidence.model_copy(update={
                "evidence_id": f"E-{item.query_id}:{rank}",
            })
            for rank, evidence in enumerate(
                (
                    value
                    for value in classified
                    if value.relation_class != "context_only"
                ),
                1,
            )
            if rank <= final_evidence_hits
        )
        curated_tasks.append(CuratedEvidenceTask(
            query_id=item.query_id,
            claim_id=item.claim_id,
            angle=item.angle,
            original_question=item.original_question,
            candidate_count=len(classified),
            direct_candidate_count=sum(
                value.relation_class == "direct_relation" for value in classified
            ),
            supporting_candidate_count=sum(
                value.relation_class == "supporting_relation" for value in classified
            ),
            context_candidate_count=sum(
                value.relation_class == "context_only" for value in classified
            ),
            selected_evidence=selected,
        ))
    return tuple(curated_tasks)



__all__ = ['curate_discovery_evidence']

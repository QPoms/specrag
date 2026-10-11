"""Candidate scoring and hit materialization for planned retrieval.

This module owns ranking policy only. Candidate discovery and reference
expansion remain in :mod:`specrag.retrieval.execution`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Literal

from .execution_models import PlannedRetrievalHit
from .models import MarkdownChunk
from .plan_models import PlannedDiscoveryRetrieval
from .reranker import CrossEncoderReranker


RankingMode = Literal[
    "full",
    "rrf",
    "rrf_anchor",
    "rrf_heading",
    "rrf_noise",
    "rrf_anchor_heading",
    "rrf_anchor_single",
    "rrf_anchor_full_coverage",
    "rrf_anchor_single_full_coverage",
    "rrf_anchor_single_if_full",
]

_LOW_VALUE_HEADINGS = (
    "change history",
    "handling of unknown, unforeseen",
    "handling of unknown unforeseen",
    "foreword",
    "references",
)


@dataclass
class RetrievalCandidate:
    """Mutable score accumulator shared by the retrieval channels."""

    chunk: MarkdownChunk
    rrf_score: float = 0.0
    route_score: float = 0.0
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    exact_anchor_rank: int | None = None
    exact_probe_ranks: dict[str, int] = field(default_factory=dict)
    lexical_probe_ranks: dict[str, int] = field(default_factory=dict)
    probe_ids: set[str] = field(default_factory=set)
    evidence_roles: set[str] = field(default_factory=set)
    anchors: set[str] = field(default_factory=set)
    heading_anchors: set[str] = field(default_factory=set)
    retrieval_hop: Literal[1, 2] = 2
    reference_source_chunk_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RankedRetrieval:
    hits: tuple[PlannedRetrievalHit, ...]
    reranker_status: Literal["disabled", "applied", "fallback"]
    reranker_error: str | None


def ordered_unique(values: set[str]) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda value: (value.casefold(), value)))


def _is_heading_only(chunk: MarkdownChunk) -> bool:
    lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
    return bool(lines) and all(re.fullmatch(r"#{1,6}\s+.+", line) for line in lines)


def _noise_penalty(candidate: RetrievalCandidate) -> float:
    headings = " ".join(candidate.chunk.heading_path).casefold()
    penalty = 0.0
    if _is_heading_only(candidate.chunk):
        penalty += 0.020
    if any(value in headings for value in _LOW_VALUE_HEADINGS):
        penalty += 0.020
    if candidate.retrieval_hop == 2:
        penalty += 0.002
    return penalty


def _deterministic_score(
    candidate: RetrievalCandidate,
    item: PlannedDiscoveryRetrieval,
    *,
    rrf_k: int,
    ranking_mode: RankingMode,
    anchor_weight: float,
    single_channel_max_rank: int,
    single_channel_weight: float,
    full_coverage_bonus: float,
) -> tuple[float, float, str]:
    anchor_score = min(0.016, anchor_weight * len(candidate.anchors))
    heading_score = min(0.015, 0.005 * len(candidate.heading_anchors))
    agreement_score = (
        0.004
        if candidate.semantic_rank is not None
        and candidate.lexical_rank is not None
        else 0.0
    )
    channel_ranks = tuple(
        rank
        for rank in (candidate.semantic_rank, candidate.lexical_rank)
        if rank is not None
    )
    single_channel_bonus = (
        single_channel_weight / (rrf_k + channel_ranks[0])
        if len(channel_ranks) == 1
        and channel_ranks[0] <= single_channel_max_rank
        else 0.0
    )
    full_anchor_coverage = any(
        probe.probe_id in candidate.probe_ids
        and set(probe.anchors).issubset(candidate.anchors)
        for probe in item.retrieval_probes
    )
    coverage_score = full_coverage_bonus if full_anchor_coverage else 0.0
    if ranking_mode == "rrf":
        final = candidate.rrf_score
    elif ranking_mode == "rrf_anchor":
        final = candidate.rrf_score + anchor_score
    elif ranking_mode == "rrf_heading":
        final = candidate.rrf_score + heading_score
    elif ranking_mode == "rrf_noise":
        final = candidate.rrf_score - _noise_penalty(candidate)
    elif ranking_mode == "rrf_anchor_heading":
        final = candidate.rrf_score + anchor_score + heading_score
    elif ranking_mode == "rrf_anchor_single":
        final = candidate.rrf_score + anchor_score + single_channel_bonus
    elif ranking_mode == "rrf_anchor_full_coverage":
        final = candidate.rrf_score + anchor_score + coverage_score
    elif ranking_mode == "rrf_anchor_single_full_coverage":
        final = (
            candidate.rrf_score
            + anchor_score
            + single_channel_bonus
            + coverage_score
        )
    elif ranking_mode == "rrf_anchor_single_if_full":
        final = (
            candidate.rrf_score
            + anchor_score
            + (single_channel_bonus if full_anchor_coverage else 0.0)
        )
    else:
        final = (
            candidate.rrf_score
            + candidate.route_score
            + anchor_score
            + agreement_score
            - _noise_penalty(candidate)
        )
    return final, candidate.rrf_score, candidate.chunk.chunk_id


def _normalized(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def rank_retrieval_candidates(
    candidates: dict[str, RetrievalCandidate],
    item: PlannedDiscoveryRetrieval,
    *,
    rrf_k: int,
    final_hits: int,
    reserve_probe_candidates: bool,
    ranking_mode: RankingMode,
    anchor_weight: float,
    single_channel_max_rank: int,
    single_channel_weight: float,
    full_coverage_bonus: float,
    reranker: CrossEncoderReranker | None,
) -> RankedRetrieval:
    """Apply deterministic ranking, optional reranking, and result selection."""

    def score(candidate: RetrievalCandidate) -> tuple[float, float, str]:
        return _deterministic_score(
            candidate,
            item,
            rrf_k=rrf_k,
            ranking_mode=ranking_mode,
            anchor_weight=anchor_weight,
            single_channel_max_rank=single_channel_max_rank,
            single_channel_weight=single_channel_weight,
            full_coverage_bonus=full_coverage_bonus,
        )

    deterministic_scores = {
        candidate.chunk.chunk_id: score(candidate)[0]
        for candidate in candidates.values()
    }
    reranker_scores: dict[str, float] = {}
    deterministic_normalized: dict[str, float] = {}
    reranker_normalized: dict[str, float] = {}
    reranker_status: Literal["disabled", "applied", "fallback"] = "disabled"
    reranker_error: str | None = None

    if reranker is not None and candidates:
        rerank_candidates = sorted(
            candidates.values(),
            key=lambda candidate: candidate.chunk.chunk_id,
        )
        query = "\n".join(dict.fromkeys(
            probe.semantic_query for probe in item.retrieval_probes
        ))
        passages = [
            "\n\n".join(filter(None, (
                " > ".join(candidate.chunk.heading_path),
                candidate.chunk.text,
            )))
            for candidate in rerank_candidates
        ]
        try:
            raw_scores = reranker.predict(query, passages)
            if len(raw_scores) != len(rerank_candidates):
                raise RuntimeError(
                    "reranker returned a different number of scores than passages"
                )
            deterministic_values = [
                deterministic_scores[candidate.chunk.chunk_id]
                for candidate in rerank_candidates
            ]
            for candidate, raw, deterministic_norm, reranker_norm in zip(
                rerank_candidates,
                raw_scores,
                _normalized(deterministic_values),
                _normalized(list(raw_scores)),
            ):
                chunk_id = candidate.chunk.chunk_id
                reranker_scores[chunk_id] = raw
                deterministic_normalized[chunk_id] = deterministic_norm
                reranker_normalized[chunk_id] = reranker_norm
            reranker_status = "applied"
        except Exception as exc:  # Deterministic ranking remains authoritative.
            reranker_status = "fallback"
            reranker_error = f"{type(exc).__name__}: {exc}"

    def ranking_score(candidate: RetrievalCandidate) -> float:
        chunk_id = candidate.chunk.chunk_id
        if reranker_status != "applied":
            return deterministic_scores[chunk_id]
        assert reranker is not None
        return (
            reranker_normalized[chunk_id]
            + reranker.deterministic_weight * deterministic_normalized[chunk_id]
        )

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (
            -ranking_score(candidate),
            -score(candidate)[1],
            score(candidate)[2],
        ),
    )
    selected: list[RetrievalCandidate] = []
    selected_ids: set[str] = set()
    if reserve_probe_candidates:
        for probe in item.retrieval_probes:
            exact_for_probe = sorted(
                (
                    value
                    for value in candidates.values()
                    if probe.probe_id in value.exact_probe_ranks
                    and value.chunk.chunk_id not in selected_ids
                ),
                key=lambda value: (
                    value.exact_probe_ranks[probe.probe_id],
                    value.chunk.chunk_id,
                ),
            )
            lexical_for_probe = sorted(
                (
                    value
                    for value in candidates.values()
                    if probe.probe_id in value.lexical_probe_ranks
                    and value.chunk.chunk_id not in selected_ids
                ),
                key=lambda value: (
                    value.lexical_probe_ranks[probe.probe_id],
                    -score(value)[0],
                    value.chunk.chunk_id,
                ),
            )
            candidate = (
                exact_for_probe[0]
                if exact_for_probe
                else lexical_for_probe[0]
                if lexical_for_probe
                else next(
                    (
                        value
                        for value in ordered
                        if probe.probe_id in value.probe_ids
                        and value.chunk.chunk_id not in selected_ids
                    ),
                    None,
                )
            )
            if candidate is not None:
                selected.append(candidate)
                selected_ids.add(candidate.chunk.chunk_id)
    for candidate in ordered:
        if len(selected) >= final_hits:
            break
        if candidate.chunk.chunk_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.chunk.chunk_id)
    selected.sort(
        key=lambda candidate: (
            -ranking_score(candidate),
            candidate.chunk.chunk_id,
        )
    )

    hits: list[PlannedRetrievalHit] = []
    for rank, candidate in enumerate(selected[:final_hits], 1):
        chunk = candidate.chunk
        source_hit_ids = set(candidate.reference_source_chunk_ids)
        for probe in item.retrieval_probes:
            if probe.probe_id in candidate.probe_ids:
                source_hit_ids.update(probe.source_hit_ids)
        hits.append(PlannedRetrievalHit(
            hit_id=f"{item.retrieval_id}:3gpp_markdown:{rank}",
            chunk_id=chunk.chunk_id,
            collection_rank=rank,
            specification=chunk.specification or "",
            version=chunk.version or "",
            source_path=chunk.source_path,
            document_sha256=chunk.document_sha256,
            heading_path=chunk.heading_path,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            excerpt=chunk.text,
            evidence_roles=ordered_unique(candidate.evidence_roles),
            probe_ids=ordered_unique(candidate.probe_ids),
            matched_anchors=ordered_unique(candidate.anchors),
            retrieval_hop=candidate.retrieval_hop,
            reference_source_chunk_ids=ordered_unique(source_hit_ids),
            semantic_rank=candidate.semantic_rank,
            lexical_rank=candidate.lexical_rank,
            rrf_score=candidate.rrf_score,
            route_score=candidate.route_score,
            anchor_score=min(0.016, anchor_weight * len(candidate.anchors)),
            heading_score=min(0.015, 0.005 * len(candidate.heading_anchors)),
            agreement_score=(
                0.004
                if candidate.semantic_rank is not None
                and candidate.lexical_rank is not None
                else 0.0
            ),
            noise_penalty=_noise_penalty(candidate),
            deterministic_score=deterministic_scores[chunk.chunk_id],
            reranker_score=reranker_scores.get(chunk.chunk_id),
            deterministic_score_normalized=deterministic_normalized.get(
                chunk.chunk_id
            ),
            reranker_score_normalized=reranker_normalized.get(chunk.chunk_id),
            reranker_model=(
                reranker.model_name if reranker_status == "applied" else None
            ),
            final_score=ranking_score(candidate),
        ))

    return RankedRetrieval(
        hits=tuple(hits),
        reranker_status=reranker_status,
        reranker_error=reranker_error,
    )


__all__ = [
    "RankedRetrieval",
    "RankingMode",
    "RetrievalCandidate",
    "ordered_unique",
    "rank_retrieval_candidates",
]

"""Researcher action instructions, validation, and deduplication policy."""

from __future__ import annotations

from typing import Any

from .models import AgentRetrievalAction
from .state import EvidenceLedger


def validate_action(
    action: AgentRetrievalAction,
    *,
    query_ids: frozenset[str],
    available_specifications: frozenset[str],
    ledger: EvidenceLedger,
) -> tuple[str, ...]:
    errors: list[str] = []
    if action.tool == "finish":
        if (
            action.query_id
            or action.lexical_query
            or action.semantic_query
            or action.clause
            or action.target_specifications
            or action.seed_evidence_ids
            or action.max_results != 1
        ):
            errors.append("finish contains non-empty tool parameters")
        return tuple(errors)
    if action.query_id not in query_ids:
        errors.append(f"unknown query_id: {action.query_id}")
        return tuple(errors)
    unknown_specs = sorted(
        set(action.target_specifications) - available_specifications
    )
    if unknown_specs:
        errors.append("unknown specifications: " + ", ".join(unknown_specs))
    if action.tool == "keyword_search":
        if not action.lexical_query.strip():
            errors.append("keyword_search requires lexical_query")
        if action.semantic_query or action.clause or action.seed_evidence_ids:
            errors.append("keyword_search contains incompatible parameters")
    elif action.tool == "semantic_search":
        if not action.semantic_query.strip():
            errors.append("semantic_search requires semantic_query")
        if action.lexical_query or action.clause or action.seed_evidence_ids:
            errors.append("semantic_search contains incompatible parameters")
    elif action.tool == "hybrid_search":
        if not action.lexical_query.strip() or not action.semantic_query.strip():
            errors.append("hybrid_search requires both query strings")
        if action.clause or action.seed_evidence_ids:
            errors.append("hybrid_search contains incompatible parameters")
    elif action.tool == "clause_lookup":
        if not action.clause.strip():
            errors.append("clause_lookup requires clause")
        if action.lexical_query or action.semantic_query or action.seed_evidence_ids:
            errors.append("clause_lookup contains incompatible parameters")
    elif action.tool == "rerank":
        if not action.semantic_query.strip():
            errors.append("rerank requires semantic_query")
        if action.lexical_query or action.clause or action.target_specifications:
            errors.append("rerank contains incompatible parameters")
        if not action.seed_evidence_ids:
            errors.append("rerank requires seed evidence IDs")
    elif action.tool == "expand_same_section":
        if action.lexical_query or action.semantic_query or action.clause:
            errors.append("expand_same_section contains incompatible parameters")
        if action.target_specifications:
            errors.append("expand_same_section cannot contain target specifications")
        if not action.seed_evidence_ids:
            errors.append("expand_same_section requires seed evidence IDs")
    if action.tool in {"rerank", "expand_same_section"}:
        unknown_ids = [
            evidence_id
            for evidence_id in action.seed_evidence_ids
            if ledger.find(action.query_id, evidence_id) is None
        ]
        if unknown_ids:
            errors.append("unknown seed evidence IDs: " + ", ".join(unknown_ids))
    return tuple(errors)


def action_signature(action: AgentRetrievalAction) -> tuple[Any, ...]:
    return (
        action.tool,
        action.query_id,
        action.lexical_query.casefold().strip(),
        action.semantic_query.casefold().strip(),
        action.clause.casefold().strip(),
        tuple(sorted(action.target_specifications)),
        tuple(sorted(action.seed_evidence_ids)),
    )


__all__ = [
    "action_signature",
    "validate_action",
]

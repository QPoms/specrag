"""Trusted validation and deduplication for native retrieval calls."""

from __future__ import annotations

from typing import Any

from specrag.tools import (
    ExpandSourceContextArguments,
    RerankArguments,
    RetrievalToolArguments,
)

from .models import AgentRetrievalCall
from .state import EvidenceLedger


def validate_call(
    call: AgentRetrievalCall,
    arguments: RetrievalToolArguments,
    *,
    query_ids: frozenset[str],
    available_specifications: frozenset[str],
    grounded_specifications: frozenset[str],
    ledger: EvidenceLedger,
) -> tuple[str, ...]:
    errors: list[str] = []
    if arguments.query_id not in query_ids:
        return (f"unknown query_id: {arguments.query_id}",)

    target_specifications = tuple(
        getattr(arguments, "target_specifications", ())
    )
    unknown_specs = sorted(
        set(target_specifications) - available_specifications
    )
    if unknown_specs:
        errors.append("unknown specifications: " + ", ".join(unknown_specs))
    ungrounded_specs = sorted(
        (set(target_specifications) & available_specifications)
        - grounded_specifications
    )
    if ungrounded_specs:
        errors.append(
            "specifications are not grounded in the task, shared context, or "
            "observed evidence: " + ", ".join(ungrounded_specs)
        )

    if isinstance(arguments, (RerankArguments, ExpandSourceContextArguments)):
        unknown_ids = [
            evidence_id
            for evidence_id in arguments.seed_evidence_ids
            if ledger.find(arguments.query_id, evidence_id) is None
        ]
        if unknown_ids:
            errors.append(
                "unknown seed evidence IDs: " + ", ".join(unknown_ids)
            )
    return tuple(errors)


def call_signature(
    call: AgentRetrievalCall,
    arguments: RetrievalToolArguments,
) -> tuple[Any, ...]:
    values = arguments.model_dump(mode="json", exclude={"max_results"})
    normalized: list[tuple[str, Any]] = []
    for name, value in sorted(values.items()):
        if isinstance(value, str):
            value = value.casefold().strip()
        elif isinstance(value, list):
            value = tuple(sorted(value))
        normalized.append((name, value))
    return (call.tool, *normalized)


__all__ = ["call_signature", "validate_call"]

"""Generic normalization helpers for protocol identifiers.

The corpus is not consistent about separators inside identifiers.  For
example, the same identifier may occur as ``NR-Uu`` or ``NR Uu``.  These
helpers normalize only identifier-like terms; they do not create semantic
aliases such as treating bare ``Uu`` as ``NR-Uu``.
"""

from __future__ import annotations

import re
from typing import Iterable


_SEPARATOR = re.compile(r"[\s_./\\-]+")
_COMPOUND_IDENTIFIER = re.compile(
    r"^[a-z0-9]+(?:[\s_./\\-]+[a-z0-9]+)+$",
    re.I,
)
_UNICODE_DASHES = re.compile(r"[\u2010-\u2015]")

# Generic evidence-shape terms are not concrete protocol identifiers. Retained
# retrieval paths use this set only to avoid promoting them as entity matches.
GENERIC_FACET_TERMS = frozenset({
    "channel",
    "channels",
    "container",
    "containers",
    "endpoint",
    "endpoints",
    "field",
    "fields",
    "ie",
    "ies",
    "information element",
    "information elements",
    "interface",
    "interfaces",
    "message",
    "messages",
    "path",
    "paths",
    "procedure",
    "procedures",
    "protocol message",
    "protocol messages",
    "ue",
    "network",
})


def _fold(value: str) -> str:
    return _UNICODE_DASHES.sub("-", value).casefold().strip()


def protocol_identifier_variants(value: str) -> tuple[str, ...]:
    """Return spelling variants for one compound protocol identifier."""

    folded = _fold(value)
    if not folded or not _COMPOUND_IDENTIFIER.fullmatch(folded):
        return (folded,) if folded else ()
    parts = tuple(part for part in _SEPARATOR.split(folded) if part)
    variants = (
        folded,
        "-".join(parts),
        " ".join(parts),
        "".join(parts),
    )
    return tuple(dict.fromkeys(variants))


def protocol_identifier_in_text(identifier: str, text: str) -> bool:
    """Check an identifier against separator-only spelling variants."""

    folded_text = _fold(text)
    return any(variant in folded_text
               for variant in protocol_identifier_variants(identifier))


def protocol_token_variants(token: str) -> tuple[str, ...]:
    """Expand a tokenizer token without adding semantic aliases."""

    folded = _fold(token)
    if not folded or not _COMPOUND_IDENTIFIER.fullmatch(folded):
        return (folded,) if folded else ()
    parts = tuple(part for part in _SEPARATOR.split(folded) if part)
    return tuple(dict.fromkeys((
        folded,
        *parts,
        "".join(parts),
    )))


def any_protocol_identifier_in_text(
    identifiers: Iterable[str],
    text: str,
) -> bool:
    return any(protocol_identifier_in_text(identifier, text)
               for identifier in identifiers)


__all__ = [
    "GENERIC_FACET_TERMS",
    "any_protocol_identifier_in_text",
    "protocol_identifier_in_text",
    "protocol_identifier_variants",
    "protocol_token_variants",
]

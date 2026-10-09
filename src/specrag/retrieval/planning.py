"""Source-grounded compilation of research questions into retrieval plans."""

from __future__ import annotations

import re
from typing import Any, Literal, TypeVar

from ..infrastructure.structured_agent import CodexAgentReceipt, CodexStructuredAgent
from .plan_models import (
    AnchorType,
    DiscoveryRetrievalPlanV3,
    ExplicitFacet,
    LLMAnchorBatch,
    LLMAnchorProposal,
    PlannedDiscoveryRetrieval,
    QueryAnchor,
    RetrievalProbe,
    SourceSpan,
)

_DEICTIC_REFERENCE = re.compile(
    r"\b(?:this|these|those|such|aforementioned)"
    r"(?:\s+[a-z][a-z-]*){0,4}\s+"
    r"(?:claim|object|objects|message|messages|information|procedure|procedures)\b",
    re.I,
)
_T = TypeVar("_T")

# These words describe the requested evidence shape. They are facets, not
# protocol entities, and must never compete with concrete identifiers during
# lexical retrieval.
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


def _angle_value(question: Any) -> str:
    angle = question.angle
    return str(getattr(angle, "value", angle))


def _span(text: str, start: int, end: int) -> SourceSpan:
    return SourceSpan(text=text[start:end], start=start, end=end)


def _canonicalize(
    surface: str,
) -> tuple[str, Literal["verbatim", "orthographic"]]:
    canonical = re.sub(r"[\u2010-\u2015]", "-", surface)
    canonical = re.sub(r"\s+", " ", canonical).strip()
    return canonical, "verbatim" if canonical == surface else "orthographic"


def _classify_anchor(surface: str) -> AnchorType:
    folded = surface.casefold()
    if re.fullmatch(r"(?:3gpp\s+)?ts\s+\d{2}\.\d{3}", folded):
        return "specification"
    if re.fullmatch(r"t\d{3,5}", folded):
        return "timer"
    if re.fullmatch(r"#\d+", folded):
        return "literal_value"
    if folded.endswith((" ie", " information element")):
        return "information_element"
    if folded.endswith((
        " request",
        " response",
        " reject",
        " accept",
        " command",
        " complete",
        " indication",
        " notification",
        " message",
    )):
        return "protocol_message"
    if any(value in folded for value in ("-registered", "-idle", "-connected")):
        return "protocol_state"
    return "technical_term"


_TECHNICAL_GROUP = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Z0-9][A-Z0-9]*(?:[-\u2010-\u2015_/][A-Z0-9]+)*(?:\(s\))?)"
    r"(?:\s+(?:[A-Z0-9][A-Z0-9]*(?:[-\u2010-\u2015_/][A-Z0-9]+)*"
    r"(?:\(s\))?)){0,5}(?![A-Za-z0-9])"
)
_MESSAGE_PHRASE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*\s+){1,5}"
    r"(?:Request|Response|Reject|Accept|Command|Complete|Indication|"
    r"Notification|Message)\b"
)
_IE_PHRASE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*(?:\(s\))?\s+){1,5}IE\b"
)
_EXPLICIT_SPEC = re.compile(r"(?:3GPP\s+)?TS\s+\d{2}\.\d{3}", re.I)
_TIMER = re.compile(r"\bT\d{3,5}\b")
_LITERAL = re.compile(r"#\d+\b")
_ANCHOR_STOPWORDS = frozenset({
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BY", "CAN", "DOES",
    "FOR", "FROM", "HOW", "IF", "IN", "IS", "IT", "OF", "ON", "OR",
    "THE", "THIS", "TO", "UNDER", "WHAT", "WHEN", "WHETHER", "WHICH",
    "WITH", "WITHOUT",
})


def _extract_anchors(text: str) -> tuple[QueryAnchor, ...]:
    candidates: list[tuple[int, int]] = []
    for pattern in (
        _EXPLICIT_SPEC,
        _MESSAGE_PHRASE,
        _IE_PHRASE,
        _TIMER,
        _LITERAL,
        _TECHNICAL_GROUP,
    ):
        candidates.extend(
            (match.start(), match.end()) for match in pattern.finditer(text)
        )

    selected: list[tuple[int, int]] = []
    for start, end in sorted(
        candidates,
        key=lambda item: (item[0], -(item[1] - item[0])),
    ):
        surface = text[start:end].strip()
        if not surface:
            continue
        adjusted_start = text.find(surface, start, end)
        adjusted_end = adjusted_start + len(surface)
        tokens = surface.upper().split()
        if tokens and all(token in _ANCHOR_STOPWORDS for token in tokens):
            continue
        if surface.casefold() in GENERIC_FACET_TERMS:
            continue
        if any(
            start0 <= adjusted_start and adjusted_end <= end0
            for start0, end0 in selected
        ):
            continue
        selected.append((adjusted_start, adjusted_end))

    anchors: list[QueryAnchor] = []
    seen: set[str] = set()
    for start, end in sorted(selected):
        surface = text[start:end]
        canonical, normalization = _canonicalize(surface)
        key = canonical.casefold()
        if key in seen:
            continue
        seen.add(key)
        anchors.append(QueryAnchor(
            surface=surface,
            canonical=canonical,
            entity_type=_classify_anchor(surface),
            source_span=_span(text, start, end),
            normalization=normalization,
        ))
    return tuple(anchors)


def _validated_llm_anchors(
    text: str,
    proposals: tuple[LLMAnchorProposal, ...],
) -> tuple[QueryAnchor, ...]:
    anchors: list[QueryAnchor] = []
    for proposal in proposals:
        start = text.find(proposal.surface)
        if start < 0:
            raise ValueError(
                "Codex anchor does not occur verbatim in the question"
            )
        end = start + len(proposal.surface)
        source = proposal.surface
        canonical, normalization = _canonicalize(source)
        anchors.append(QueryAnchor(
            surface=source,
            canonical=canonical,
            entity_type=proposal.entity_type,
            source_span=_span(text, start, end),
            normalization=normalization,
        ))
    return tuple(anchors)


def _merge_anchors(
    *groups: tuple[QueryAnchor, ...],
) -> tuple[QueryAnchor, ...]:
    # Prefer a longer extractive phrase to a contained token. This lets an LLM
    # identify natural-language protocol names while retaining strict spans.
    candidates = [anchor for group in groups for anchor in group]
    selected: list[QueryAnchor] = []
    for anchor in sorted(
        candidates,
        key=lambda item: (
            item.source_span.start,
            -(item.source_span.end - item.source_span.start),
        ),
    ):
        span = anchor.source_span
        if any(
            existing.source_span.start <= span.start
            and span.end <= existing.source_span.end
            for existing in selected
        ):
            continue
        if any(existing.canonical.casefold() == anchor.canonical.casefold() for existing in selected):
            continue
        selected.append(anchor)
    return tuple(sorted(selected, key=lambda item: item.source_span.start))


_FACET_CATALOG: dict[str, tuple[tuple[str, str], ...]] = {
    "format": (
        ("representation", r"\brepresent(?:ation|ed|s|ing)?\b"),
        ("format", r"\bformat\b"),
        ("structure", r"\bstructur(?:e|ed)\b"),
        ("fields", r"\bfields?\b"),
        ("encoding", r"\bencod(?:e|ed|ing)\b"),
        ("values", r"\bvalues?\b"),
    ),
    "carrier": (
        ("protocol_message", r"\b(?:protocol\s+)?messages?\b"),
        ("information_element", r"\binformation elements?\b|\bIEs?\b"),
        ("container", r"\bcontainers?\b"),
        ("procedure", r"\bprocedures?\b"),
        ("channel", r"\bchannels?\b"),
    ),
    "reachability": (
        ("endpoint", r"\bendpoints?\b"),
        ("interface", r"\binterfaces?\b"),
        ("direction", r"\bdirections?\b|\buplink\b|\bdownlink\b"),
        ("transport", r"\btransport\b"),
        ("path", r"\bpaths?\b"),
        ("encapsulation", r"\bencapsulation\b"),
        ("channel", r"\bchannels?\b"),
    ),
    "authority": (
        ("role", r"\broles?\b"),
        ("identity", r"\bidentity\b"),
        ("authorization", r"\bauthori[sz](?:ation|ed)\b"),
        ("entity", r"\bentit(?:y|ies)\b"),
    ),
    "protection": (
        ("integrity", r"\bintegrity(?:-protection| protection)?\b"),
        ("confidentiality", r"\bconfidentiality\b"),
        ("authentication", r"\bauthentication\b"),
        ("replay", r"\breplay\b"),
        ("freshness", r"\bfreshness\b"),
        ("security_context", r"\bsecurity(?:-| )context\b"),
        ("protection", r"\bprotection\b"),
    ),
    "state_timing": (
        ("state", r"\bstates?\b"),
        ("phase", r"\bphases?\b"),
        ("timing", r"\btiming\b"),
        ("timer", r"\btimers?\b"),
        ("condition", r"\bconditions?\b"),
        ("ordering", r"\bordering\b"),
        ("window", r"\bwindows?\b"),
        ("context", r"\bcontexts?\b"),
    ),
    "required_inputs": (
        ("prerequisite", r"\bprerequisites?\b"),
        ("mandatory", r"\bmandatory\b"),
        ("conditional", r"\bconditionally?\b"),
        ("input", r"\binputs?\b"),
    ),
    "relation": (
        ("mapping", r"\bmapping\b"),
        ("relationship", r"\brelationships?\b"),
        ("association", r"\bassociate(?:s|d)?\b|\bassociation\b"),
        ("link", r"\blinks?\b"),
    ),
    "resource_constraints": (
        ("repetition", r"\brepetition\b|\brepeated\b|\bsuccessive\b"),
        ("rate", r"\brate\b"),
        ("duration", r"\bduration\b"),
        ("constraint", r"\bconstraints?\b"),
        ("attempt", r"\battempts?\b"),
    ),
}


def _extract_facets(text: str, angle: str) -> tuple[ExplicitFacet, ...]:
    matches: list[tuple[int, int, str]] = []
    matched_names: set[str] = set()
    for name, pattern in _FACET_CATALOG.get(angle, ()):
        for match in re.finditer(pattern, text, re.I):
            if name in matched_names:
                break
            if any(
                start <= match.start() and match.end() <= end
                for start, end, _ in matches
            ):
                continue
            matches.append((match.start(), match.end(), name))
            matched_names.add(name)
    return tuple(
        ExplicitFacet(name=name, source_span=_span(text, start, end))
        for start, end, name in sorted(matches)
    )


def _partition(values: tuple[_T, ...], count: int) -> tuple[tuple[_T, ...], ...]:
    groups: list[list[_T]] = [[] for _ in range(count)]
    for index, value in enumerate(values):
        groups[index % count].append(value)
    return tuple(tuple(group) for group in groups if group)


def _compile_probes(
    text: str,
    angle: str,
    anchors: tuple[QueryAnchor, ...],
    facets: tuple[ExplicitFacet, ...],
) -> tuple[RetrievalProbe, ...]:
    if not anchors:
        return (RetrievalProbe(
            probe_id="P1",
            evidence_role=angle,
            lexical_query=text,
            semantic_query=text,
        ),)

    # One explicit aspect per probe keeps dense retrieval meaningfully
    # different across message, IE, container, path, and similar facets.
    facet_groups = tuple((facet,) for facet in facets[:5]) or ((),)
    anchor_groups = (anchors,) * len(facet_groups)

    probes: list[RetrievalProbe] = []
    for index, (anchor_group, facet_group) in enumerate(
        zip(anchor_groups, facet_groups, strict=True),
        1,
    ):
        lexical_parts = [anchor.surface for anchor in anchor_group]
        lexical_parts.extend(facet.source_span.text for facet in facet_group)
        lexical = " ".join(dict.fromkeys(lexical_parts)).strip() or text
        semantic = _facet_semantic_query(
            text,
            angle,
            anchor_group,
            facet_group,
        )
        probes.append(RetrievalProbe(
            probe_id=f"P{index}",
            evidence_role=angle,
            lexical_query=lexical,
            semantic_query=semantic,
            anchors=tuple(anchor.canonical for anchor in anchor_group),
            facets=tuple(facet.name for facet in facet_group),
        ))
    return tuple(probes)


def _facet_semantic_query(
    original_question: str,
    angle: str,
    anchors: tuple[QueryAnchor, ...],
    facets: tuple[ExplicitFacet, ...],
) -> str:
    """Build a source-faithful semantic query focused on one explicit facet."""

    if not facets:
        return original_question
    concrete = "; ".join(anchor.canonical for anchor in anchors)
    requested = "; ".join(facet.source_span.text for facet in facets)
    if not concrete:
        return original_question
    templates = {
        "carrier": "Which {facet} conveys or carries {anchors}?",
        "format": "What {facet} specifies the representation of {anchors}?",
        "reachability": "What {facet} describes the reachability of {anchors}?",
        "authority": "What {facet} specifies authority for {anchors}?",
        "protection": "What {facet} specifies protection of {anchors}?",
        "state_timing": "What {facet} specifies state or timing for {anchors}?",
        "required_inputs": "What {facet} specifies required inputs for {anchors}?",
        "relation": "What {facet} specifies the relationship of {anchors}?",
        "resource_constraints": "What {facet} specifies constraints on {anchors}?",
    }
    template = templates.get(angle)
    if template is None:
        return f"{concrete}. {requested}. {angle}."
    return template.format(facet=requested, anchors=concrete)


def _validate_source_grounding(item: PlannedDiscoveryRetrieval) -> None:
    text = item.original_question
    for anchor in item.anchors:
        span = anchor.source_span
        if text[span.start:span.end] != span.text:
            raise ValueError(f"ungrounded anchor in {item.query_id}: {anchor.surface}")
    for facet in item.explicit_facets:
        span = facet.source_span
        if text[span.start:span.end] != span.text:
            raise ValueError(f"ungrounded facet in {item.query_id}: {facet.name}")


def _has_unresolved_reference(
    text: str,
    anchors: tuple[QueryAnchor, ...],
) -> bool:
    for match in _DEICTIC_REFERENCE.finditer(text):
        if not any(
            match.start() <= anchor.source_span.start
            and anchor.source_span.end <= match.end()
            for anchor in anchors
        ):
            return True
    return False


def _compile_one(
    question: Any,
    llm_anchors: tuple[QueryAnchor, ...] = (),
) -> PlannedDiscoveryRetrieval:
    text = question.question
    angle = _angle_value(question)
    anchors = _merge_anchors(llm_anchors, _extract_anchors(text))
    facets = _extract_facets(text, angle)
    context_dependent = _has_unresolved_reference(text, anchors)
    warnings: list[str] = []
    if context_dependent:
        warnings.append(
            "question contains an unresolved reference; the planner did not "
            "import entities from another question or claim"
        )
    if not anchors:
        warnings.append(
            "no extractive protocol anchor was found; use full-corpus fallback"
        )
    action: Literal["planned", "fallback_full_corpus"] = (
        "planned" if anchors and not context_dependent else "fallback_full_corpus"
    )
    item = PlannedDiscoveryRetrieval(
        retrieval_id=f"P-{question.discovery_id}",
        query_id=question.discovery_id,
        claim_id=question.claim_id,
        angle=angle,
        original_question=text,
        planner_action=action,
        anchors=anchors,
        explicit_facets=facets,
        context_dependent=context_dependent,
        warnings=tuple(warnings),
        retrieval_probes=_compile_probes(text, angle, anchors, facets),
    )
    _validate_source_grounding(item)
    return item


def compile_discovery_retrieval_plan(
    discovery_questions: tuple[Any, ...],
    *,
    planner_agent: CodexStructuredAgent | None = None,
) -> DiscoveryRetrievalPlanV3:
    """Compile questions without importing unstated protocol facts.

    If a Codex agent is supplied, it is used once for batch extractive NER. Its
    surface terms are accepted only when they occur verbatim in the original
    question. Trusted code, not the agent, performs source locating,
    normalization, and probe construction.
    """

    llm_by_query: dict[str, tuple[QueryAnchor, ...]] = {}
    receipt: CodexAgentReceipt | None = None
    method: Literal[
        "deterministic_extractive",
        "codex_agent_extractive_validated",
    ] = "deterministic_extractive"
    if planner_agent is not None:
        response = planner_agent.parse(
            purpose="extract protocol anchors from Discovery questions",
            prompt_version="discovery-anchor-extraction/v1.0",
            instructions=(
                "Extract only protocol-relevant entities whose exact surface "
                "text occurs verbatim in each question. Do not infer or add a "
                "message, information element, interface, state, specification, "
                "relationship, or other fact absent from the question. Do not "
                "resolve pronouns from other questions or claims. Classify each "
                "surface using only the supplied schema. It is valid to return no "
                "anchors. Return every query_id exactly once."
            ),
            input_payload={
                "questions": [
                    {
                        "query_id": item.discovery_id,
                        "angle": _angle_value(item),
                        "question": item.question,
                    }
                    for item in discovery_questions
                ]
            },
            output_type=LLMAnchorBatch,
        )
        expected_ids = [item.discovery_id for item in discovery_questions]
        returned_ids = [item.query_id for item in response.output.questions]
        if returned_ids != expected_ids:
            raise ValueError(
                "Codex anchor extraction must preserve ordered Discovery query IDs"
            )
        question_by_id = {
            item.discovery_id: item.question for item in discovery_questions
        }
        llm_by_query = {
            item.query_id: _validated_llm_anchors(
                question_by_id[item.query_id],
                item.anchors,
            )
            for item in response.output.questions
        }
        receipt = response.receipt
        method = "codex_agent_extractive_validated"

    return DiscoveryRetrievalPlanV3(
        anchor_extraction_method=method,
        planner_agent_receipt=receipt,
        retrievals=tuple(
            _compile_one(item, llm_by_query.get(item.discovery_id, ()))
            for item in discovery_questions
        ),
    )
__all__ = ["GENERIC_FACET_TERMS", "compile_discovery_retrieval_plan"]

from __future__ import annotations

from types import SimpleNamespace

from specrag.retrieval.planner import (
    compile_discovery_retrieval_plan,
    resolve_discovery_retrieval_plan,
)


def _question(
    discovery_id: str,
    question: str,
    *,
    angle: str = "carrier",
) -> SimpleNamespace:
    return SimpleNamespace(
        discovery_id=discovery_id,
        claim_id=f"C-{discovery_id}",
        angle=angle,
        question=question,
    )


def test_deterministic_planner_preserves_extractable_spans_and_facets() -> None:
    text = (
        "In 3GPP TS 24.501, which information element carries the "
        "Registration Request?"
    )

    plan = compile_discovery_retrieval_plan((_question("Q1", text),))

    item = plan.retrievals[0]
    assert plan.anchor_extraction_method == "deterministic_extractive"
    assert item.planner_action == "planned"
    assert [anchor.canonical for anchor in item.anchors] == [
        "3GPP TS 24.501",
        "Registration Request",
    ]
    assert [facet.name for facet in item.explicit_facets] == [
        "information_element",
    ]
    assert all(
        text[anchor.source_span.start : anchor.source_span.end]
        == anchor.source_span.text
        for anchor in item.anchors
    )
    assert item.retrieval_probes[0].lexical_query == (
        "3GPP TS 24.501 Registration Request information element"
    )


def test_unresolved_reference_falls_back_without_importing_prior_context() -> None:
    questions = (
        _question("Q1", "Which message carries the Registration Request?"),
        _question("Q2", "What message carries this object?"),
    )

    plan = compile_discovery_retrieval_plan(questions)

    second = plan.retrievals[1]
    assert second.context_dependent is True
    assert second.planner_action == "fallback_full_corpus"
    assert second.anchors == ()
    assert second.retrieval_probes[0].lexical_query == questions[1].question
    assert any("unresolved reference" in warning for warning in second.warnings)


def test_explicit_specification_routes_without_document_level_search() -> None:
    text = "What does 3GPP TS 24.501 require for Registration Request?"
    plan = compile_discovery_retrieval_plan((_question("Q1", text),))

    resolved = resolve_discovery_retrieval_plan(plan, SimpleNamespace())

    routes = resolved.retrievals[0].retrieval_probes[0].routes
    assert len(routes) == 1
    assert routes[0].specification == "24.501"
    assert routes[0].source == "explicit_query"
    assert routes[0].weight == 1.0

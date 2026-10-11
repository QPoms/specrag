"""Source-faithful evidence selection and evidence-ID validation."""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping

from ..infrastructure.structured_agent import CodexAgentReceipt, CodexStructuredAgent
from .answer_models import (
    AgentQuestionEvidenceSelectionProposal,
    DiscoveryEvidenceSelection,
    EvidenceSourceStage,
    HypothesisEvidenceAssessment,
    HypothesisSupportStatus,
    QuestionEvidenceSelection,
    SelectedEvidenceGroup,
    SelectedSpecificationEvidence,
)
from .execution import fold_protocol_text as _fold_protocol_text
from .execution_models import ExpandedRetrievalExecution, GroundedRelationEvidence
from .relation_evidence import _RELATION_MESSAGE, _useful_relation_term
def resolve_evidence_id_aliases(
    evidence_ids: tuple[str, ...],
    available_ids: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Resolve only unique rank-shortened evidence IDs from the agent.

    Codex occasionally returns ``HB-D4:2`` instead of the supplied
    ``HB-D4:3gpp_markdown:2``.  Accept that form only when it maps to one and
    only one supplied candidate; ambiguous or unrelated IDs remain invalid.
    """

    resolved: list[str] = []
    unknown: list[str] = []
    aliases: list[tuple[str, str]] = []
    for value in evidence_ids:
        if value in available_ids:
            resolved.append(value)
            continue
        prefix, separator, rank = value.rpartition(":")
        matches = tuple(sorted(
            candidate
            for candidate in available_ids
            if separator
            and rank.isdigit()
            and candidate.startswith(prefix + ":")
            and candidate.rpartition(":")[2] == rank
        ))
        if len(matches) == 1:
            resolved.append(matches[0])
            aliases.append((value, matches[0]))
        else:
            unknown.append(value)
    return tuple(resolved), tuple(unknown), tuple(aliases)


def _compact_agent_evidence_excerpt(
    excerpt: str,
    *,
    focus_terms: tuple[str, ...],
    max_chars: int = 1800,
) -> str:
    """Create a small review window while leaving stored evidence untouched."""

    if len(excerpt) <= max_chars:
        return excerpt
    lines = excerpt.splitlines()
    folded_terms = tuple(
        _fold_protocol_text(term)
        for term in focus_terms
        if _useful_relation_term(term)
    )
    scored: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        folded = _fold_protocol_text(line)
        score = sum(8 for term in folded_terms if term and term in folded)
        if line.strip().startswith("|") and score:
            score += 6
        if any(
            cue in folded
            for cue in (
                "shall include", "may include", "contains", "conveys",
                "message content", "type/reference", "purpose of the",
            )
        ):
            score += 2
        if score:
            scored.append((score, index))

    if not scored:
        return excerpt[:max_chars].rstrip() + "\n[review window truncated]"

    selected_indexes: set[int] = set()
    for _, center in sorted(scored, key=lambda item: (-item[0], item[1])):
        selected_indexes.update(
            range(max(0, center - 2), min(len(lines), center + 3))
        )
        rendered = "\n".join(lines[index] for index in sorted(selected_indexes))
        if len(rendered) >= max_chars:
            break

    parts: list[str] = []
    previous: int | None = None
    for index in sorted(selected_indexes):
        if previous is not None and index != previous + 1:
            parts.append("[...]")
        parts.append(lines[index])
        previous = index
    rendered = "\n".join(parts)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars].rstrip() + "\n[review window truncated]"
    return rendered


def _relation_candidate_name(evidence: GroundedRelationEvidence) -> str:
    message_type = re.search(
        r"Message\s+type\s*:\s*(?P<name>[A-Z][A-Z0-9 /_-]{2,100})",
        evidence.excerpt,
    )
    if message_type is not None:
        return message_type.group("name").strip()
    message = _RELATION_MESSAGE.search(evidence.excerpt)
    if message is not None:
        return message.group(0).strip()
    if evidence.heading_path:
        return evidence.heading_path[-1]
    return evidence.evidence_id


def select_expanded_discovery_evidence(
    execution: ExpandedRetrievalExecution,
    *,
    selection_agent: CodexStructuredAgent,
    max_evidence_groups_per_question: int = 10,
    max_agent_excerpt_chars: int = 1800,
    answer_hypotheses_by_query: Mapping[
        str,
        tuple[Mapping[str, Any], ...],
    ] | None = None,
) -> DiscoveryEvidenceSelection:
    """Select source-faithful evidence groups without authoring answer text."""

    if not 1 <= max_evidence_groups_per_question <= 10:
        raise ValueError("max_evidence_groups_per_question must be between 1 and 10")
    if max_agent_excerpt_chars < 400:
        raise ValueError("max_agent_excerpt_chars must be at least 400")
    if execution.resolved_plan is None:
        return DiscoveryEvidenceSelection(
            status="unavailable",
            max_evidence_groups_per_question=max_evidence_groups_per_question,
            errors=execution.errors or ("resolved retrieval plan is unavailable",),
        )

    first_by_query = {
        task.query_id: task for task in execution.first_stage.tasks
    }
    second_by_query: dict[str, list[GroundedRelationEvidence]] = {}
    for evidence in execution.second_stage_candidates:
        second_by_query.setdefault(evidence.query_id, []).append(evidence)

    hypotheses_by_query = dict(answer_hypotheses_by_query or {})
    instructions = (
        "Select original 3GPP specification evidence for one Discovery question. "
        "Do not write, paraphrase, or summarize an answer. Return only groups of "
        "supplied evidence IDs. Each group must support one semantically distinct, "
        "answer-bearing proposition to the original question, rather than merely "
        "one named protocol entity. For carrier or reachability questions, one "
        "group represents one distinct carrier or path relation. A group may combine "
        "up to four excerpts when they support that same proposition or form a "
        "grounded relation chain, such as carried-object-to-IE and IE-to-message. "
        "Prefer direct normative sentences and exact message-content "
        "table rows. Evidence marked hypothesis_route is original specification "
        "text retrieved from an unverified answer hypothesis; judge only the text, "
        "not the hypothesis that led to it. Treat second-stage material as candidate discovery: select it "
        "only when it establishes a distinct candidate absent from the first-stage "
        "evidence, or when it is required to complete such a relation chain. Do not "
        "select generic term overlap, unrelated tables, headings without supporting "
        "content, or duplicate formulations. Use no protocol knowledge outside the "
        "provided excerpts. Perform two reviews in this single response. First, "
        "assess every supplied answer hypothesis as supported, partial, or "
        "unsupported using only the evidence catalog, cite evidence IDs for "
        "supported or partial assessments, and place hypothesis-backed groups in "
        "evidence_groups. Second, treat the hypotheses as non-exhaustive and inspect "
        "the entire evidence catalog again against the original question. Place "
        "direct, distinct answer-bearing propositions not adequately represented by the hypotheses in "
        "independent_evidence_groups. A hypothesis is never evidence by itself and "
        "must never act as an answer whitelist. Do not create an independent group "
        "for a payload definition, carried message, endpoint, direction, interface, "
        "security condition, or procedure context when it only supports an existing "
        "answer proposition. Attach that supporting evidence to the relevant group. "
        "Different protocol-layer messages may share a group when their excerpts "
        "jointly establish the same carrier or forwarding relation. Do not combine "
        "unrelated alternative carriers or unrelated answer propositions in one group. "
        "Do not require a fixed source/interface/carrier/relay template and do not "
        "infer missing path edges. Keep the combined number of hypothesis-backed and "
        "independent groups within max_evidence_groups. Return fewer than the maximum "
        "when fewer distinct source groups are supported. The system will copy the "
        "corresponding full original excerpts after selection."
    )
    receipts: list[CodexAgentReceipt] = []
    questions: list[QuestionEvidenceSelection] = []
    errors: list[str] = []

    for item in execution.resolved_plan.retrievals:
        first_task = first_by_query[item.query_id]
        anchors = tuple(anchor.canonical for anchor in item.anchors)
        source_catalog: dict[str, SelectedSpecificationEvidence] = {}
        review_catalog: list[dict[str, Any]] = []

        for hit in first_task.hits:
            source_stage: EvidenceSourceStage = (
                "hypothesis_route"
                if hit.hit_id.startswith("HB-")
                else "first_stage"
            )
            source_catalog[hit.hit_id] = SelectedSpecificationEvidence(
                evidence_id=hit.hit_id,
                source_stage=source_stage,
                chunk_id=hit.chunk_id,
                specification=hit.specification,
                version=hit.version,
                source_path=hit.source_path,
                document_sha256=hit.document_sha256,
                heading_path=hit.heading_path,
                line_start=hit.line_start,
                line_end=hit.line_end,
                excerpt=hit.excerpt,
                retrieval_sources=hit.retrieval_sources,
            )
            review_catalog.append({
                "evidence_id": hit.hit_id,
                "source_stage": source_stage,
                "specification": hit.specification,
                "version": hit.version,
                "heading_path": list(hit.heading_path),
                "line_start": hit.line_start,
                "line_end": hit.line_end,
                "retrieval_sources": list(hit.retrieval_sources),
                "review_excerpt": _compact_agent_evidence_excerpt(
                    hit.excerpt,
                    focus_terms=tuple((*anchors, *hit.matched_anchors)),
                    max_chars=max_agent_excerpt_chars,
                ),
            })

        relation_candidates: dict[tuple[str, tuple[str, ...]], list[str]] = {}
        for evidence in second_by_query.get(item.query_id, ()):
            source_catalog[evidence.evidence_id] = SelectedSpecificationEvidence(
                evidence_id=evidence.evidence_id,
                source_stage="second_stage",
                chunk_id=evidence.chunk_id,
                specification=evidence.specification,
                version=evidence.version,
                source_path=evidence.source_path,
                document_sha256=evidence.document_sha256,
                heading_path=evidence.heading_path,
                line_start=evidence.line_start,
                line_end=evidence.line_end,
                excerpt=evidence.excerpt,
                relation_kinds=evidence.relation_kinds,
                bridge_terms=evidence.bridge_terms,
            )
            candidate_name = _relation_candidate_name(evidence)
            candidate_key = (
                candidate_name.casefold(),
                tuple(sorted(value.casefold() for value in evidence.bridge_terms)),
            )
            relation_candidates.setdefault(candidate_key, []).append(
                evidence.evidence_id
            )
            focus_terms = tuple((
                *anchors,
                *evidence.bridge_terms,
                *(term.surface for term in evidence.grounded_terms),
            ))
            review_catalog.append({
                "evidence_id": evidence.evidence_id,
                "source_stage": "second_stage",
                "candidate_name": candidate_name,
                "bridge_terms": list(evidence.bridge_terms),
                "relation_kinds": list(evidence.relation_kinds),
                "specification": evidence.specification,
                "version": evidence.version,
                "heading_path": list(evidence.heading_path),
                "line_start": evidence.line_start,
                "line_end": evidence.line_end,
                "review_excerpt": _compact_agent_evidence_excerpt(
                    evidence.excerpt,
                    focus_terms=focus_terms,
                    max_chars=max_agent_excerpt_chars,
                ),
            })

        candidate_payload = [
            {
                "candidate_id": f"C-{item.query_id}:{index}",
                "candidate_name": key[0],
                "bridge_terms": list(key[1]),
                "evidence_ids": evidence_ids,
            }
            for index, (key, evidence_ids) in enumerate(
                relation_candidates.items(),
                1,
            )
        ]
        answer_hypotheses = hypotheses_by_query.get(item.query_id, ())
        payload = {
            "query_id": item.query_id,
            "claim_id": item.claim_id,
            "angle": item.angle,
            "original_question": item.original_question,
            "original_anchors": list(anchors),
            "explicit_facets": [facet.name for facet in item.explicit_facets],
            "max_evidence_groups": max_evidence_groups_per_question,
            "answer_hypotheses": [dict(value) for value in answer_hypotheses],
            "relation_candidates": candidate_payload,
            "evidence_catalog": review_catalog,
        }
        try:
            response = selection_agent.parse(
                purpose="select source-faithful Discovery evidence",
                prompt_version="discovery-evidence-selection/v1.3",
                instructions=instructions,
                input_payload=payload,
                output_type=AgentQuestionEvidenceSelectionProposal,
            )
        except Exception as exc:
            error = f"{item.query_id}: {type(exc).__name__}: {exc}"
            errors.append(error)
            questions.append(QuestionEvidenceSelection(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                warnings=(error,),
            ))
            continue
        receipts.append(response.receipt)
        if response.output.query_id != item.query_id:
            error = (
                f"{item.query_id}: agent returned query_id "
                f"{response.output.query_id}"
            )
            errors.append(error)
            questions.append(QuestionEvidenceSelection(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                warnings=(error,),
            ))
            continue

        warnings: list[str] = []
        known_hypotheses = {
            str(value.get("hypothesis_id")): value
            for value in answer_hypotheses
            if value.get("hypothesis_id")
        }
        proposed_assessments = {
            value.hypothesis_id: value
            for value in response.output.hypothesis_assessments
            if value.hypothesis_id in known_hypotheses
        }
        for value in response.output.hypothesis_assessments:
            if value.hypothesis_id not in known_hypotheses:
                warnings.append(
                    "ignored assessment for unknown hypothesis: "
                    + value.hypothesis_id
                )

        assessments: list[HypothesisEvidenceAssessment] = []
        for hypothesis_id in known_hypotheses:
            proposed = proposed_assessments.get(hypothesis_id)
            if proposed is None:
                warnings.append(
                    f"agent omitted hypothesis assessment: {hypothesis_id}"
                )
                assessments.append(HypothesisEvidenceAssessment(
                    hypothesis_id=hypothesis_id,
                    status="unassessed",
                ))
                continue
            evidence_ids = tuple(dict.fromkeys(
                value
                for value in proposed.evidence_ids
                if value in source_catalog
            ))
            unknown_evidence = tuple(
                value
                for value in proposed.evidence_ids
                if value not in source_catalog
            )
            if unknown_evidence:
                warnings.append(
                    f"{hypothesis_id} assessment ignored unknown evidence IDs: "
                    + ", ".join(unknown_evidence)
                )
            status: HypothesisSupportStatus = proposed.status
            if status in {"supported", "partial"} and not evidence_ids:
                warnings.append(
                    f"{hypothesis_id} was marked {status} without valid evidence"
                )
                status = "unassessed"
            if status == "unsupported":
                evidence_ids = ()
            assessments.append(HypothesisEvidenceAssessment(
                hypothesis_id=hypothesis_id,
                status=status,
                evidence_ids=evidence_ids,
            ))

        groups: list[SelectedEvidenceGroup] = []
        group_indexes: dict[tuple[str, ...], int] = {}
        for proposed in response.output.evidence_groups[
            :max_evidence_groups_per_question
        ]:
            evidence_ids = tuple(dict.fromkeys(proposed.evidence_ids))
            unknown = tuple(
                evidence_id
                for evidence_id in evidence_ids
                if evidence_id not in source_catalog
            )
            if unknown:
                warnings.append(
                    "skipped group with unknown evidence IDs: " + ", ".join(unknown)
                )
                continue
            hypothesis_ids = tuple(dict.fromkeys(
                hypothesis_id
                for hypothesis_id in proposed.hypothesis_ids
                if hypothesis_id in known_hypotheses
            ))
            unknown_hypotheses = tuple(
                hypothesis_id
                for hypothesis_id in proposed.hypothesis_ids
                if hypothesis_id not in known_hypotheses
            )
            if unknown_hypotheses:
                warnings.append(
                    "ignored unknown hypothesis IDs in evidence group: "
                    + ", ".join(unknown_hypotheses)
                )
            group_key = tuple(sorted(evidence_ids))
            existing_index = group_indexes.get(group_key)
            if existing_index is not None:
                existing = groups[existing_index]
                groups[existing_index] = existing.model_copy(update={
                    "hypothesis_ids": tuple(dict.fromkeys((
                        *existing.hypothesis_ids,
                        *hypothesis_ids,
                    ))),
                })
                continue
            group_indexes[group_key] = len(groups)
            selected = tuple(source_catalog[value] for value in evidence_ids)
            groups.append(SelectedEvidenceGroup(
                group_id=f"E-{item.query_id}:{len(groups) + 1}",
                source_stages=tuple(dict.fromkeys(
                    value.source_stage for value in selected
                )),
                hypothesis_ids=hypothesis_ids,
                selection_basis="hypothesis_assessment",
                evidence=selected,
            ))

        # A supported hypothesis must not disappear merely because the agent
        # forgot to duplicate its cited evidence in evidence_groups. Reuse or
        # append the agent's own cited IDs; this is a soft completeness guard,
        # not a protocol-specific path validator.
        for assessment in assessments:
            if assessment.status not in {"supported", "partial"}:
                continue
            if any(
                assessment.hypothesis_id in group.hypothesis_ids
                for group in groups
            ):
                continue
            assessment_ids = tuple(sorted(assessment.evidence_ids))
            matching_index = next(
                (
                    index
                    for index, group in enumerate(groups)
                    if set(assessment.evidence_ids).issubset(
                        evidence.evidence_id for evidence in group.evidence
                    )
                ),
                None,
            )
            if matching_index is not None:
                existing = groups[matching_index]
                groups[matching_index] = existing.model_copy(update={
                    "hypothesis_ids": tuple(dict.fromkeys((
                        *existing.hypothesis_ids,
                        assessment.hypothesis_id,
                    ))),
                })
                continue
            if len(groups) >= max_evidence_groups_per_question:
                warnings.append(
                    "supported hypothesis could not be added because the evidence "
                    f"group limit was reached: {assessment.hypothesis_id}"
                )
                continue
            selected = tuple(
                source_catalog[value] for value in assessment.evidence_ids
            )
            groups.append(SelectedEvidenceGroup(
                group_id=f"E-{item.query_id}:{len(groups) + 1}",
                source_stages=tuple(dict.fromkeys(
                    value.source_stage for value in selected
                )),
                hypothesis_ids=(assessment.hypothesis_id,),
                selection_basis="hypothesis_assessment",
                evidence=selected,
            ))
            group_indexes[assessment_ids] = len(groups) - 1

        # Hypotheses are deliberately non-exhaustive. Keep direct answers found
        # by the agent's independent second review even when no hypothesis named
        # them. This runs in the same model call and stores only source IDs.
        for proposed in response.output.independent_evidence_groups:
            if len(groups) >= max_evidence_groups_per_question:
                warnings.append(
                    "independent evidence group limit reached; remaining groups "
                    "were not added"
                )
                break
            evidence_ids = tuple(dict.fromkeys(proposed.evidence_ids))
            unknown = tuple(
                evidence_id
                for evidence_id in evidence_ids
                if evidence_id not in source_catalog
            )
            if unknown:
                warnings.append(
                    "skipped independent group with unknown evidence IDs: "
                    + ", ".join(unknown)
                )
                continue
            group_key = tuple(sorted(evidence_ids))
            if group_key in group_indexes:
                continue
            selected = tuple(source_catalog[value] for value in evidence_ids)
            group_indexes[group_key] = len(groups)
            groups.append(SelectedEvidenceGroup(
                group_id=f"E-{item.query_id}:{len(groups) + 1}",
                source_stages=tuple(dict.fromkeys(
                    value.source_stage for value in selected
                )),
                hypothesis_ids=(),
                selection_basis="original_question_independent",
                evidence=selected,
            ))
        questions.append(QuestionEvidenceSelection(
            query_id=item.query_id,
            claim_id=item.claim_id,
            angle=item.angle,
            original_question=item.original_question,
            status="completed" if groups else "no_evidence",
            hypothesis_assessments=tuple(assessments),
            evidence_groups=tuple(groups),
            warnings=tuple(warnings),
        ))

    unavailable_count = sum(value.status == "unavailable" for value in questions)
    status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if questions and unavailable_count == len(questions)
        else "partial"
        if unavailable_count or errors
        else "completed"
    )
    return DiscoveryEvidenceSelection(
        status=status,
        max_evidence_groups_per_question=max_evidence_groups_per_question,
        agent_receipts=tuple(receipts),
        questions=tuple(questions),
        errors=tuple(errors),
    )



__all__ = ['resolve_evidence_id_aliases', 'select_expanded_discovery_evidence']

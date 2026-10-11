"""Final answer synthesis from preselected source-evidence groups."""

from __future__ import annotations

import re
from typing import Any, Literal

from ..infrastructure.structured_agent import CodexAgentReceipt, CodexStructuredAgent
from .answer_models import (
    AgentQuestionAnswerProposal,
    DiscoveryEvidenceSelection,
    EvidenceAnswerSummary,
    EvidenceBackedAnswer,
    QuestionAnswerSummary,
)
from .evidence_selection import resolve_evidence_id_aliases
def summarize_selected_discovery_evidence(
    selection: DiscoveryEvidenceSelection,
    *,
    summary_agent: CodexStructuredAgent,
    max_answers_per_question: int = 10,
) -> EvidenceAnswerSummary:
    """Turn selected source groups into final, evidence-backed answers."""

    if not 1 <= max_answers_per_question <= 10:
        raise ValueError("max_answers_per_question must be between 1 and 10")
    if selection.status == "unavailable" and not selection.questions:
        return EvidenceAnswerSummary(
            status="unavailable",
            max_answers_per_question=max_answers_per_question,
            errors=selection.errors or ("selected evidence is unavailable",),
        )

    instructions = (
        "Produce the final semantic answer to one 3GPP Discovery question from "
        "the supplied selected source-evidence groups. Evidence groups are "
        "candidate support structures, not a list of answers, and their count "
        "does not determine the answer count. Return at most the requested "
        "maximum and return fewer when fewer distinct answers are supported. "
        "One answer must be one proposition that directly answers the original "
        "question. For carrier or reachability questions, one answer represents "
        "one independently answer-bearing carrier or path relation. Merge groups "
        "and excerpts that describe the same relation. The queried carried object, "
        "payload name, information element, endpoint, direction, interface, "
        "security condition, and procedure context are supporting facts for the "
        "relevant relation; do not emit them as separate answers merely because "
        "they appear in a separate excerpt or group. An excerpt that only defines "
        "or sends the queried object is not a carrier answer unless it establishes "
        "the carrier relation asked by the question. For other Discovery angles, "
        "apply the same principle: distinguish direct answers from facts that only "
        "qualify or support those answers. Do not require a fixed number of answers "
        "or a fixed end-to-end path schema. Every answer must cite one to eight "
        "supplied evidence IDs copied exactly character-for-character; never "
        "shorten, reconstruct, or rename an evidence ID. Combine excerpts only when every link needed for "
        "the stated relation is present. Use no protocol knowledge outside the "
        "provided excerpts, do not infer missing path edges, and do not introduce "
        "any message, IE, container, condition, direction, or fact absent from the "
        "evidence."
    )
    receipts: list[CodexAgentReceipt] = []
    summaries: list[QuestionAnswerSummary] = []
    errors: list[str] = list(selection.errors)

    for item in selection.questions:
        if item.status == "unavailable":
            summaries.append(QuestionAnswerSummary(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                evidence_status="unavailable",
                warnings=item.warnings,
            ))
            continue
        if not item.evidence_groups:
            summaries.append(QuestionAnswerSummary(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="no_answers",
                warnings=item.warnings,
            ))
            continue

        available_ids: set[str] = set()
        evidence_groups: list[dict[str, Any]] = []
        for group in item.evidence_groups:
            evidence_payload: list[dict[str, Any]] = []
            for evidence in group.evidence:
                available_ids.add(evidence.evidence_id)
                evidence_payload.append({
                    "evidence_id": evidence.evidence_id,
                    "source_stage": evidence.source_stage,
                    "specification": evidence.specification,
                    "version": evidence.version,
                    "heading_path": list(evidence.heading_path),
                    "line_start": evidence.line_start,
                    "line_end": evidence.line_end,
                    "excerpt": evidence.excerpt,
                })
            evidence_groups.append({
                "group_id": group.group_id,
                "selection_basis": group.selection_basis,
                "hypothesis_ids": list(group.hypothesis_ids),
                "evidence": evidence_payload,
            })

        payload = {
            "query_id": item.query_id,
            "claim_id": item.claim_id,
            "angle": item.angle,
            "original_question": item.original_question,
            "max_answers": max_answers_per_question,
            "selected_evidence_groups": evidence_groups,
        }
        try:
            response = summary_agent.parse(
                purpose="summarize selected Discovery evidence",
                prompt_version="discovery-selected-evidence-answer-summary/v1.0",
                instructions=instructions,
                input_payload=payload,
                output_type=AgentQuestionAnswerProposal,
            )
        except Exception as exc:
            error = f"{item.query_id}: {type(exc).__name__}: {exc}"
            errors.append(error)
            summaries.append(QuestionAnswerSummary(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                evidence_status="unavailable",
                warnings=(*item.warnings, error),
            ))
            continue

        receipts.append(response.receipt)
        if response.output.query_id != item.query_id:
            error = (
                f"{item.query_id}: agent returned query_id "
                f"{response.output.query_id}"
            )
            errors.append(error)
            summaries.append(QuestionAnswerSummary(
                query_id=item.query_id,
                claim_id=item.claim_id,
                angle=item.angle,
                original_question=item.original_question,
                status="unavailable",
                evidence_status="unavailable",
                warnings=(*item.warnings, error),
            ))
            continue

        answers: list[EvidenceBackedAnswer] = []
        warnings: list[str] = list(item.warnings)
        seen_statements: set[str] = set()
        for proposed in response.output.answers[:max_answers_per_question]:
            resolved_evidence_ids, unknown_ids, aliases = (
                resolve_evidence_id_aliases(
                    proposed.evidence_ids,
                    available_ids,
                )
            )
            if aliases:
                warnings.append(
                    "normalized abbreviated evidence IDs: "
                    + ", ".join(
                        f"{source} -> {target}"
                        for source, target in aliases
                    )
                )
            if unknown_ids:
                warnings.append(
                    "skipped answer with unknown evidence IDs: "
                    + ", ".join(unknown_ids)
                )
                continue
            key = re.sub(r"\s+", " ", proposed.statement).strip().casefold()
            if key in seen_statements:
                warnings.append("skipped duplicate answer statement")
                continue
            seen_statements.add(key)
            answers.append(proposed.model_copy(update={
                "answer_id": f"A-{item.query_id}:{len(answers) + 1}",
                "evidence_ids": tuple(dict.fromkeys(resolved_evidence_ids)),
            }))
        summaries.append(QuestionAnswerSummary(
            query_id=item.query_id,
            claim_id=item.claim_id,
            angle=item.angle,
            original_question=item.original_question,
            status="completed" if answers else "no_answers",
            answers=tuple(answers),
            warnings=tuple(warnings),
        ))

    unavailable_count = sum(item.status == "unavailable" for item in summaries)
    status: Literal["completed", "partial", "unavailable"] = (
        "unavailable"
        if summaries and unavailable_count == len(summaries)
        else "partial"
        if unavailable_count or errors or selection.status == "partial"
        else "completed"
    )
    return EvidenceAnswerSummary(
        status=status,
        max_answers_per_question=max_answers_per_question,
        agent_receipts=tuple(receipts),
        questions=tuple(summaries),
        errors=tuple(dict.fromkeys(errors)),
    )



__all__ = ['summarize_selected_discovery_evidence']

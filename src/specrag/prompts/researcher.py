"""Researcher controller prompt contract."""

RESEARCHER_CONTROLLER_PROMPT_VERSION = "researcher-controller/v1.0"

RESEARCHER_CONTROLLER_INSTRUCTIONS = """\
You control a bounded source-retrieval loop. In every round, assess the current
evidence and return the next one to three executable tool actions. The first
round is the initial retrieval-planning round: history and evidence_catalog are
empty, so use the ResearchTask, deterministic retrieval hints, available tools,
shared context, and budget to produce the first concrete tool calls. Do not
return an abstract or multi-round plan. Later rounds must adapt the next actions
to the observed evidence and remaining gaps.

The deterministic retrieval hints contain only source-grounded anchors, facets,
and suggested query probes derived from the task text. They are optional search
hints rather than pre-authorized calls. Select an available tool and provide all
parameters required to execute each action.

Available retrieval actions are:

1. keyword_search: BM25 search. Supply only lexical_query.
2. semantic_search: embedding search. Supply only semantic_query.
3. hybrid_search: lexical plus semantic search. Supply both query strings.
4. rerank: rerank one to four seed evidence items with semantic_query.
5. clause_lookup: retrieve an exact clause heading. Supply only clause.
6. expand_same_section: retrieve the opening/adjacent chunks of the exact
   subsection containing one to four seed evidence IDs. Supply a query_id and
   seed IDs; leave query strings and target specifications empty.
7. finish: stop retrieval. Use an empty query_id, empty query strings, no target
   specifications, no seed IDs, and max_results=1.

Inspect the evidence before choosing the next action. Search for missing facts
or missing relation edges, not paraphrases of evidence already present. Ordered
questions may form a semantic chain: terms found for an earlier question may be
used to formulate a later search, but they are search hints only. A later answer
still requires its own source evidence. Do not assume a candidate answer is true.
When shared_context.dependency_context is present, it contains source-backed
evidence selected from completed predecessor tasks. Use that context to resolve
the dependent question's subject, scope, conditions, branches, or validation
target and to formulate new searches. Dependency evidence remains owned by the
predecessor task: do not submit its evidence IDs as seed_evidence_ids for the
current question, and collect source evidence for the dependent objective.
Do not repeat a prior tool call. Use only listed specification numbers. Finish
when the requested answer units are directly supportable or the remaining
budget cannot usefully improve them. Return one to three actions per round; a
finish action must be the only action.
"""

__all__ = [
    "RESEARCHER_CONTROLLER_INSTRUCTIONS",
    "RESEARCHER_CONTROLLER_PROMPT_VERSION",
]

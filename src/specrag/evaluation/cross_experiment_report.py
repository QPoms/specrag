"""Markdown rendering for cross-experiment descriptive analysis."""

from __future__ import annotations

from .cross_experiment import CrossExperimentAnalysis, PairedScoreSlice


def _display_score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1f}"


def _display_delta(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.1f}"


def _append_slice_rows(
    lines: list[str],
    slices: tuple[tuple[str, PairedScoreSlice], ...],
) -> None:
    for name, score_slice in slices:
        lines.append(
            f"| {name} "
            f"| {score_slice.paired_score_count}/{score_slice.expected_case_count} "
            f"| {_display_score(score_slice.left_average_score)} "
            f"| {_display_score(score_slice.right_average_score)} "
            f"| {_display_delta(score_slice.mean_delta_right_minus_left)} "
            f"| {_display_delta(score_slice.median_delta_right_minus_left)} "
            f"| {score_slice.right_higher_count} | {score_slice.equal_count} "
            f"| {score_slice.left_higher_count} |"
        )


def render_cross_experiment_markdown(
    analysis: CrossExperimentAnalysis,
    *,
    top_deltas: int = 10,
) -> str:
    """Render a stable Markdown report without deriving success decisions."""

    if top_deltas < 1:
        raise ValueError("top_deltas must be at least one")
    lines = [
        "# SpecRAG Cross-Experiment Analysis",
        "",
        "## Provenance",
        "",
        f"- Benchmark: `{analysis.benchmark_id}` / `{analysis.benchmark_version}`",
        f"- Judge: `{analysis.judge_model}` / `{analysis.judge_prompt_version}`",
        f"- Trial count per model and question: `{analysis.trial_count}`",
        "",
        "| Experiment | Source | SHA-256 |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| `{item.experiment_id}` | `{item.path}` | `{item.sha256}` |"
        for item in analysis.sources
    )
    lines.extend([
        "",
        "## Model overview",
        "",
        "| Experiment / model | Overall | Outcome scores | Judge-scored answers | Simple | Complex | No answer | Workflow failed | Judge failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for item in analysis.model_observations:
        summary = item.score_summary
        lines.append(
            f"| `{item.experiment_id}` / `{item.model_id}` "
            f"| {_display_score(summary.overall.observed_average_score)} "
            f"| {summary.overall.observed_score_count}/{summary.overall.expected_count} "
            f"| {summary.scored_answer_count}/{summary.expected_case_count} "
            f"| {_display_score(summary.simple.observed_average_score)} "
            f"| {_display_score(summary.complex.observed_average_score)} "
            f"| {summary.no_answer_count} | {summary.workflow_failed_count} "
            f"| {summary.judge_failed_count} |"
        )

    for index, pair in enumerate(analysis.pairwise_comparisons, start=1):
        left = f"{pair.left_experiment_id} / {pair.left_model_id}"
        right = f"{pair.right_experiment_id} / {pair.right_model_id}"
        lines.extend([
            "",
            f"## Pair {index}: `{right}` minus `{left}`",
            "",
            "| Slice | Paired | Left avg | Right avg | Mean delta | Median delta | Right higher | Equal | Left higher |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        _append_slice_rows(lines, (
            ("Overall", pair.overall),
            ("Simple", pair.simple),
            ("Complex", pair.complex),
        ))
        lines.extend([
            "",
            "### Joint answered cases",
            "",
            "This view excludes cases where either side did not receive a normal `scored` Judge outcome.",
            "",
            "| Slice | Paired | Left avg | Right avg | Mean delta | Median delta | Right higher | Equal | Left higher |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        _append_slice_rows(lines, (
            ("Overall", pair.answered_only_overall),
            ("Simple", pair.answered_only_simple),
            ("Complex", pair.answered_only_complex),
        ))
        paired = tuple(
            item
            for item in pair.case_deltas
            if item.delta_right_minus_left is not None
        )
        positive = sorted(
            (item for item in paired if float(item.delta_right_minus_left) > 0),
            key=lambda item: (-float(item.delta_right_minus_left), item.question_id),
        )[:top_deltas]
        negative = sorted(
            (item for item in paired if float(item.delta_right_minus_left) < 0),
            key=lambda item: (float(item.delta_right_minus_left), item.question_id),
        )[:top_deltas]
        lines.extend([
            "",
            "### Largest score changes",
            "",
            "| Direction | Question | Difficulty | Left | Right | Delta | Judge statuses (left -> right) |",
            "|---|---|---|---:|---:|---:|---|",
        ])
        for direction, items in (
            ("Right higher", positive),
            ("Left higher", negative),
        ):
            for item in items:
                lines.append(
                    f"| {direction} | `{item.question_id}` | {item.difficulty} "
                    f"| {_display_score(item.left_score)} "
                    f"| {_display_score(item.right_score)} "
                    f"| {_display_delta(item.delta_right_minus_left)} "
                    f"| `{item.left_judge_status}` -> `{item.right_judge_status}` |"
                )
        if not positive and not negative:
            lines.append(
                "| No non-zero paired changes | N/A | N/A | N/A | N/A | N/A | N/A |"
            )

    observation_labels = tuple(
        f"{item.experiment_id} / {item.model_id}"
        for item in analysis.model_observations
    )
    lines.extend([
        "",
        "## Per-question matrix",
        "",
        "| Question | Difficulty | " + " | ".join(observation_labels) + " |",
        "|---|---|" + "---|" * len(observation_labels),
    ])
    for case in analysis.cases:
        cells = tuple(
            f"{_display_score(item.score)} "
            f"(`{item.workflow_status}` / `{item.judge_status}`)"
            for item in case.observations
        )
        lines.append(
            f"| `{case.question_id}` | {case.difficulty} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend([
        "",
        "## Interpretation boundaries",
        "",
        "- These are descriptive statistics from one trial per model and question.",
        "- Compatibility checks cover the Benchmark, Judge, question order, and difficulty present in `comparison.json`; other frozen Workflow and environment variables must be verified from the experiment manifests.",
        "- The report does not define Task Success, a pass threshold, Pass@k/Pass^k, reliability, or statistical significance.",
        "- `no_answer` contributes its explicit zero score but remains visible as a distinct Judge status.",
        "- End-to-end Judge scores do not identify the quality or causal contribution of an individual workflow stage.",
        "",
    ])
    return "\n".join(lines)


__all__ = ["render_cross_experiment_markdown"]

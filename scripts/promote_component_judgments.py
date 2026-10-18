"""Promote manually audited component judgments to reviewed JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pydantic import BaseModel

from specrag.evaluation.component_evidence_models import (
    FinalEvidenceJudgment,
    ResearcherEvidenceJudgment,
    RetrievalCallJudgment,
    SynthesizerJudgment,
)


MODELS: dict[str, type[BaseModel]] = {
    "retriever": RetrievalCallJudgment,
    "researcher": ResearcherEvidenceJudgment,
    "judge": FinalEvidenceJudgment,
    "synthesizer": SynthesizerJudgment,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record an explicit manual-review promotion decision."
    )
    parser.add_argument("component", choices=tuple(MODELS))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument(
        "--audit-complete",
        action="store_true",
        help="affirm that semantic labels were manually inspected and accepted",
    )
    return parser


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != text:
            raise FileExistsError(f"refusing to overwrite reviewed judgments: {path}")
        return
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.audit_complete:
        raise ValueError("manual promotion requires --audit-complete")
    model = MODELS[args.component]
    promoted = []
    for line_number, raw_line in enumerate(
        args.input.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        value.update({
            "review_status": "reviewed",
            "reviewed_by": args.reviewed_by,
            "reviewed_at": args.reviewed_at,
        })
        try:
            promoted.append(model.model_validate(value))
        except Exception as exc:
            raise ValueError(f"invalid judgment at line {line_number}: {exc}") from exc
    if not promoted:
        raise ValueError("judgment input is empty")
    serialized = "".join(item.model_dump_json() + "\n" for item in promoted)
    _write_once(args.output, serialized)
    print(json.dumps({
        "component": args.component,
        "reviewed_count": len(promoted),
        "reviewed_by": args.reviewed_by,
        "reviewed_at": args.reviewed_at,
        "output": args.output.as_posix(),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())

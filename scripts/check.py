"""Run the canonical local quality gate for Specrag."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from check_qa_dataset import main as check_qa_dataset
from check_structure import main as check_structure


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.dont_write_bytecode = True
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    structure_status = check_structure()
    if structure_status != 0:
        return structure_status
    qa_status = check_qa_dataset()
    if qa_status != 0:
        return qa_status
    return int(pytest.main(["-p", "no:cacheprovider", "-q"]))


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail fast when the repository drifts away from its package boundaries."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "specrag"

ALLOWED_SOURCE_ENTRIES = {
    "__init__.py",
    "agents",
    "cli.py",
    "config.py",
    "evaluation",
    "evidence",
    "infrastructure",
    "memory",
    "models",
    "orchestration",
    "prompts",
    "retrieval",
    "settings.py",
    "tools",
}

FORBIDDEN_PATHS = (
    "src/specagent",
    "src/specrag/screening",
    "src/specrag/flow",
    "src/specrag/nodes",
    "src/specrag/rag",
    "src/specrag/testgen",
    "src/specrag/eval",
    "threatmodel",
)

FORBIDDEN_IMPORT_FRAGMENTS = (
    "specagent.",
    "specrag.screening",
    "specrag.flow",
    "specrag.nodes",
    "specrag.testgen",
    "specrag.eval",
)

VERSIONED_MODULE = re.compile(
    r"(?:^|_)(?:v\d+(?:_\d+)*|new|old|final|backup|temp)(?:_|$)",
    re.IGNORECASE,
)

DEFAULT_MAX_LINES = 600
LINE_RATCHETS: dict[str, int] = {}

ALLOWED_IMPORTS = {
    "models": {"models"},
    "infrastructure": {"config", "models", "prompts", "infrastructure"},
    "retrieval": {"config", "models", "infrastructure", "retrieval"},
    "tools": {"config", "models", "infrastructure", "retrieval", "memory", "tools"},
    "evidence": {"config", "models", "infrastructure", "retrieval", "evidence"},
    "memory": {"config", "models", "infrastructure", "memory"},
    "agents": {
        "config", "models", "infrastructure", "retrieval", "tools",
        "evidence", "memory", "agents",
    },
    "orchestration": {
        "config", "models", "infrastructure", "retrieval", "tools",
        "evidence", "memory", "agents", "orchestration",
    },
    "evaluation": {
        "config", "models", "infrastructure", "retrieval", "tools",
        "evidence", "memory", "agents", "orchestration", "evaluation",
    },
    "prompts": {"prompts"},
}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _target_package(path: Path, node: ast.Import | ast.ImportFrom) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            parts = alias.name.split(".")
            if parts[0] == "specrag" and len(parts) > 1:
                return parts[1]
        return None

    module_parts = node.module.split(".") if node.module else []
    if node.level == 0:
        if module_parts[:1] == ["specrag"] and len(module_parts) > 1:
            return module_parts[1]
        return None

    package_parts = ["specrag", *path.relative_to(SOURCE_ROOT).parts[:-1]]
    keep = len(package_parts) - (node.level - 1)
    resolved = package_parts[:keep] + module_parts
    return resolved[1] if len(resolved) > 1 else "config"


def _imports_langgraph(node: ast.Import | ast.ImportFrom) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.split(".", 1)[0] == "langgraph" for alias in node.names)
    return bool(node.module and node.module.split(".", 1)[0] == "langgraph")


def main() -> int:
    errors: list[str] = []

    for relative in FORBIDDEN_PATHS:
        if (ROOT / relative).exists():
            errors.append(f"forbidden legacy path exists: {relative}")

    actual_entries = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.name != "__pycache__"
    }
    for name in sorted(actual_entries - ALLOWED_SOURCE_ENTRIES):
        errors.append(f"unapproved top-level source entry: src/specrag/{name}")

    for path in SOURCE_ROOT.rglob("*.py"):
        relative = _relative(path)
        parts = path.relative_to(SOURCE_ROOT).parts
        source_package = parts[0] if len(parts) > 1 else None
        if path.name != "__init__.py" and VERSIONED_MODULE.search(path.stem):
            errors.append(f"versioned or temporary module name: {relative}")

        text = path.read_text(encoding="utf-8")
        for fragment in FORBIDDEN_IMPORT_FRAGMENTS:
            if fragment in text:
                errors.append(f"legacy import {fragment!r} in {relative}")

        line_count = len(text.splitlines())
        ceiling = LINE_RATCHETS.get(relative, DEFAULT_MAX_LINES)
        if line_count > ceiling:
            errors.append(
                f"module grew beyond {ceiling} lines ({line_count}): {relative}"
            )

        allowed = ALLOWED_IMPORTS.get(source_package or "")
        if allowed is None:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"syntax error in {relative}: {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if source_package != "orchestration" and _imports_langgraph(node):
                errors.append(
                    f"LangGraph import outside orchestration: {relative}:{node.lineno}"
                )
            target = _target_package(path, node)
            if target is not None and target not in allowed:
                errors.append(
                    f"forbidden dependency {source_package} -> {target} "
                    f"in {relative}:{node.lineno}"
                )

    if errors:
        print("Structure check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Structure check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

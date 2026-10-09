"""Build a catalog for every locally retained 3GPP specification version."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "data" / "3gpp"
CATALOG = CORPUS_ROOT / "catalog.jsonl"
MANIFEST = CORPUS_ROOT / "manifest.json"
SPECIFICATION = re.compile(r"^\d{2}\.\d{3}$")
VERSION = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _repo_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec_dir in sorted(path for path in CORPUS_ROOT.iterdir() if path.is_dir()):
        if SPECIFICATION.fullmatch(spec_dir.name) is None:
            continue
        for version_dir in sorted(path for path in spec_dir.iterdir() if path.is_dir()):
            version_match = VERSION.fullmatch(version_dir.name)
            if version_match is None:
                continue
            markdown = sorted(version_dir.glob("*.md"))
            if len(markdown) > 1:
                raise ValueError(
                    f"expected at most one primary Markdown file in {version_dir}"
                )
            files = sorted(path for path in version_dir.rglob("*") if path.is_file())
            rows.append({
                "specification": spec_dir.name,
                "version": ".".join(version_match.groups()),
                "markdown_path": _repo_path(markdown[0]) if markdown else None,
                "files": [
                    {
                        "path": _repo_path(path),
                        "format": path.suffix.lower().removeprefix("."),
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in files
                ],
            })
    return rows


def main() -> None:
    rows = build()
    CATALOG.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "specrag-3gpp-corpus/v1",
        "layout": "data/3gpp/<specification>/v<version>/",
        "historical_versions_retained": True,
        "catalog": _repo_path(CATALOG),
        "specification_count": len({row["specification"] for row in rows}),
        "version_count": len(rows),
        "file_count": sum(len(row["files"]) for row in rows),
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} version records to {CATALOG}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4
import zipfile

from specrag.infrastructure import threegpp
from specrag.models import SpecificationSourceCandidate


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            value, self.payload = self.payload, b""
            return value
        value, self.payload = self.payload[:size], self.payload[size:]
        return value


def test_official_provider_decodes_revision_codes_and_selects_latest(monkeypatch) -> None:
    listing = b"""
    <html><body>
      <a href="24501-h00.zip">old</a>
      <a href="24501-h71.zip">middle</a>
      <a href="24501-hi1.zip">latest</a>
      <a href="24501-i00.zip">other Release</a>
    </body></html>
    """
    monkeypatch.setattr(threegpp, "urlopen", lambda *_args, **_kwargs: _Response(listing))

    provider = threegpp.ThreeGppArchiveProvider(specifications=("24.501",))
    result = provider.discover((17,))

    assert len(result) == 1
    assert result[0].specification == "24.501"
    assert result[0].release == 17
    assert result[0].version == "17.18.1"
    assert result[0].source_uri.endswith("/24.501/24501-hi1.zip")
    assert result[0].target_source_path.endswith("/24.501/v17.18.1/24501-hi1.md")


def _docx_bytes() -> bytes:
    document = """<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>5 Registration</w:t></w:r></w:p>
        <w:p><w:r><w:t>The UE registers.</w:t></w:r></w:p>
        <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Field</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
      </w:body>
    </w:document>"""
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("word/document.xml", document)
    return output.getvalue()


def test_materializer_downloads_extracts_and_writes_canonical_markdown(monkeypatch) -> None:
    tmp_path = Path(".local") / f"test-threegpp-materializer-{uuid4().hex}"
    tmp_path.mkdir(parents=True)
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as outer:
        outer.writestr("documents/24501-hi1.docx", _docx_bytes())
    payload = archive.getvalue()
    monkeypatch.setattr(threegpp, "urlopen", lambda *_args, **_kwargs: _Response(payload))
    candidate = SpecificationSourceCandidate(
        specification="24.501",
        version="17.18.1",
        release=17,
        source_uri="https://www.3gpp.org/ftp/Specs/archive/24_series/24.501/24501-hi1.zip",
        target_source_path="data/3gpp/24.501/v17.18.1/24501-hi1.md",
        source_format="zip",
    )

    source = threegpp.ThreeGppSourceMaterializer(repository_root=tmp_path).materialize(candidate)

    assert source.document_sha256
    markdown = tmp_path / source.source_path
    assert markdown.read_text(encoding="utf-8") == (
        "# 5 Registration\n\nThe UE registers.\n\n"
        "| Field | Value |\n| --- | --- |\n"
    )
    assert markdown.with_suffix(".zip").is_file()
    assert markdown.with_suffix(".docx").is_file()


def test_zip_extraction_rejects_path_traversal() -> None:
    tmp_path = Path(".local/test-threegpp-zip")
    tmp_path.mkdir(parents=True, exist_ok=True)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "no")

    try:
        threegpp._extract_zip(archive, tmp_path / "out")
    except ValueError as exc:
        assert "unsafe ZIP member" in str(exc)
    else:
        raise AssertionError("unsafe ZIP member was extracted")

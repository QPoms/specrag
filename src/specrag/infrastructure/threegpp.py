"""Official 3GPP FTP archive discovery and source materialization.

The adapter is deliberately explicit: it is only called by ``corpus sync
--official``.  Discovery reads the public directory listings under the
official 3GPP archive, while materialization downloads one ZIP, extracts it
to a version-isolated directory, and converts the contained Word document to
Markdown without replacing any existing version.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import zipfile
from uuid import uuid4

from specrag.config import REPO_ROOT
from specrag.models import CorpusSourceDocument, SpecificationSourceCandidate


_SPECIFICATION = re.compile(r"^(\d{2})\.(\d{3})$")
_ARCHIVE_FILE = re.compile(r"^(\d{5})-([0-9A-Za-z]{3})\.zip$", re.IGNORECASE)
_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = "{" + _WORD_NS + "}"


class _DirectoryListing(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def _component(value: str) -> int:
    if value.isdigit():
        return int(value)
    return 10 + ord(value.lower()) - ord("a")


def _revision_version(revision: str) -> tuple[int, str] | None:
    """Decode 3GPP's three-character revision code (e.g. ``hi1``)."""

    if len(revision) != 3 or not revision.isalnum():
        return None
    release = _component(revision[0])
    if release < 1:
        return None
    return release, f"{release}.{_component(revision[1])}.{_component(revision[2])}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _safe_member_path(root: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe ZIP member path: {member_name}")
    target = (root / Path(*relative.parts)).resolve()
    target.relative_to(root.resolve())
    return target


def _extract_zip(archive: Path, destination: Path) -> tuple[Path, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            target = _safe_member_path(destination, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as input_handle, target.open("wb") as output:
                shutil.copyfileobj(input_handle, output)
            extracted.append(target)
    return tuple(extracted)


def _text(element: ET.Element) -> str:
    return "".join(element.itertext()).replace("\xa0", " ").strip()


def _word_heading(paragraph: ET.Element) -> int | None:
    style = paragraph.find(f"{_W}pPr/{_W}pStyle")
    value = style.get(f"{_W}val", "") if style is not None else ""
    match = re.search(r"(?:heading|titre|标题)[ _-]*([1-6])$", value, re.IGNORECASE)
    if match:
        return int(match.group(1))
    outline = paragraph.find(f"{_W}pPr/{_W}outlineLvl")
    if outline is not None and outline.get(f"{_W}val", "").isdigit():
        level = int(outline.get(f"{_W}val", "0")) + 1
        if 1 <= level <= 6:
            return level
    return None


def _word_table(table: ET.Element) -> str:
    rows: list[list[str]] = []
    for row in table.findall(f"{_W}tr"):
        cells = [
            re.sub(r"\s+", " ", _text(cell)).replace("|", r"\|")
            for cell in row.findall(f"{_W}tc")
        ]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(normalized[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def docx_to_markdown(source: Path, destination: Path) -> Path:
    """Convert the main document part of a DOCX using only the stdlib."""

    with zipfile.ZipFile(source) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find(f"{_W}body")
    if body is None:
        raise ValueError(f"DOCX has no document body: {source.name}")
    blocks: list[str] = []
    for block in body:
        if block.tag == f"{_W}p":
            text = re.sub(r"\s+", " ", _text(block))
            if not text:
                continue
            heading = _word_heading(block)
            if heading is not None:
                blocks.append("#" * heading + " " + text)
            else:
                blocks.append(text)
        elif block.tag == f"{_W}tbl":
            table = _word_table(block)
            if table:
                blocks.append(table)
    if not blocks:
        raise ValueError(f"DOCX contains no readable paragraphs: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(blocks).rstrip() + "\n", encoding="utf-8")
    return destination


def _find_soffice() -> str | None:
    command = shutil.which("soffice") or shutil.which("soffice.com")
    if command:
        return command
    for candidate in (
        Path(r"C:\Program Files\LibreOffice\program\soffice.com"),
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _doc_to_docx(source: Path, destination_dir: Path) -> Path:
    soffice = _find_soffice()
    if soffice is None:
        raise RuntimeError("LibreOffice is required to materialize a legacy .doc file")
    destination_dir.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="specrag-lo-", dir=destination_dir))
    try:
        result = subprocess.run(
            [
                soffice,
                f"-env:UserInstallation={profile.as_uri()}",
                "--headless", "--nologo", "--nodefault", "--nofirststartwizard",
                "--nolockcheck", "--convert-to", "docx",
                "--outdir", str(destination_dir), str(source),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        converted = destination_dir / (source.stem + ".docx")
        if result.returncode != 0 or not converted.is_file():
            details = (result.stderr or result.stdout).decode("utf-8", errors="replace")
            raise RuntimeError(f"LibreOffice DOC conversion failed: {details.strip()}")
        return converted
    finally:
        shutil.rmtree(profile, ignore_errors=True)


@dataclass(frozen=True)
class ThreeGppArchiveProvider:
    """Discover latest revision ZIPs from the official 3GPP archive."""

    specifications: tuple[str, ...] | None = None
    source_root_relative: str = "data/3gpp"
    archive_base_url: str = "https://www.3gpp.org/ftp/Specs/archive"
    timeout_seconds: int = 30
    user_agent: str = "SpecRAG/1.0 (official 3GPP archive adapter)"

    def _specifications(self) -> tuple[str, ...]:
        if self.specifications is not None:
            values = self.specifications
        else:
            root = REPO_ROOT / self.source_root_relative
            values = tuple(path.name for path in root.iterdir() if path.is_dir()) if root.is_dir() else ()
        valid = sorted({value for value in values if _SPECIFICATION.fullmatch(value)})
        if not valid:
            raise ValueError("no 3GPP specifications were configured for discovery")
        return tuple(valid)

    def _read_listing(self, url: str) -> tuple[str, ...]:
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in {"www.3gpp.org", "3gpp.org"}
        ):
            raise ValueError(f"refusing non-official 3GPP URL: {url}")
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            parser = _DirectoryListing()
            parser.feed(response.read().decode("utf-8", errors="replace"))
        return tuple(urljoin(url if url.endswith("/") else url + "/", link) for link in parser.links)

    def discover(self, releases: tuple[int, ...]) -> tuple[SpecificationSourceCandidate, ...]:
        found: list[SpecificationSourceCandidate] = []
        for specification in self._specifications():
            match = _SPECIFICATION.fullmatch(specification)
            assert match is not None
            series = f"{match.group(1)}_series"
            listing_url = f"{self.archive_base_url.rstrip('/')}/{series}/{specification}/"
            candidates: list[SpecificationSourceCandidate] = []
            for link in self._read_listing(listing_url):
                filename = Path(urlparse(link).path).name
                file_match = _ARCHIVE_FILE.fullmatch(filename)
                if file_match is None or file_match.group(1) != specification.replace(".", ""):
                    continue
                decoded = _revision_version(file_match.group(2))
                if decoded is None or decoded[0] not in releases:
                    continue
                release, version = decoded
                target = (
                    f"{self.source_root_relative}/{specification}/v{version}/"
                    f"{Path(filename).stem}.md"
                )
                candidates.append(SpecificationSourceCandidate(
                    specification=specification,
                    version=version,
                    release=release,
                    source_uri=link,
                    target_source_path=target,
                    source_format="zip",
                ))
            if candidates:
                found.append(max(candidates, key=lambda item: tuple(int(part) for part in item.version.split('.'))))
        return tuple(sorted(found, key=lambda item: (item.release, item.specification)))


@dataclass(frozen=True)
class ThreeGppSourceMaterializer:
    """Download one official archive and create its canonical Markdown file."""

    repository_root: Path = REPO_ROOT
    timeout_seconds: int = 120
    user_agent: str = "SpecRAG/1.0 (official 3GPP archive adapter)"
    max_download_bytes: int = 100 * 1024 * 1024

    def _download(self, uri: str, destination: Path) -> None:
        parsed = urlparse(uri)
        if parsed.scheme != "https" or parsed.hostname not in {"www.3gpp.org", "3gpp.org"}:
            raise ValueError(f"refusing non-official 3GPP source URI: {uri}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = Request(uri, headers={"User-Agent": self.user_agent})
        temporary = destination.with_suffix(destination.suffix + ".part")
        total = 0
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > self.max_download_bytes:
                        raise ValueError("3GPP archive exceeds the configured download limit")
                    output.write(block)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def materialize(self, candidate: SpecificationSourceCandidate) -> CorpusSourceDocument:
        repository_root = self.repository_root.resolve()
        target = (repository_root / candidate.target_source_path).resolve()
        data_root = (repository_root / "data/3gpp").resolve()
        target.relative_to(data_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size > 0:
            return CorpusSourceDocument(
                source_collection="3gpp_markdown",
                specification=candidate.specification,
                version=candidate.version,
                release=candidate.release,
                source_path=target.relative_to(repository_root).as_posix(),
                document_sha256=_sha256(target),
            )

        archive = target.with_suffix(".zip")
        self._download(candidate.source_uri, archive)
        if (
            candidate.expected_sha256 is not None
            and _sha256(archive) != candidate.expected_sha256
        ):
            raise ValueError(f"downloaded archive hash mismatch: {archive.name}")
        staging = target.parent / f"specrag-extract-{uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            extracted = _extract_zip(archive, staging)
            documents = sorted((item for item in extracted if item.suffix.lower() in {".docx", ".doc"}), key=lambda item: (item.suffix.lower() != ".docx", str(item).lower()))
            if not documents:
                raise ValueError(f"3GPP archive contains no DOC/DOCX source: {archive.name}")
            source = documents[0]
            if source.suffix.lower() == ".doc":
                source = _doc_to_docx(source, staging)
            docx_target = target.with_suffix(".docx")
            shutil.copy2(source, docx_target)
            docx_to_markdown(source, target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return CorpusSourceDocument(
            source_collection="3gpp_markdown",
            specification=candidate.specification,
            version=candidate.version,
            release=candidate.release,
            source_path=target.relative_to(repository_root).as_posix(),
            document_sha256=_sha256(target),
        )


__all__ = [
    "ThreeGppArchiveProvider",
    "ThreeGppSourceMaterializer",
    "docx_to_markdown",
]

"""Auditable hybrid retrieval for technical-specification questions.

The retriever consumes a semantic Retrieval Plan.  It never answers the
question and never promotes a passage to a SpecFact.  Its only output is a
ranked set of source-faithful Markdown chunks with stable provenance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import AppConfig, RetrievalCfg, load_config
from .normalization import protocol_token_variants


ROOT = Path(__file__).resolve().parents[3]
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SPECIFICATION = re.compile(r"^\d{2}\.\d{3}$")
_VERSION = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_TOKEN = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*|[\u4e00-\u9fff]+", re.I)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does",
    "for", "from", "how", "if", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "under", "what", "when", "whether",
    "which", "with", "without",
})

CorpusVersionPolicy = Literal[
    "latest_per_spec",
    "highest_version_within_release",
]


def _validate_corpus_scope(
    version_policy: CorpusVersionPolicy,
    release: int | None,
) -> None:
    if version_policy == "highest_version_within_release" and release is None:
        raise ValueError(
            "corpus_release is required for highest_version_within_release"
        )
    if version_policy == "latest_per_spec" and release is not None:
        raise ValueError(
            "corpus_release may only be set for highest_version_within_release"
        )


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceDocument(_StrictModel):
    source_collection: Literal["3gpp_markdown"]
    specification: str | None = Field(default=None, pattern=r"^\d{2}\.\d{3}$")
    version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    source_path: str
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")

    @model_validator(mode="after")
    def _coherent_source_identity(self):
        if self.specification is None or self.version is None:
            raise ValueError("3GPP source requires specification and version")
        return self


class MarkdownChunk(_StrictModel):
    chunk_id: str = Field(min_length=1)
    source_collection: Literal["3gpp_markdown"]
    specification: str | None = Field(default=None, pattern=r"^\d{2}\.\d{3}$")
    version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    source_path: str
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")
    heading_path: tuple[str, ...] = ()
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    character_start: int | None = Field(default=None, ge=0)
    character_end: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_locator(self):
        if self.line_end < self.line_start:
            raise ValueError("chunk line_end precedes line_start")
        if (self.character_start is None) != (self.character_end is None):
            raise ValueError("chunk character bounds must be supplied together")
        if self.character_start is not None and (
            self.line_start != self.line_end
            or (self.character_end or 0) <= self.character_start
        ):
            raise ValueError("chunk character bounds must locate one non-empty line")
        return self


class RetrievalHit(_StrictModel):
    hit_id: str
    chunk_id: str
    collection_rank: int = Field(ge=1)
    source_collection: Literal["3gpp_markdown"]
    specification: str | None = Field(default=None, pattern=r"^\d{2}\.\d{3}$")
    version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    source_path: str
    document_sha256: str = Field(pattern=r"^[0-9A-F]{64}$")
    heading_path: tuple[str, ...] = ()
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    character_start: int | None = Field(default=None, ge=0)
    character_end: int | None = Field(default=None, ge=1)
    excerpt: str = Field(min_length=1)
    semantic_rank: int | None = Field(default=None, ge=1)
    semantic_distance: float | None = None
    lexical_rank: int | None = Field(default=None, ge=1)
    lexical_score: float | None = None
    rrf_score: float = Field(ge=0)

    @model_validator(mode="after")
    def _valid_locator(self):
        if self.line_end < self.line_start:
            raise ValueError("hit line_end precedes line_start")
        if (self.character_start is None) != (self.character_end is None):
            raise ValueError("hit character bounds must be supplied together")
        if self.character_start is not None and (
            self.line_start != self.line_end
            or (self.character_end or 0) <= self.character_start
        ):
            raise ValueError("hit character bounds must locate one non-empty line")
        return self


class RetrievalTaskResult(_StrictModel):
    retrieval_id: str
    query_id: str
    status: Literal["completed", "partial", "no_hits", "unavailable"]
    hits: tuple[RetrievalHit, ...] = ()
    errors: tuple[str, ...] = ()


class RetrievalExecution(_StrictModel):
    schema_version: Literal["retrieval-execution/v1.0"] = (
        "retrieval-execution/v1.0"
    )
    status: Literal["completed", "partial", "unavailable"]
    index_fingerprint: str | None = None
    index_manifest_path: str | None = None
    embedding_model: str
    corpus_version_policy: CorpusVersionPolicy
    corpus_release: int | None = Field(default=None, ge=1)
    tasks: tuple[RetrievalTaskResult, ...] = ()
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _coherent_corpus_scope(self):
        _validate_corpus_scope(
            self.corpus_version_policy,
            self.corpus_release,
        )
        return self


class RetrievalIndexManifest(_StrictModel):
    schema_version: Literal["specification-retrieval-index/v1.0"] = (
        "specification-retrieval-index/v1.0"
    )
    index_fingerprint: str
    embedding_model: str
    corpus_version_policy: CorpusVersionPolicy
    corpus_release: int | None = Field(default=None, ge=1)
    chunk_characters: int
    chunk_overlap_characters: int
    collection_name: str
    documents: tuple[SourceDocument, ...]
    chunk_count: int

    @model_validator(mode="after")
    def _coherent_corpus_scope(self):
        _validate_corpus_scope(
            self.corpus_version_policy,
            self.corpus_release,
        )
        return self


@dataclass(frozen=True)
class _Section:
    heading_path: tuple[str, ...]
    first_line: int
    lines: tuple[str, ...]


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _repository_path(raw_path: str, label: str) -> Path:
    path = (ROOT / raw_path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository: {raw_path}") from exc
    return path


def _version_key(path: Path) -> tuple[int, int, int]:
    match = _VERSION.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid specification version directory: {path}")
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def _select_version_source(
    version_sources: list[tuple[Path, Path]],
    cfg: RetrievalCfg,
) -> tuple[Path, Path] | None:
    eligible = version_sources
    if cfg.version_policy == "highest_version_within_release":
        eligible = [
            item
            for item in version_sources
            if _version_key(item[0])[0] == cfg.target_release
        ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: _version_key(item[0]))


def discover_source_documents(cfg: RetrievalCfg) -> tuple[SourceDocument, ...]:
    """Resolve the immutable source snapshot selected by global policy."""

    documents: list[SourceDocument] = []

    spec_root = _repository_path(cfg.specification_root, "specification root")
    if not spec_root.is_dir():
        raise FileNotFoundError(f"3GPP Markdown corpus not found: {spec_root}")
    for spec_dir in sorted(path for path in spec_root.iterdir() if path.is_dir()):
        if _SPECIFICATION.fullmatch(spec_dir.name) is None:
            continue
        version_sources: list[tuple[Path, Path]] = []
        for version_dir in spec_dir.iterdir():
            if not version_dir.is_dir() or _VERSION.fullmatch(version_dir.name) is None:
                continue
            markdown = sorted(version_dir.glob("*.md"))
            if len(markdown) > 1:
                raise ValueError(
                    f"expected at most one Markdown file in {version_dir}, "
                    f"found {len(markdown)}"
                )
            if markdown:
                version_sources.append((version_dir, markdown[0]))
        if not version_sources:
            continue
        selected = _select_version_source(version_sources, cfg)
        if selected is None:
            continue
        version_dir, source = selected
        documents.append(SourceDocument(
            source_collection="3gpp_markdown",
            specification=spec_dir.name,
            version=version_dir.name[1:],
            source_path=_relative(source),
            document_sha256=hashlib.sha256(source.read_bytes()).hexdigest().upper(),
        ))
    if not documents:
        release_detail = (
            f" for Release {cfg.target_release}"
            if cfg.target_release is not None
            else ""
        )
        raise ValueError(
            "the 3GPP Markdown source snapshot is empty" + release_detail
        )
    return tuple(documents)


def _sections(lines: tuple[str, ...], *, skip_preamble: bool) -> tuple[_Section, ...]:
    starts = [index for index, line in enumerate(lines) if _HEADING.match(line)]
    if not starts:
        return (_Section((), 0, lines),) if lines else ()
    sections: list[_Section] = []
    hierarchy: list[str] = []
    if not skip_preamble and starts[0] > 0:
        sections.append(_Section((), 0, lines[:starts[0]]))
    for position, start in enumerate(starts):
        match = _HEADING.match(lines[start])
        assert match is not None
        level = len(match.group(1))
        hierarchy = hierarchy[: level - 1]
        hierarchy.append(match.group(2).strip())
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        sections.append(_Section(tuple(hierarchy), start, lines[start:end]))
    return tuple(sections)


def _window_section(
    section: _Section,
    *,
    max_characters: int,
    overlap_characters: int,
) -> Iterable[tuple[int, int, int | None, int | None, str]]:
    """Split on source lines and preserve exact inclusive line bounds."""

    lines = section.lines
    start = 0
    while start < len(lines):
        if len(lines[start]) > max_characters:
            source_line = lines[start]
            offset = 0
            stride = max_characters - overlap_characters
            while offset < len(source_line):
                character_end = min(len(source_line), offset + max_characters)
                text = source_line[offset:character_end]
                if text.strip():
                    source_line_number = section.first_line + start + 1
                    yield (
                        source_line_number,
                        source_line_number,
                        offset,
                        character_end,
                        text,
                    )
                if character_end == len(source_line):
                    break
                offset += stride
            start += 1
            continue
        end = start
        size = 0
        while end < len(lines):
            added = len(lines[end]) + (1 if end > start else 0)
            if end > start and size + added > max_characters:
                break
            size += added
            end += 1
            if size >= max_characters:
                break
        if end == start:
            end += 1
        text = "\n".join(lines[start:end])
        if text.strip():
            yield (
                section.first_line + start + 1,
                section.first_line + end,
                None,
                None,
                text,
            )
        if end >= len(lines):
            break
        retained = 0
        next_start = end
        while next_start > start and retained < overlap_characters:
            next_start -= 1
            retained += len(lines[next_start]) + 1
        start = max(start + 1, next_start)


def chunk_source_documents(
    documents: tuple[SourceDocument, ...],
    cfg: RetrievalCfg,
) -> tuple[MarkdownChunk, ...]:
    chunks: list[MarkdownChunk] = []
    for document in documents:
        source = ROOT / document.source_path
        lines = tuple(source.read_text(encoding="utf-8").splitlines())
        sections = _sections(
            lines,
            skip_preamble=document.source_collection == "3gpp_markdown",
        )
        for section in sections:
            for segment, (
                line_start,
                line_end,
                character_start,
                character_end,
                text,
            ) in enumerate(
                _window_section(
                    section,
                    max_characters=cfg.chunk_characters,
                    overlap_characters=cfg.chunk_overlap_characters,
                ),
                start=1,
            ):
                identity = "|".join((
                    document.source_path,
                    document.document_sha256,
                    str(line_start),
                    str(line_end),
                    str(segment),
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                ))
                chunks.append(MarkdownChunk(
                    chunk_id="CH-" + hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest()[:24].upper(),
                    source_collection=document.source_collection,
                    specification=document.specification,
                    version=document.version,
                    source_path=document.source_path,
                    document_sha256=document.document_sha256,
                    heading_path=section.heading_path,
                    line_start=line_start,
                    line_end=line_end,
                    character_start=character_start,
                    character_end=character_end,
                    text=text,
                ))
    return tuple(chunks)


def _fingerprint(
    documents: tuple[SourceDocument, ...],
    cfg: RetrievalCfg,
) -> str:
    payload = {
        "schema": "specification-retrieval-index/v1.0",
        "embedding_model": cfg.embedding_model,
        "version_policy": cfg.version_policy,
        "target_release": cfg.target_release,
        "chunk_characters": cfg.chunk_characters,
        "chunk_overlap_characters": cfg.chunk_overlap_characters,
        "documents": [item.model_dump(mode="json") for item in documents],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def _write_snapshot(
    directory: Path,
    chunks: tuple[MarkdownChunk, ...],
    manifest: RetrievalIndexManifest,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "chunks.jsonl").write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for item in chunks
        ),
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _load_chunks(path: Path) -> tuple[MarkdownChunk, ...]:
    return tuple(
        MarkdownChunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _chroma_metadata(chunk: MarkdownChunk) -> dict[str, str | int]:
    return {
        "source_collection": chunk.source_collection,
        "specification": chunk.specification or "",
        "version": chunk.version or "",
        "source_path": chunk.source_path,
        "document_sha256": chunk.document_sha256,
        "heading_path": json.dumps(chunk.heading_path, ensure_ascii=False),
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
    }


def _searchable_text(chunk: MarkdownChunk) -> str:
    if not chunk.heading_path:
        return chunk.text
    return "\n".join((*chunk.heading_path, chunk.text))


_MARKDOWN_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+.*$")


def evidence_body_text(text: str) -> str:
    """Return evidentiary body text without Markdown section headings.

    Heading paths remain useful retrieval and provenance metadata, but they do
    not establish facts during evidence assessment.
    """

    return "\n".join(
        line for line in text.splitlines()
        if not _MARKDOWN_HEADING_LINE.match(line)
    ).strip()


def _open_vector_collection(
    directory: Path,
    chunks: tuple[MarkdownChunk, ...],
    cfg: RetrievalCfg,
):
    import chromadb
    from chromadb.utils import embedding_functions

    embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg.embedding_model,
        normalize_embeddings=True,
        local_files_only=True,
    )
    client = chromadb.PersistentClient(path=str(directory / "chroma"))
    name = "specification_evidence"
    collection = client.get_or_create_collection(
        name,
        embedding_function=embedding,
        metadata={"hnsw:space": "cosine"},
    )
    expected_ids = {item.chunk_id for item in chunks}
    existing_ids = set(collection.get(include=[]).get("ids", []))
    if not existing_ids.issubset(expected_ids):
        client.delete_collection(name)
        collection = client.create_collection(
            name,
            embedding_function=embedding,
            metadata={"hnsw:space": "cosine"},
        )
        existing_ids = set()
    missing = tuple(item for item in chunks if item.chunk_id not in existing_ids)
    for offset in range(0, len(missing), 128):
        batch = missing[offset : offset + 128]
        collection.add(
            ids=[item.chunk_id for item in batch],
            documents=[_searchable_text(item) for item in batch],
            metadatas=[_chroma_metadata(item) for item in batch],
        )
    return collection


def _tokenize(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(text):
        raw_token = match.group(0)
        token_variants = protocol_token_variants(raw_token)
        if not token_variants:
            continue
        token = token_variants[0]
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) == 1:
                tokens.append(token)
            else:
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
            continue
        for variant in token_variants:
            if variant not in _STOPWORDS and len(variant) > 1:
                tokens.append(variant)
    return tuple(tokens)


class _BM25:
    def __init__(self, chunks: tuple[MarkdownChunk, ...]):
        self.chunks = chunks
        self.frequencies = tuple(
            Counter(_tokenize(_searchable_text(chunk))) for chunk in chunks
        )
        self.lengths = tuple(sum(freq.values()) for freq in self.frequencies)
        lengths_by_collection: dict[str, list[int]] = defaultdict(list)
        frequency_by_collection: dict[str, Counter[str]] = defaultdict(Counter)
        for chunk, frequencies, length in zip(
            chunks, self.frequencies, self.lengths, strict=True
        ):
            lengths_by_collection[chunk.source_collection].append(length)
            frequency_by_collection[chunk.source_collection].update(
                frequencies.keys()
            )
        self.average_length = {
            collection: sum(lengths) / max(1, len(lengths))
            for collection, lengths in lengths_by_collection.items()
        }
        self.idf = {
            collection: {
                term: math.log(
                    1 + (len(lengths_by_collection[collection]) - count + 0.5)
                    / (count + 0.5)
                )
                for term, count in frequencies.items()
            }
            for collection, frequencies in frequency_by_collection.items()
        }

    def search(
        self,
        text: str,
        *,
        source_collection: str,
        limit: int,
        allowed_specifications: frozenset[str] | None = None,
    ) -> list[tuple[str, float]]:
        query_terms = set(_tokenize(text))
        average_length = self.average_length.get(source_collection, 1.0)
        inverse_frequency = self.idf.get(source_collection, {})
        scored: list[tuple[str, float]] = []
        for chunk, frequencies, length in zip(
            self.chunks, self.frequencies, self.lengths, strict=True
        ):
            if chunk.source_collection != source_collection:
                continue
            if (
                allowed_specifications is not None
                and chunk.specification not in allowed_specifications
            ):
                continue
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * length / max(1.0, average_length)
                )
                score += (
                    inverse_frequency.get(term, 0.0)
                    * frequency
                    * 2.5
                    / denominator
                )
            if score > 0:
                scored.append((chunk.chunk_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]


class HybridMarkdownRetriever:
    """Local BGE/Chroma semantic recall plus deterministic BM25/RRF."""

    def __init__(self, cfg: RetrievalCfg):
        self.cfg = cfg
        self.documents = discover_source_documents(cfg)
        self.fingerprint = _fingerprint(self.documents, cfg)
        index_root = _repository_path(cfg.index_dir, "retrieval index directory")
        snapshot = index_root / self.fingerprint
        self.index_manifest_path = _relative(snapshot / "manifest.json")
        chunks_path = snapshot / "chunks.jsonl"
        manifest_path = snapshot / "manifest.json"
        cached_chunks: tuple[MarkdownChunk, ...] | None = None
        if chunks_path.is_file() and manifest_path.is_file():
            try:
                cached_manifest = RetrievalIndexManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                loaded_chunks = _load_chunks(chunks_path)
                if (
                    cached_manifest.index_fingerprint == self.fingerprint
                    and cached_manifest.chunk_count == len(loaded_chunks)
                ):
                    cached_chunks = loaded_chunks
            except (OSError, ValueError):
                cached_chunks = None
        self.chunks = cached_chunks or chunk_source_documents(self.documents, cfg)
        manifest = RetrievalIndexManifest(
            index_fingerprint=self.fingerprint,
            embedding_model=cfg.embedding_model,
            corpus_version_policy=cfg.version_policy,
            corpus_release=cfg.target_release,
            chunk_characters=cfg.chunk_characters,
            chunk_overlap_characters=cfg.chunk_overlap_characters,
            collection_name="specification_evidence",
            documents=self.documents,
            chunk_count=len(self.chunks),
        )
        if cached_chunks is None:
            _write_snapshot(snapshot, self.chunks, manifest)
        self.chunk_by_id = {item.chunk_id: item for item in self.chunks}
        self.bm25 = _BM25(self.chunks)
        self.collection = None
        self.vector_error: str | None = None
        try:
            self.collection = _open_vector_collection(snapshot, self.chunks, cfg)
        except Exception as exc:
            self.vector_error = f"{type(exc).__name__}: {exc}"

    def _dense(
        self,
        text: str,
        source_collection: str,
        allowed_specifications: tuple[str, ...] = (),
    ) -> list[tuple[str, float]]:
        if self.collection is None:
            raise RuntimeError(self.vector_error or "vector index is unavailable")
        where_clauses: tuple[dict[str, Any], ...]
        if source_collection == "3gpp_markdown" and allowed_specifications:
            where_clauses = tuple(
                {
                    "$and": [
                        {"source_collection": source_collection},
                        {"specification": specification},
                    ]
                }
                for specification in allowed_specifications
            )
        else:
            where_clauses = ({"source_collection": source_collection},)

        candidates: dict[str, float] = {}
        for where in where_clauses:
            result = self.collection.query(
                query_texts=[text],
                n_results=self.cfg.semantic_candidates_per_collection,
                where=where,
                include=["distances"],
            )
            ids = result.get("ids", [[]])[0]
            distances = result.get("distances", [[]])[0]
            for chunk_id, distance in zip(ids, distances):
                value = float(distance)
                candidates[chunk_id] = min(candidates.get(chunk_id, value), value)
        return sorted(candidates.items(), key=lambda item: (item[1], item[0]))[
            : self.cfg.semantic_candidates_per_collection
        ]

    def _collection_hits(
        self,
        retrieval_id: str,
        text: str,
        source_collection: str,
    ) -> tuple[list[RetrievalHit], list[str]]:
        errors: list[str] = []
        try:
            dense = self._dense(text, source_collection)
        except Exception as exc:
            dense = []
            errors.append(f"semantic: {type(exc).__name__}: {exc}")
        try:
            lexical = self.bm25.search(
                text,
                source_collection=source_collection,
                limit=self.cfg.lexical_candidates_per_collection,
            )
        except Exception as exc:
            lexical = []
            errors.append(f"lexical: {type(exc).__name__}: {exc}")
        dense_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense, 1)}
        dense_distance = dict(dense)
        lexical_rank = {
            chunk_id: rank for rank, (chunk_id, _) in enumerate(lexical, 1)
        }
        lexical_score = dict(lexical)
        fused: dict[str, float] = defaultdict(float)
        for chunk_id, rank in dense_rank.items():
            fused[chunk_id] += 1 / (self.cfg.rrf_k + rank)
        for chunk_id, rank in lexical_rank.items():
            fused[chunk_id] += 1 / (self.cfg.rrf_k + rank)
        ordered = sorted(fused, key=lambda item: (-fused[item], item))[
            : self.cfg.final_hits_per_collection
        ]
        hits: list[RetrievalHit] = []
        for rank, chunk_id in enumerate(ordered, 1):
            chunk = self.chunk_by_id[chunk_id]
            hits.append(RetrievalHit(
                hit_id=f"{retrieval_id}:{source_collection}:{rank}",
                chunk_id=chunk.chunk_id,
                collection_rank=rank,
                source_collection=chunk.source_collection,
                specification=chunk.specification,
                version=chunk.version,
                source_path=chunk.source_path,
                document_sha256=chunk.document_sha256,
                heading_path=chunk.heading_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                character_start=chunk.character_start,
                character_end=chunk.character_end,
                excerpt=chunk.text,
                semantic_rank=dense_rank.get(chunk_id),
                semantic_distance=dense_distance.get(chunk_id),
                lexical_rank=lexical_rank.get(chunk_id),
                lexical_score=lexical_score.get(chunk_id),
                rrf_score=fused[chunk_id],
            ))
        return hits, errors

    def retrieve(self, item: Any) -> RetrievalTaskResult:
        hits: list[RetrievalHit] = []
        errors: list[str] = []
        for source_collection in item.source_collections:
            try:
                collection_hits, collection_errors = self._collection_hits(
                    item.retrieval_id,
                    item.retrieval_text,
                    source_collection,
                )
                hits.extend(collection_hits)
                errors.extend(
                    f"{source_collection}: {error}"
                    for error in collection_errors
                )
            except Exception as exc:  # preserve a task artifact and fail closed
                errors.append(
                    f"{source_collection}: {type(exc).__name__}: {exc}"
                )
        if hits and errors:
            status = "partial"
        elif hits:
            status = "completed"
        elif errors:
            status = "unavailable"
        else:
            status = "no_hits"
        return RetrievalTaskResult(
            retrieval_id=item.retrieval_id,
            query_id=item.query_id,
            status=status,
            hits=tuple(hits),
            errors=tuple(errors),
        )


def execute_retrieval_plan(
    plan: Any,
    *,
    app_config: AppConfig | None = None,
) -> RetrievalExecution:
    """Execute all Retrieval Plan items and return a serializable audit artifact."""

    cfg = (app_config or load_config()).retrieval
    try:
        retriever = HybridMarkdownRetriever(cfg)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return RetrievalExecution(
            status="unavailable",
            embedding_model=cfg.embedding_model,
            corpus_version_policy=cfg.version_policy,
            corpus_release=cfg.target_release,
            tasks=tuple(
                RetrievalTaskResult(
                    retrieval_id=item.retrieval_id,
                    query_id=item.query_id,
                    status="unavailable",
                    errors=(error,),
                )
                for item in plan.retrievals
            ),
            errors=(error,),
        )

    tasks = tuple(retriever.retrieve(item) for item in plan.retrievals)
    status: Literal["completed", "partial", "unavailable"]
    if tasks and all(item.status == "unavailable" for item in tasks):
        status = "unavailable"
    elif any(item.status in {"partial", "unavailable"} for item in tasks):
        status = "partial"
    else:
        status = "completed"
    return RetrievalExecution(
        status=status,
        index_fingerprint=retriever.fingerprint,
        index_manifest_path=retriever.index_manifest_path,
        embedding_model=cfg.embedding_model,
        corpus_version_policy=cfg.version_policy,
        corpus_release=cfg.target_release,
        tasks=tasks,
        errors=tuple(error for task in tasks for error in task.errors),
    )


__all__ = [
    "HybridMarkdownRetriever",
    "MarkdownChunk",
    "RetrievalExecution",
    "RetrievalHit",
    "RetrievalIndexManifest",
    "RetrievalTaskResult",
    "SourceDocument",
    "chunk_source_documents",
    "discover_source_documents",
    "execute_retrieval_plan",
]

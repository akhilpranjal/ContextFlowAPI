from __future__ import annotations

import io
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .config import Settings

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class PageText:
    document_name: str
    page_number: int | None
    text: str


@dataclass(slots=True)
class ChunkRecord:
    source_id: str
    document_name: str
    page_number: int | None
    chunk_index: int
    content: str


class UnsupportedDocumentError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def detect_document_type(filename: str, content_type: str | None = None) -> str:
    """Detect the supported document type from filename or MIME type."""

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf" or content_type == "application/pdf":
        return "pdf"
    if suffix == ".txt" or content_type in {"text/plain", "application/txt"}:
        return "txt"
    guessed = mimetypes.guess_type(filename)[0]
    if guessed == "application/pdf":
        return "pdf"
    if guessed == "text/plain":
        return "txt"
    raise UnsupportedDocumentError("Only PDF and TXT files are supported.")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving natural reading flow."""

    text = text.replace("\x00", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def extract_pages(filename: str, file_bytes: bytes, content_type: str | None = None) -> list[PageText]:
    """Extract normalized page text from a PDF or TXT document."""

    document_type = detect_document_type(filename, content_type)
    if document_type == "pdf":
        return _extract_pdf_pages(filename, file_bytes)
    return [_extract_txt_page(filename, file_bytes)]


def _extract_pdf_pages(filename: str, file_bytes: bytes) -> list[PageText]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append(PageText(document_name=filename, page_number=index, text=text))
    return pages


def _extract_txt_page(filename: str, file_bytes: bytes) -> PageText:
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("utf-8", errors="ignore")
    return PageText(document_name=filename, page_number=None, text=normalize_text(text))


def chunk_page_text(page: PageText, chunk_size: int, chunk_overlap: int, document_key: str) -> list[ChunkRecord]:
    """Split one page into sentence-aware chunks with configurable overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(page.text) if sentence.strip()]
    if not sentences:
        return []

    chunks: list[ChunkRecord] = []
    current_sentences: list[str] = []
    current_length = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_length = len(sentence) + 1
        if current_sentences and current_length + sentence_length > chunk_size:
            chunks.append(_build_chunk(page, current_sentences, chunk_index, document_key))
            chunk_index += 1
            current_sentences = _build_overlap_for_next_chunk(
                current_sentences=current_sentences,
                chunk_overlap=chunk_overlap,
                chunk_size=chunk_size,
                incoming_sentence_length=sentence_length,
            )
            current_length = sum(len(item) + 1 for item in current_sentences)
        current_sentences.append(sentence)
        current_length += sentence_length

    if current_sentences:
        chunks.append(_build_chunk(page, current_sentences, chunk_index, document_key))

    return chunks


def chunk_pages(pages: Iterable[PageText], settings: Settings, document_key: str) -> list[ChunkRecord]:
    """Chunk every extracted page into overlapping context windows."""

    chunks: list[ChunkRecord] = []
    for page in pages:
        chunks.extend(
            chunk_page_text(
                page=page,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                document_key=document_key,
            )
        )
    if not chunks:
        raise EmptyDocumentError("The uploaded file did not contain extractable text.")
    return chunks


def _build_overlap_for_next_chunk(
    current_sentences: list[str],
    chunk_overlap: int,
    chunk_size: int,
    incoming_sentence_length: int,
) -> list[str]:
    """Select trailing sentences that preserve overlap while leaving room for new content."""

    if chunk_overlap <= 0:
        return []

    overlap: list[str] = []
    total_chars = 0
    for sentence in reversed(current_sentences):
        sentence_chars = len(sentence) + 1
        if overlap and total_chars + sentence_chars > chunk_overlap:
            break
        overlap.append(sentence)
        total_chars += sentence_chars

    overlap.reverse()
    while overlap and (sum(len(item) + 1 for item in overlap) + incoming_sentence_length > chunk_size):
        overlap.pop(0)
    return overlap


def _build_chunk(page: PageText, sentences: list[str], chunk_index: int, document_key: str) -> ChunkRecord:
    """Create a deterministic chunk record with stable source identifiers."""

    content = normalize_text(" ".join(sentences))
    source_id = f"{document_key}:p{page.page_number or 1}:c{chunk_index}"
    return ChunkRecord(
        source_id=source_id,
        document_name=page.document_name,
        page_number=page.page_number,
        chunk_index=chunk_index,
        content=content,
    )

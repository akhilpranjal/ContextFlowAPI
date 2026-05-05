from __future__ import annotations

import io
# The mimetypes module in Python's standard library is used to map filenames 
# or URLs to MIME types and vice-versa. It relies on file extensions to 
# guess the type and does not inspect the actual content of the file.
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    # If file without suffix, like a weburl then it relies on mimetypes.guess_type
    guessed = mimetypes.guess_type(filename)[0]
    if guessed == "application/pdf":
        return "pdf"
    if guessed == "text/plain":
        return "txt"
    raise UnsupportedDocumentError("Only PDF and TXT files are supported.")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving natural reading flow."""

    # Replace null characters with a space
    text = text.replace("\x00", " ")
    # Substitute all consecutive whitespaces with a single space
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
            pages.append(PageText(document_name=filename, page_number=index, text=normalize_text(text)))
    return pages


def _extract_txt_page(filename: str, file_bytes: bytes) -> PageText:
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("utf-8", errors="ignore")
    return PageText(document_name=filename, page_number=None, text=normalize_text(text))


def chunk_pages(
    pages: Iterable[PageText],
    settings: Settings,
    document_key: str | None = None,
) -> list[ChunkRecord]:
    """Chunk every extracted page into overlapping context windows."""

    chunks: list[ChunkRecord] = []
    for page in pages:
        chunks.extend(
            chunk_page_text(
                page=page,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                document_key=document_key or page.document_name,
            )
        )
    if not chunks:
        raise EmptyDocumentError("The uploaded file did not contain extractable text.")
    return chunks


def chunk_page_text(page: PageText, chunk_size: int, chunk_overlap: int, document_key: str) -> list[ChunkRecord]:
    """Split one page into chunks using LangChain's splitter."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(page.text)

    chunk_records: list[ChunkRecord] = []
    for chunk_index, chunk in enumerate(chunks):
        source_id = f"{document_key}:p{page.page_number or 1}:c{chunk_index}"
        chunk_records.append(
            ChunkRecord(
                source_id=source_id,
                document_name=page.document_name,
                page_number=page.page_number,
                chunk_index=chunk_index,
                content=normalize_text(chunk),
            )
        )
    return chunk_records


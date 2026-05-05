Module 2 — Document Ingestion

- Overview
  - Purpose: turn uploaded PDF/TXT into normalized text chunks ready for embeddings and indexing.
  - Key responsibilities: detect type, extract text, normalize whitespace, split text with LangChain's `RecursiveCharacterTextSplitter`, and preserve stable metadata for each chunk.

- Relevant files & symbols
  - `app/ingest.py`: `detect_document_type()`, `normalize_text()`, `extract_pages()`, `_extract_pdf_pages()`, `_extract_txt_page()`, `chunk_page_text()`, `chunk_pages()`
  - External dependency: `langchain-text-splitters` (`RecursiveCharacterTextSplitter`)
  - Errors: `UnsupportedDocumentError`, `EmptyDocumentError`

- Detailed walkthrough
  1. Document type detection (`detect_document_type(filename, content_type)`)
     - Uses file suffix, provided content-type, then `mimetypes.guess_type`.
     - Accepts `.pdf` or PDF MIME → returns `"pdf"`. Accepts `.txt` or text MIME → returns `"txt"`.
     - Raises `UnsupportedDocumentError("Only PDF and TXT files are supported.")` otherwise.
     - Edge cases: files without suffix rely on MIME guess; binary PDFs with wrong MIME will raise.

  2. Text normalization (`normalize_text(text)`)
     - Replaces null chars `\x00` with space.
     - Collapses runs of whitespace into a single space via `_WHITESPACE_RE`.
     - Strips leading/trailing whitespace.
     - Purpose: produce consistent chunks and avoid embedding noise.

  3. Page extraction (`extract_pages(filename, file_bytes, content_type)`)
     - Calls `detect_document_type`.
     - For PDFs: `_extract_pdf_pages()` uses `pypdf.PdfReader` and `page.extract_text()` per page, normalizes via `normalize_text`, returns a list of `PageText(document_name, page_number, text)` where `page_number` starts at 1 and pages with empty text are skipped.
     - For TXT: `_extract_txt_page()` decodes bytes as UTF-8 (falls back to ignoring errors), normalizes, returns a single `PageText` with `page_number=None`.
     - Edge cases: empty or image-only PDFs produce zero pages → later raises `EmptyDocumentError`.

  4. Chunking with LangChain (`chunk_page_text(page, chunk_size, chunk_overlap, document_key)`)
     - Validates `chunk_size > 0`, `0 <= chunk_overlap < chunk_size`.
     - Creates a `RecursiveCharacterTextSplitter` with the configured size and overlap.
     - Uses `split_text(page.text)` to produce text segments.
     - Builds `ChunkRecord` objects with deterministic `source_id` values based on `document_key`, `page_number`, and chunk index.
     - Preserves `document_name`, `page_number`, and normalized chunk content.

  5. Page-level chunking helper (`chunk_pages(pages, settings, document_key)`)
     - Iterates pages and calls `chunk_page_text()` using `settings.chunk_size` and `settings.chunk_overlap`.
     - If no chunks are created (empty document) → raises `EmptyDocumentError("The uploaded file did not contain extractable text.")`.

- Design rationale & tradeoffs
  - LangChain's recursive splitter keeps chunking logic simple and battle-tested instead of maintaining a custom implementation.
  - Character-based chunking is deterministic and fast. Cons: it is not token-aware, so a chunk may still need adjustment if a downstream model has tighter token limits.
  - `source_id` encodes page and chunk index — good for traceability and linking back to the original doc.

- Edge cases & failures to handle
  - Image-only PDFs (no extractable text) → `EmptyDocumentError`.
  - Non-UTF8 text files → decoded by ignoring errors (may lose characters).
  - Very long input segments can still be split by the recursive splitter, but chunk boundaries are character-based rather than token-based.

- Quick examples (how to call)
  - Upload flow (already in `app/main.py`): endpoint `/upload` reads file bytes, calls `extract_pages()`, `chunk_pages()`, then pipeline ingest.
  - Programmatic usage (pseudo):
    ```python
    from app.ingest import extract_pages, chunk_pages
    from app.config import Settings

    pages = extract_pages("doc.pdf", file_bytes, content_type="application/pdf")
    chunks = chunk_pages(pages, Settings(), document_key="doc.pdf")
    ```


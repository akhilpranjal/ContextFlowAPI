Module 2 — Document Ingestion

- Overview
  - Purpose: turn uploaded PDF/TXT into normalized, sentence-aware text chunks ready for embeddings and indexing.
  - Key responsibilities: detect type, extract text, normalize whitespace, split into sentences, build overlapping chunks with stable metadata.

- Relevant files & symbols
  - `app/ingest.py`: `detect_document_type()`, `normalize_text()`, `extract_pages()`, `_extract_pdf_pages()`, `_extract_txt_page()`, `chunk_page_text()`, `chunk_pages()`, `_build_overlap_for_next_chunk()`, `_build_chunk()`
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

  4. Sentence splitting (regex `_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")`)
     - Splits on punctuation followed by whitespace (keeps punctuation).
     - Produces sentences used as atomic units for chunking.
     - Note: It's a simple heuristic (won't handle abbreviations/neat edge cases), but preserves natural sentence boundaries in most documents.

  5. Chunking algorithm (`chunk_page_text(page, chunk_size, chunk_overlap, document_key)`)
     - Parameters validated: `chunk_size > 0`, `0 <= chunk_overlap < chunk_size`.
     - Flow:
       - Split page text into sentences (strip each).
       - Iterate sentences, accumulate `current_sentences` until adding another sentence would exceed `chunk_size` (length measured by characters + 1 per sentence).
       - When chunk boundary reached: build chunk via `_build_chunk()`, increment `chunk_index`.
       - Compute overlap for next chunk with `_build_overlap_for_next_chunk()` which selects trailing sentences whose combined char-length ≈ `chunk_overlap`, but also ensures overlap + incoming sentence fits `chunk_size`.
       - Final partial chunk appended at end.
     - Chunk content is `normalize_text(" ".join(sentences))` to ensure cleanliness.
     - Returns list of `ChunkRecord(source_id, document_name, page_number, chunk_index, content)`.

  6. Overlap selection (`_build_overlap_for_next_chunk(...)`)
     - If `chunk_overlap <= 0` → no overlap returned.
     - Iterates sentences in reverse to accumulate overlap until character budget `chunk_overlap` reached.
     - Reverses the overlap list to restore order.
     - Trims oldest overlap sentences if overlap + incoming sentence would exceed `chunk_size`.
     - Ensures deterministic, sentence-aligned overlap windows.

  7. Chunk identity and metadata (`_build_chunk(...)`)
     - `content` is normalized merged sentence text.
     - `source_id` format: `"{document_key}:p{page_number or 1}:c{chunk_index}"` — deterministic and stable.
     - `document_name` and `page_number` are preserved for provenance.
     - These fields are used later to present sources in responses.

  8. Page-level chunking helper (`chunk_pages(pages, settings, document_key)`)
     - Iterates pages and calls `chunk_page_text` using `settings.chunk_size` and `settings.chunk_overlap`.
     - If no chunks created (empty document) → raises `EmptyDocumentError("The uploaded file did not contain extractable text.")`.

- Design rationale & tradeoffs
  - Sentence-aware chunks reduce context splits in the middle of sentences and improve retrieval usefulness.
  - Character-length measurement (len +1) is a simple proxy; pros: deterministic and fast. Cons: not token-aware — may slightly overshoot LLM token windows; could switch to token-based chunking later.
  - Overlap is sentence-aligned to avoid repeating partial sentences and maintain readable context.
  - `source_id` encodes page and chunk index — good for traceability and linking back to the original doc.

- Edge cases & failures to handle
  - Image-only PDFs (no extractable text) → `EmptyDocumentError`.
  - Non-UTF8 text files → decoded by ignoring errors (may lose characters).
  - Abbreviations with periods (e.g., "e.g.") may create artificial sentence splits.
  - Very long sentences larger than `chunk_size` — current implementation will place them into chunks possibly exceeding the limit; consider splitting long sentences by tokens or fallback.

- Practical exercises
  1. Run unit tests targeting ingestion: `pytest tests/test_ingest.py` (adjust environment/venv).
  2. Add a test for an image-only PDF to verify `EmptyDocumentError`.
  3. Implement a token-aware chunker using a tokenizer (e.g., from `tiktoken` or `sentence-transformers` tokenizer) to ensure chunks respect model token limits.
  4. Add a fallback to split extremely long sentences by approximate sub-sentence windows.

- Quick examples (how to call)
  - Upload flow (already in `app/main.py`): endpoint `/upload` reads file bytes, calls `extract_pages()`, `chunk_pages()`, then pipeline ingest.
  - Programmatic usage (pseudo):
    ```python
    from app.ingest import extract_pages, chunk_pages
    from app.config import Settings

    pages = extract_pages("doc.pdf", file_bytes, content_type="application/pdf")
    chunks = chunk_pages(pages, Settings(), document_key="doc.pdf")
    ```

- Suggested next steps from me
  - I can format this exact content into `docs/module_2_document_ingestion.md` and save it (like Module 1). Want me to create that file now?
  - Or I can produce a hands-on exercise + unit test template to add to `tests/test_ingest_extra.py`.

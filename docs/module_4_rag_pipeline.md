Module 4 — RAG Pipeline (Retrieval-Augmented Generation)

- Overview
  - Purpose: coordinate retrieval of relevant context and produce grounded answers from an LLM.
  - Key responsibilities: embed queries, retrieve top-k chunks, build a grounded prompt, call the LLM safely, and return answer + provenance.

- Relevant files & symbols
  - `app/rag_pipeline.py`: `RAGPipeline`, `ingest_chunks()`, `answer_question()`, `_build_prompt()`, `_call_llm()`
  - `app/embeddings.py`: `EmbeddingService.embed_texts()`
  - `app/vectorstore.py`: `FaissVectorStore.search()`, `to_source_chunks()`
  - `app/schemas.py`: `SourceChunk`
  - Errors: `MissingConfigurationError`, `LLMInvocationError`

- Detailed walkthrough

  1. High-level flow
     - Ingest path: `ingest_chunks()` receives `StoredChunk` records, extracts `content`, computes embeddings via `EmbeddingService.embed_texts()`, and calls `vector_store.add()` to index them.
     - Query path: `answer_question(question)`:
       1. Check LLM client configured (`GROQ_API_KEY`) — raise `MissingConfigurationError` if missing.
       2. Embed the `question` using `EmbeddingService.embed_texts([question])`.
       3. Retrieve top-k candidates from `FaissVectorStore.search()` using `settings.top_k`.
       4. Convert retrieved records to API `SourceChunk` objects with `to_source_chunks()`.
       5. If no sources found, return a safe message indicating no context.
       6. Build a grounded prompt with `_build_prompt(question, sources)` and call `_call_llm(prompt)`.
       7. Return `AnswerPayload(answer, sources)`.

  2. Prompt construction (`_build_prompt`)
     - Purpose: provide the LLM with only the retrieved context plus clear instructions to avoid hallucinations.
     - Format used:
       - Context lines: `[source_id | document_name | page X] content`
       - Combined into a `CONTEXT:` block, followed by `QUESTION:` and `OUTPUT RULES:`.
     - Output rules instruct the model to:
       1. Give a short, grounded answer.
       2. Mention uncertainty explicitly when evidence is thin.
       3. Not invent facts or sources.
     - Rationale: explicit rules plus direct evidence help force grounding and reduce hallucination risk.

  3. LLM invocation (`_call_llm`)
     - Uses Groq client (`self.client.chat.completions.create`) with a system message and a user message containing the prompt.
     - Parameters controlled by settings: `model`, `temperature`, `max_tokens`.
     - Error handling: any exception during the API call is caught and re-raised as `LLMInvocationError` with a friendly message.
     - Response normalization: returns `response.choices[0].message.content.strip()`.

  4. Safety and failure modes
     - Missing API key → `MissingConfigurationError` (maps to HTTP 503 in `main.py`).
     - LLM API failures → `LLMInvocationError` (maps to HTTP 502).
     - No retrieved context → returns a canned answer saying no relevant context was found.
     - Prompt size: concatenating many large chunks may exceed LLM context limits; consider truncation or filtering strategies.

- Design rationale & tradeoffs
  - Separation of concerns: pipeline only orchestrates embedding, retrieval, prompt assembly, and LLM call — model details remain abstracted behind `EmbeddingService` and `FaissVectorStore`.
  - Deterministic prompt structure simplifies debugging and provenance tracing.
  - Using retrieved source text directly in prompt ensures answers can be traced to chunks, improving auditability.
  - Tradeoffs: large context blocks can exceed model token limits or increase cost; selective filtering/re-ranking may be necessary.

- Improvements & extensions
  1. Context window management: implement token-counting and dynamic truncation to ensure prompts fit model limits.
  2. Reranking: apply a lightweight cross-encoder or re-scoring model to refine top-k before prompting.
  3. Redaction/sanitization: remove PII or sensitive data before sending to LLM.
  4. Asynchronous LLM calls or streaming responses for better UX and throughput.
  5. Add provenance links in responses to jump directly to source documents/pages.

- Practical exercises
  1. Unit test `answer_question()` behavior when `GROQ_API_KEY` is missing (expect `MissingConfigurationError`).
  2. Simulate no-retrieval case by using an empty `FaissVectorStore` and verify the returned canned message.
  3. Create a test that injects a mock `EmbeddingService` and `FaissVectorStore` to assert `_build_prompt` formatting.
  4. Implement token-aware prompt truncation and add tests for behavior when many sources are present.

- Quick example (end-to-end)

  - Upload flow: `/upload` endpoint → `RAGPipeline.ingest_chunks()` → embeddings + FAISS add.
  - Query flow: `/query` endpoint → `RAGPipeline.answer_question()` → embed question → FAISS search → build prompt → call Groq → return answer + sources.

---

If you want, I'll save this exact content to `docs/module_4_rag_pipeline.md` now. Proceed? (yes/no)
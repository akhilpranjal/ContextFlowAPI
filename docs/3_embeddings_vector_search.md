Module 3 — Embeddings & Vector Search

- Overview
  - Purpose: convert text chunks into numeric vectors (embeddings) and perform efficient nearest-neighbor search to retrieve relevant context for RAG.
  - Key responsibilities: produce L2-normalized embeddings, index vectors in Qdrant, run similarity search, return ranked source metadata.

- Relevant files & symbols
  - `app/embeddings.py`: `EmbeddingService`, `embed_texts()`
  - `app/vectorstore.py`: `QdrantVectorStore`, `StoredChunk`, `add()`, `search()`, `ensure_collection()`, `to_source_chunks()`

- Detailed walkthrough

  1. Embeddings: what and why
     - Embeddings are fixed-size numeric vectors representing semantic meaning of text.
     - Similar texts map to nearby vectors; distance (or similarity) can be used for retrieval.
     - This project uses `sentence-transformers` model `all-MiniLM-L6-v2` by default.

  2. `EmbeddingService` (in `app/embeddings.py`)
     - Lazy model loading:
       - The `model` property loads `SentenceTransformer` on first use to avoid heavy startup cost.
     - `embed_texts(texts: list[str]) -> np.ndarray`:
       - Returns L2-normalized embeddings (float32).
       - Steps:
         1. If `texts` empty → return empty array shape (0,0).
         2. Call `self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, ...)` to get batched, normalized embeddings.
         3. Return full 2D array of embeddings as dtype `np.float32`.
     - Notes:
       - This simplified implementation does not include an in-process LRU cache or explicit thread locks; the model instance is cached on the `EmbeddingService` instance itself.
       - For high-concurrency or multi-process deployments consider using an external embedding service or process isolation.

  3. Qdrant vector store (`QdrantVectorStore` in `app/vectorstore.py`)
     - Purpose: efficiently store and search vector embeddings with provenance metadata.
     - StoredChunk dataclass holds `source_id`, `document_name`, `page_number`, `chunk_index`, `content`.
     - Index choice:
- Uses a Qdrant collection with cosine distance.
     - Because embeddings are L2-normalized, cosine distance gives stable semantic nearest-neighbor search.
     - `add(chunks, embeddings)`:
       - Validations: non-empty, embeddings must be 2D, lengths must match.
       - Lazily calls `ensure_collection(embeddings.shape[1])` to create the collection when needed.
       - Upserts points with chunk provenance stored in the payload.
     - `search(query_embedding, top_k)`:
       - Validates `top_k` and reshapes a single query vector when needed.
       - Calls Qdrant search and returns the scored points ordered by highest score first.
     - `to_source_chunks(results)`:
       - Converts Qdrant payloads back into `SourceChunk` response models.

- Design rationale & tradeoffs
  - Inner-product index + normalized embeddings gives fast cosine-similarity search.
  - `IndexFlatIP` is exact (no approximation) but not scalable to billions of vectors; simple and deterministic for small/medium datasets.
  - Persistence option is simple local disk persistence (suitable for single-host dev or when using a persistent volume). For production at scale, a dedicated vector DB is recommended.

- Edge cases & failure modes
  - Embedding dimension mismatch:
    - If new embeddings have different dimension than the index, `_ensure_index` raises ValueError.
  - Empty index:
    - `search()` returns empty list if no vectors present.
  - Concurrency:
    - Concurrent `add` and `search` are protected by `_lock` but high-concurrency workloads may need more robust strategies or external DBs.
  - Persistence inconsistencies:
    - If index file exists but metadata missing (or vice-versa), `_load_if_available()` skips load.
  - Memory & performance:
- Qdrant keeps vectors in the configured Qdrant backend rather than in-process RAM.
     - For larger datasets, scale Qdrant using its own storage and deployment options instead of keeping vectors inside the API process.

- Practical exercises
  1. Run the vectorstore unit test:
     - `pytest tests/test_vectorstore.py`
  2. Programmatic usage example remains the same; call `EmbeddingService.embed_texts(texts)` to get normalized embeddings.
  3. Test persistence:
     - Set `QDRANT_URL`, add vectors, restart the process, and verify the same collection still contains the indexed chunks.
  4. Replace `IndexFlatIP` with `IndexHNSWFlat` for approximate nearest neighbor and measure latency/recall tradeoffs.
  5. Add a benchmark to measure embed + search latency for N documents.

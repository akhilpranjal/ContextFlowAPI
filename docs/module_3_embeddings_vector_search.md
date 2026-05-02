Module 3 — Embeddings & Vector Search

- Overview
  - Purpose: convert text chunks into numeric vectors (embeddings) and perform efficient nearest-neighbor search to retrieve relevant context for RAG.
  - Key responsibilities: produce L2-normalized embeddings, cache repeated requests, index vectors in FAISS, run similarity search, return ranked source metadata.

- Relevant files & symbols
  - `app/embeddings.py`: `EmbeddingService`, `embed_texts()`, `_get_cached_embedding()`, `_put_cached_embedding()`
  - `app/vectorstore.py`: `FaissVectorStore`, `StoredChunk`, `add()`, `search()`, `_ensure_index()`, `_persist()`, `_load_if_available()`, `to_source_chunks()`

- Detailed walkthrough

  1. Embeddings: what and why
     - Embeddings are fixed-size numeric vectors representing semantic meaning of text.
     - Similar texts map to nearby vectors; distance (or similarity) can be used for retrieval.
     - This project uses `sentence-transformers` model `all-MiniLM-L6-v2` by default.

  2. `EmbeddingService` (in `app/embeddings.py`)
     - Lazy model loading:
       - The `model` property loads `SentenceTransformer` on first use under `_model_lock` to avoid heavy startup cost.
     - Caching:
       - Uses an `OrderedDict` as an LRU cache with `_cache_lock` to store recently computed embeddings.
       - Controlled by `settings.enable_embedding_cache` and `settings.embedding_cache_max_items`.
     - `embed_texts(texts: list[str]) -> np.ndarray`:
       - Returns L2-normalized embeddings (float32).
       - Steps:
         1. If `texts` empty → return empty array shape (0,0).
         2. For each text, check cache (if enabled). If cached, reuse embedding.
         3. Collect missing texts and call `self.model.encode(missing, normalize_embeddings=True, convert_to_numpy=True, ...)`.
            - `normalize_embeddings=True` ensures returned vectors are L2-normalized (unit length) — important for cosine similarity with inner product index.
         4. Insert newly computed vectors back into appropriate positions and update cache if enabled.
         5. Return full 2D array of embeddings as dtype `np.float32`.
     - Thread-safety:
       - `_model_lock` ensures only one thread loads the heavy model.
       - `_cache_lock` (RLock) protects the in-memory cache.

  3. Faiss vector store (`FaissVectorStore` in `app/vectorstore.py`)
     - Purpose: efficiently store and search vector embeddings with provenance metadata.
     - StoredChunk dataclass holds `source_id`, `document_name`, `page_number`, `chunk_index`, `content`.
     - Index choice:
       - Uses `faiss.IndexFlatIP(dimension)` — an inner-product (dot product) index.
       - Because embeddings are L2-normalized, inner product is equivalent to cosine similarity (cosine = dot(u,v) when ||u||=||v||=1).
     - `add(chunks, embeddings)`:
       - Validations: non-empty, embeddings must be 2D, lengths must match.
       - Lazily calls `_ensure_index(embeddings.shape[1])` to initialize FAISS index and set expected dimension.
       - Adds embeddings to the index and appends `chunks` to `_records`.
       - If `enable_faiss_persistence` is true, persist index and metadata to disk.
     - `search(query_embedding, top_k)`:
       - Validates `top_k`.
       - Ensures index exists and not empty.
       - Normalizes/reshapes query embedding to (1, d) if needed.
       - Calls FAISS `search` to get `scores` and `indices` and returns a list of `(StoredChunk, float(score))` ordered by highest score first.
       - Note: scores are inner-product values (range depends on embedding values; with normalized embeddings range is [-1,1]).
     - Persistence:
       - `_persist()` writes FAISS index to `settings.faiss_index_path` and `self._records` metadata to `settings.faiss_metadata_path` as JSON.
       - `_load_if_available()` loads both files if present and reconstructs `_records`.
     - `to_source_chunks(results)` converts internal `StoredChunk` + score tuples into API `SourceChunk` schema for responses.

- Design rationale & tradeoffs
  - Inner-product index + normalized embeddings gives fast cosine-similarity search.
  - `IndexFlatIP` is exact (no approximation) but not scalable to billions of vectors; simple and deterministic for small/medium datasets.
  - LRU caching reduces repeated embedding compute for duplicated or frequently-requested chunks (common during ingestion retries or repeated queries).
  - Thread locks provide safe concurrent access in multithreaded server (embedding model load and cache operations).
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
    - `IndexFlatIP` keeps all vectors in RAM; memory grows linearly with number of vectors.
    - For larger datasets, consider approximate indexes (HNSW or IVF) or external vector stores.

- Practical exercises
  1. Run the vectorstore unit test:
     - `pytest tests/test_vectorstore.py`
  2. Verify embedding cache behavior:
     - Call `EmbeddingService.embed_texts(["same text","same text"])` and assert only one encode call happens (mock `SentenceTransformer.encode`).
  3. Test persistence:
     - Enable `ENABLE_FAISS_PERSISTENCE=true`, add vectors, restart process, and verify `FaissVectorStore` reloads index and metadata.
  4. Replace `IndexFlatIP` with `IndexHNSWFlat` for approximate nearest neighbor and measure latency/recall tradeoffs.
  5. Add a benchmark to measure embed + search latency for N documents.

- Quick examples

  - Programmatic usage:
    ```python
    from app.config import Settings
    from app.embeddings import EmbeddingService
    from app.vectorstore import FaissVectorStore, StoredChunk
    import numpy as np

    settings = Settings(enable_faiss_persistence=False)
    emb = EmbeddingService(settings)
    store = FaissVectorStore(settings)

    texts = ["First chunk text.", "Second chunk text."]
    embeddings = emb.embed_texts(texts)  # (2, d) normalized float32
    chunks = [
      StoredChunk(source_id="doc:p1:c0", document_name="doc.pdf", page_number=1, chunk_index=0, content=texts[0]),
      StoredChunk(source_id="doc:p1:c1", document_name="doc.pdf", page_number=1, chunk_index=1, content=texts[1]),
    ]
    store.add(chunks, embeddings)

    q = emb.embed_texts(["query text"])  # (1, d)
    results = store.search(q, top_k=2)
    for record, score in results:
        print(record.source_id, score)
    ```

- Improvements & next steps
  - Token-aware chunking: ensure chunks respect LLM token limits by using a tokenizer during chunking.
  - Batched embedding requests & async encoding: if model supports async/batched processing, optimize throughput.
  - Replace local FAISS with managed vector DB (Pinecone, Milvus, Qdrant) for horizontal scale and persistence.
  - Add background tasks to periodically persist snapshots or rebuild indexes.

---

If you want I will now save this exact content to `docs/module_3_embeddings_vector_search.md`. Proceed to save? (yes/no)
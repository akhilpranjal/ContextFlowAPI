Module 5 — Production Patterns

- Overview
  - Purpose: make the RAG API robust, observable, maintainable, and production-ready.
  - Key responsibilities: configuration management, logging & monitoring, thread safety, resource lifecycle, persistence strategies, testing, security, deployment, and scaling.

- Relevant files & symbols
  - `app/config.py`: `Settings`, `get_settings()`
  - `app/main.py`: lifespan, middleware, exception handlers
  - `app/ingest.py`: upload extraction and chunking via LangChain's `RecursiveCharacterTextSplitter`
  - `app/embeddings.py`: lazy model loading, cache locks
  - `app/vectorstore.py`: FAISS persistence and locking
  - `tests/`: unit tests and test patterns

- Topics and best practices

  1. Configuration management
     - Use `pydantic`/`pydantic-settings` `Settings` to load env vars and `.env` files.
     - Keep secrets out of source control; set `GROQ_API_KEY` in environment or secrets manager.
     - Provide sensible defaults and validate types (e.g., `max_upload_size_mb`).
     - Use `@lru_cache` for a singleton `get_settings()` to avoid recreating settings and directories per request.
     - For multi-instance deployments, prefer external configuration (Kubernetes ConfigMaps/Secrets, Vault, or cloud secret managers).

  2. Logging & monitoring
     - Initialize logging once in the lifespan using `logging.basicConfig(level=...)` and include structured logs where helpful.
     - Log key events: startup, shutdown, ingestion results (chunks indexed), errors (with stack traces), and external API failures.
     - Integrate metrics: expose Prometheus-compatible metrics (request latencies, error counts, queue sizes, embedding/search latencies).
     - Health & readiness endpoints: `GET /health` and optionally readiness checks to indicate when the model/index is ready.

  3. Thread-safety & concurrency
     - Use locks for shared in-memory resources: `_model_lock` in `EmbeddingService` and `_lock` in `FaissVectorStore` are correct patterns.
     - Prefer process-level isolation for heavy models: multiple worker processes (uvicorn/gunicorn) reduce GIL contention, but increase memory usage due to separate model copies.
     - Consider using a separate service for heavy tasks (embedding or indexing) behind a queue (Redis/RabbitMQ) to scale independently.

  4. Lazy loading & startup cost
     - Lazily load heavy models on first use to keep cold-start time low; or pre-load during startup if you want readiness gating.
     - Use the `lifespan` context to initialize long-lived singletons (embedding model, FAISS store) once and share via `app.state`.
     - Keep runtime dependencies aligned with the active environment; this repository expects `langchain-text-splitters` to be installed in the same venv that runs tests and the API.

  5. Persistence & durability
     - `FaissVectorStore` supports local file persistence when `ENABLE_FAISS_PERSISTENCE=true` — suitable for single-host setups.
     - For production, use managed or distributed vector DBs (Qdrant, Milvus, Pinecone) for durability, scaling, and high availability.
     - Backups: persist FAISS index and metadata regularly; keep snapshots in durable object storage (S3, GCS).

  6. Prompt & context management
     - Implement token-aware context management to avoid exceeding the LLM context window (count tokens, truncate older/less relevant sources).
     - Prefer re-ranking (cross-encoder) for top-k selection before prompting to reduce prompt size and improve answer quality.

  7. Error handling & retry strategies
     - Map internal errors to appropriate HTTP codes (shown in `main.py`).
     - Use exponential backoff and retry for transient external failures (e.g., Groq API) but avoid retrying expensive or non-idempotent operations without care.
     - Return helpful error messages without leaking secrets.

  8. Testing & CI
     - Unit tests: isolate services via dependency injection and mocks (mock `EmbeddingService` and `FaissVectorStore`).
     - Integration tests: run ingestion + query flows against a local FAISS instance and small embedding model; use pytest markers to separate slow tests.
     - End-to-end: containerize and run end-to-end smoke tests in CI to validate startup, upload, and query.
     - Add test coverage checks and linting in CI.
     - Run the suite with the workspace interpreter when validating local dependency installs: `d:/Programming/Github/ContextFlowAPI/.venv/Scripts/python.exe -m pytest -q`.

  9. Security
     - Protect API endpoints in production (authentication & authorization). Add API keys or OAuth for clients.
     - Sanitize user inputs and consider redaction of PII before storing or sending to LLMs.
     - Limit upload sizes and scan uploads if necessary.

  10. Observability & tracing
     - Emit traces for request flows (OpenTelemetry) to track time spent in extraction, embedding, search, and LLM calls.
     - Attach source metadata to traces for easier debugging.

  11. Deployment patterns
     - Containerize application with a minimal image; use prebuilt wheel if startup cost matters.
     - Use multiple replicas behind a load balancer; ensure FAISS persistence uses shared volumes or external DB.
     - Use readiness probes that fail until index and model are ready.

  12. Scaling considerations
     - Embedding model CPU/GPU constraints: use GPU-backed workers for large models or dedicate an embedding microservice.
     - Vector search: scale with approximate indexes (HNSW/IVF) or external vector DBs; shard by tenant/document partitions if necessary.

- Practical exercises & checklist
  1. Add Prometheus metrics: instrument ingestion count, embed latency, search latency, LLM latency.
  2. Add an integration test that runs `uvicorn app.main:app` in a test container and performs an upload + query.
  3. Configure CI to run linting, tests, and a small smoke test on each PR.
  4. Add an environment-based configuration example for production (`.env.example` already present) and document deployment steps in `README.md`.
  5. Verify that `langchain-text-splitters` stays pinned in `requirements.txt` so ingestion imports remain stable.

- Templates & snippets
  - Example readiness check in `lifespan`:
    ```python
    app.state.ready = False
    # after model and index loaded
    app.state.ready = True

    @app.get('/ready')
    async def ready():
        return JSONResponse(status_code=200 if app.state.ready else 503, content={'ready': app.state.ready})
    ```

  - Prometheus instrumentation (example using `prometheus_client`):
    ```python
    from prometheus_client import Summary, Counter
    EMBED_LATENCY = Summary('embed_latency_seconds', 'Time spent creating embeddings')
    INGEST_COUNTER = Counter('documents_indexed_total', 'Number of documents indexed')

    @EMBED_LATENCY.time()
    def embed_texts(...):
        ...
    ```
## Module 1: FastAPI & REST API Fundamentals

---

## **Section 1.1: What is FastAPI? Core Concepts**

### The Big Picture
FastAPI is a modern Python web framework for building **async REST APIs**. It automatically:
- Validates incoming requests
- Serializes outgoing responses  
- Generates interactive API documentation
- Handles HTTP properly

In your project, it serves as the **HTTP interface** between users and your RAG pipeline.

### Why Async?
```python
# SYNC (blocks while waiting)
response = requests.get("http://groq-api.com")  # ⏸️ blocks entire thread

# ASYNC (non-blocking)
response = await client.get("http://groq-api.com")  # ⚡ thread stays free
```

With async, one thread can handle **thousands of requests** instead of one. Critical for production APIs!

### The Foundation: Your App Object
```python
app = FastAPI(title="ContextFlow RAG API", version="1.0.0", lifespan=lifespan)
```

**What this does:**
- Creates the FastAPI application
- Sets metadata for documentation
- Attaches the lifespan context manager (we'll explain soon)

---

## **Section 1.2: Configuration Management with Pydantic Settings**

### Why Separate Configuration?

Your code needs settings (API keys, model names, file sizes) but you **don't want to hardcode them**. Why?

1. **Security**: Never commit secrets to git
2. **Flexibility**: Different settings for dev/prod
3. **Environment**: Settings change per deployment

### How It Works in Your Project

From `config.py`:

```python
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",                    # Load from .env file
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
    )
    
    # Example: Load GROQ_API_KEY from environment
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-70b-8192", validation_alias="GROQ_MODEL")
```

### What's Happening?

| Concept | Explanation |
|---------|-------------|
| `BaseSettings` | Pydantic class that reads from environment |
| `Field(default="...", validation_alias="...")` | Maps environment variable `GROQ_API_KEY` → `groq_api_key` |
| `SettingsConfigDict` | Tells Pydantic where/how to load settings |

### Loading Order (Priority)
1. **Environment variables** (`$env:GROQ_API_KEY=xxx`)
2. **.env file** (if `.env.example` exists)
3. **Default values** (fallback)

### Caching Settings (Important!)

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance for the application."""
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
```

**Why cache?**
- Creating `Settings()` every request is wasteful
- `@lru_cache` ensures it's only created once
- Subsequent calls return the cached instance

---

## **Section 1.3: Data Validation with Pydantic Models**

### What is a Pydantic Model?

From `schemas.py`:

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user question to answer.")
```

This creates a **contract** for incoming data:
- **Input**: Raw JSON from API client
- **Pydantic validates**: Is `question` a string? Is it at least 1 character?
- **Output**: Python object you can use in code

### Real Example: What Happens When Someone Calls Your API

**Client sends:**
```json
{
  "question": "What is RAG?"
}
```

**Pydantic does:**
```python
payload = QueryRequest(**{"question": "What is RAG?"})
# ✓ Validates: is it a string?
# ✓ Validates: min_length=1?
# ✓ Creates: QueryRequest(question="What is RAG?")
```

**If validation fails** (e.g., missing `question`):
```json
{"question": null}  # ❌ FastAPI returns HTTP 422 with error details
```

### Schemas in Your Project

All your data contracts:

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

class SourceChunk(BaseModel):
    source_id: str
    document_name: str
    page_number: int | None = None
    chunk_index: int
    score: float
    content: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]

class UploadResponse(BaseModel):
    document_name: str
    chunks_indexed: int
    message: str

class ErrorResponse(BaseModel):
    detail: str
    extra: dict[str, Any] | None = None
```

**Why this matters:**
- **API documentation**: FastAPI auto-generates docs showing what fields are required
- **Type safety**: Your IDE knows the shape of data
- **Validation**: Invalid data never reaches your business logic

---

## **Section 1.4: Application Lifespan - Startup & Shutdown**

### The Problem

Your app needs **expensive resources** that should be initialized **once at startup**:
- Embedding model (100MB+ loaded into RAM)
- FAISS vector store
- Groq API client

If you initialize these **per request**, your API would be **glacially slow**.

### The Solution: Lifespan Context Manager

From `main.py`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize singleton services for application runtime."""
    
    # STARTUP CODE (runs once when app starts)
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    app.state.settings = settings
    app.state.embedding_service = EmbeddingService(settings)
    app.state.vector_store = FaissVectorStore(settings)
    app.state.pipeline = RAGPipeline(settings, app.state.embedding_service, app.state.vector_store)
    
    yield  # ⬅️ App runs here, handling requests
    
    # SHUTDOWN CODE (runs when app stops)
    # Cleanup happens here (we could close connections, save state, etc.)
```

### Timeline

```
1. Server starts
2. lifespan() runs UP TO yield
   → Initializes: settings, embedding_service, vector_store, pipeline
3. yield
   → App accepts requests
4. User sends request → handlers use app.state.settings, app.state.pipeline
5. User sends request → handlers use app.state.settings, app.state.pipeline
6. Server stops
7. Code AFTER yield runs (cleanup)
```

### Accessing Resources in Endpoints

Once initialized in lifespan, endpoints can access them:

```python
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}  # No dependencies needed for simple endpoints
```

---

## **Section 1.5: Dependency Injection - The Magic Pattern**

### What Problem Does It Solve?

Without dependency injection:
```python
@app.post("/upload")
async def upload_document(file: UploadFile):
    settings = get_settings()  # Called every request ❌ wasteful
    pipeline = create_pipeline()  # Called every request ❌ wasteful
    # ... do stuff
```

With dependency injection (your code):
```python
@app.post("/upload")
async def upload_document(
    file: UploadFile,
    settings: Settings = Depends(get_app_settings),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    # settings and pipeline are already created
    # ... do stuff
```

### How It Works

From `main.py`:

```python
def get_pipeline(request: Request) -> RAGPipeline:
    """Resolve shared RAG pipeline from application state."""
    return request.app.state.pipeline


def get_app_settings(request: Request) -> Settings:
    """Resolve cached application settings from state."""
    return request.app.state.settings
```

### FastAPI's Magic: `Depends()`

When you write:
```python
async def upload_document(
    pipeline: RAGPipeline = Depends(get_pipeline),
):
```

FastAPI automatically:
1. Sees `Depends(get_pipeline)`
2. Calls `get_pipeline(request)`
3. Injects result as `pipeline` parameter
4. Your endpoint receives the ready-to-use object

### Why This Pattern?

| Benefit | Example |
|---------|---------|
| **Testing** | Replace `get_pipeline()` with mock for unit tests |
| **Flexibility** | Change where settings come from without touching endpoints |
| **Clarity** | Endpoint signature shows all dependencies |
| **Reusability** | Multiple endpoints share same `get_pipeline()` logic |

---

## **Section 1.6: Exception Handling - Graceful Error Responses**

### The Problem

Your code can fail in many ways:
- User uploads unsupported file format
- GROQ API is down
- Request validation fails

Without proper handling, users get **raw Python errors** or **500 Internal Server Error**.

### The Solution: Custom Exception Handlers

FastAPI lets you catch specific exceptions and return proper HTTP responses:

```python
@app.exception_handler(UnsupportedDocumentError)
async def unsupported_document_handler(_: Request, exc: UnsupportedDocumentError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(detail=str(exc)).model_dump()
    )
```

### Your Exception Strategy

From `main.py`:

```python
# Document-related errors → 400 Bad Request
@app.exception_handler(UnsupportedDocumentError)
@app.exception_handler(EmptyDocumentError)

# Validation error → 422 Unprocessable Entity
@app.exception_handler(RequestValidationError)

# Configuration error → 503 Service Unavailable
@app.exception_handler(MissingConfigurationError)

# LLM error → 502 Bad Gateway
@app.exception_handler(LLMInvocationError)
@app.exception_handler(GroqAPIError)
```

### HTTP Status Codes Explained

| Code | Meaning | When Used |
|------|---------|-----------|
| 400 | Bad Request | Client's fault (bad format, unsupported file) |
| 422 | Unprocessable Entity | Validation failed (missing required field) |
| 502 | Bad Gateway | External service failed (Groq API down) |
| 503 | Service Unavailable | Our service is broken (missing config) |

### Standard Error Response

All errors return this format:
```python
class ErrorResponse(BaseModel):
    detail: str
    extra: dict[str, Any] | None = None
```

**Example:**
```json
{
  "detail": "Only PDF and TXT files are supported.",
  "extra": null
}
```

---

## **Section 1.7: Middleware & CORS**

### What is Middleware?

Middleware is code that **processes every request and response**:

```
Request → Middleware → Endpoint Handler → Middleware → Response
```

### CORS: The Browser Security Problem

**Scenario:**
```
Browser: https://app.example.com
Server: https://api.example.com
```

Without CORS, browser blocks the request! Why? **Security against malicious websites.**

### Your CORS Configuration

From `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allow all origins (⚠️ dev only!)
    allow_credentials=True,
    allow_methods=["*"],           # Allow all HTTP methods
    allow_headers=["*"],           # Allow all headers
)
```

### What This Does

| Setting | Meaning |
|---------|---------|
| `allow_origins=["*"]` | Any website can call this API |
| `allow_credentials=True` | Cookies/auth can be sent |
| `allow_methods=["*"]` | GET, POST, PUT, DELETE all allowed |
| `allow_headers=["*"]` | Any custom headers accepted |

**⚠️ Security Note:** `"*"` is for development. In production:
```python
allow_origins=["https://app.example.com", "https://www.example.com"]
```

---

## **Section 1.8: API Endpoints Deep Dive**

### Endpoint 1: Health Check

```python
@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "ok"}
```

**Purpose**: Kubernetes, load balancers, and monitoring services use this to verify the app is alive.

**Why separate from root?** Monitoring tools expect a dedicated health endpoint.

---

### Endpoint 2: Root

```python
@app.get("/")
async def root() -> dict[str, str]:
    """Basic root endpoint for platform probes and quick sanity checks."""
    return {"service": "ContextFlow RAG API", "status": "ok"}
```

**Purpose**: Human-friendly response if someone visits `http://localhost:8000/`

---

### Endpoint 3: Upload Document

```python
@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    settings: Settings = Depends(get_app_settings),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Upload, parse, chunk, embed, and index a single document."""
    
    # Read file bytes
    file_bytes = await file.read()
    
    # Validate file size
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the maximum upload size.")
    
    # Process: extract → chunk → embed → store
    pages = extract_pages(file.filename or "uploaded_document", file_bytes, file.content_type)
    chunks = chunk_pages(pages, settings, document_key=file.filename or "uploaded_document")
    stored_chunks = [
        StoredChunk(
            source_id=chunk.source_id,
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
        )
        for chunk in chunks
    ]
    count = pipeline.ingest_chunks(stored_chunks)
    logger.info("Indexed %s chunks from %s", count, file.filename)
    
    # Return response (Pydantic validates before sending)
    return UploadResponse(
        document_name=file.filename or "uploaded_document",
        chunks_indexed=count,
        message="Document indexed successfully."
    )
```

**Breakdown:**

| Step | Code |
|------|------|
| **1. Parse parameters** | `file`, `settings`, `pipeline` |
| **2. Validate file size** | Check against `max_upload_size_bytes` |
| **3. Extract text** | `extract_pages()` → list of pages |
| **4. Chunk text** | `chunk_pages()` → overlapping chunks |
| **5. Embed chunks** | `pipeline.ingest_chunks()` → vectors + store |
| **6. Return response** | `UploadResponse` object (auto-serialized to JSON) |

**Key Points:**
- `UploadFile` handles multipart file uploads
- `await file.read()` reads the entire file asynchronously
- Dependencies auto-inject `settings` and `pipeline`
- Response is automatically validated and serialized

---

### Endpoint 4: Query Documents

```python
@app.post("/query", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    """Answer a user question from indexed document context."""
    
    result = pipeline.answer_question(payload.question)
    return QueryResponse(answer=result.answer, sources=result.sources)
```

**Flow:**

1. **Input validation**: `QueryRequest` validates `{"question": "..."}`
2. **Pipeline call**: `pipeline.answer_question()` orchestrates RAG
3. **Output**: Returns `QueryResponse` with answer + source chunks

---

## **Section 1.9: Async Programming Fundamentals**

### Sync vs Async: The Key Difference

```python
# SYNC (traditional)
def get_user(user_id: int) -> User:
    response = requests.get(f"http://api.example.com/users/{user_id}")  # ⏸️ BLOCKS HERE
    return User(**response.json())

# ASYNC (modern)
async def get_user(user_id: int) -> User:
    response = await client.get(f"http://api.example.com/users/{user_id}")  # ⚡ NON-BLOCKING
    return User(**response.json())
```

### Why Async Matters for APIs

**Sync scenario** (can handle ~1000 requests):
```
Request 1 → Wait 1 second → Thread 1 blocked
Request 2 → Wait 1 second → Thread 2 blocked
Request 3 → Wait 1 second → Thread 3 blocked
... need 1000 threads for 1000 concurrent requests
```

**Async scenario** (can handle 100,000+ requests):
```
Request 1 → Wait 1 second → Thread 1 free!
Request 2 → Wait 1 second → Thread 1 free!
Request 3 → Wait 1 second → Thread 1 free!
... one thread handles all!
```

### Your Code Uses Async Everywhere

```python
@asynccontextmanager
async def lifespan(app: FastAPI):  # ← async
    # ...
    yield

@app.post("/upload")
async def upload_document(...):  # ← async
    file_bytes = await file.read()  # ← await
    # ...

@app.post("/query")
async def query_documents(...):  # ← async
    result = pipeline.answer_question(payload.question)
    return QueryResponse(answer=result.answer, sources=result.sources)
```

### Key Concepts

| Term | Meaning |
|------|---------|
| `async def` | Function that can be paused (awaited) |
| `await` | Pause function, let others run, resume when ready |
| `@asynccontextmanager` | Async version of context manager |

---

## **Section 1.10: Putting It All Together - Request Flow**

Let me show you the **complete flow** of one request from start to finish:

### User uploads a PDF

```
CLIENT                        SERVER
  │
  ├─ POST /upload
  │   multipart/form-data
  │   file: document.pdf
  │                              ↓
  │                         1. FastAPI receives request
  │                         2. CORS Middleware processes
  │                         3. Parse UploadFile
  │                         4. FastAPI sees dependencies:
  │                            - Calls get_app_settings()
  │                            - Calls get_pipeline()
  │                         5. upload_document() executes:
  │                            a) await file.read() → bytes
  │                            b) extract_pages() → pages
  │                            c) chunk_pages() → chunks
  │                            d) pipeline.ingest_chunks() → stored
  │                         6. Create UploadResponse
  │                         7. Pydantic validates & serializes
  │                         8. Return JSON
  │                              ↓
  │  HTTP 200
  │  {
  │    "document_name": "document.pdf",
  │    "chunks_indexed": 45,
  │    "message": "Document indexed successfully."
  │  }
  ├─
```

---

## **Quiz Time! Test Your Understanding**

Before we move forward, answer these questions (I'll reveal answers):

1. **Why do we use `@lru_cache` for `get_settings()`?**
2. **What's the difference between `Field(default="")` and `Field(...)`?**
3. **What happens if a Pydantic validation fails?**
4. **Why do we need the lifespan context manager?**
5. **How does dependency injection help with testing?**

---

<i>I've completed a thorough walkthrough of **Module 1: FastAPI & REST APIs**. We covered:</i>

✅ Core FastAPI concepts  
✅ Configuration management with Pydantic Settings  
✅ Data validation with Pydantic models  
✅ Application lifespan (startup/shutdown)  
✅ Dependency injection pattern  
✅ Exception handling and HTTP status codes  
✅ CORS middleware  
✅ All four API endpoints  
✅ Async programming fundamentals  
✅ Complete request-response flow

Would you like me to:
1. **Answer the quiz questions** above?
2. **Create a practical exercise** you can code?
3. **Move to Module 2: Document Ingestion**?
4. **Deep dive on any topic** from Module 1?

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from groq import APIError as GroqAPIError

from .config import Settings, get_settings
from .embeddings import EmbeddingService
from .ingest import EmptyDocumentError, UnsupportedDocumentError, chunk_pages, extract_pages
from .rag_pipeline import LLMInvocationError, MissingConfigurationError, RAGPipeline
from .schemas import ErrorResponse, QueryRequest, QueryResponse, UploadResponse
from .vectorstore import QdrantVectorStore, StoredChunk

logger = logging.getLogger("contextflow")


# asynccontextmanager provides 
    # Async support: Your services might use async I/O (DBs, APIs)
    # Two-phase lifecycle: Normal functions can’t “pause” like yield does here
    # Separates setup, runtime and cleanup phases
# On app startup:
#     lifespan() runs
#     Services like settings, enbedding_service, vector_store, pipeline are created
#     Stored in app.state
#     During requests: Any endpoint can access request.app.state.pipeline
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize singleton services for application runtime."""

    # Startup code (runs once when app starts)
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    app.state.settings = settings
    app.state.embedding_service = EmbeddingService(settings)
    app.state.vector_store = QdrantVectorStore(settings)
    app.state.pipeline = RAGPipeline(settings, app.state.embedding_service, app.state.vector_store)
    # App runs here, handling requests
    yield
    # Shutdown code (if any) (runs when app stops) (cleanup happens here)


app = FastAPI(title="ContextFlow RAG API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Will be used for dependency injection
def get_pipeline(request: Request) -> RAGPipeline:
    """Resolve shared RAG pipeline from application state."""

    return request.app.state.pipeline


def get_app_settings(request: Request) -> Settings:
    """Resolve cached application settings from state."""

    return request.app.state.settings


# Dependencies are injected to the APIs when the API uses the Depends(dependency_return_func) method 
# dependency_return_func gets and returns the dependencies
# This endpoint verifies if the app is alive
@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check endpoint."""

    return {"status": "ok"}


# This is root endpoint for human friendly response if someone visits
@app.get("/")
async def root() -> dict[str, str]:
    """Basic root endpoint for platform probes and quick sanity checks."""

    return {"service": "ContextFlow RAG API", "status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    # Parse parameters, dependecy injection
    file: Annotated[UploadFile, File(...)],
    settings: Settings = Depends(get_app_settings),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Upload, parse, chunk, embed, and index a single document."""

    file_bytes = await file.read()
    # Validate file size
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the maximum upload size.")

    # Extract text 
    pages = extract_pages(file.filename or "uploaded_document", file_bytes, file.content_type)
    # Chunk text
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
    # Embed chunks
    count = pipeline.ingest_chunks(stored_chunks)
    logger.info("Indexed %s chunks from %s", count, file.filename)
    return UploadResponse(document_name=file.filename or "uploaded_document", chunks_indexed=count, message="Document indexed successfully.")


@app.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """Answer a user question from indexed document context."""

    # Start answering with RAG
    result = pipeline.answer_question(payload.question)
    # Return answer + source chunk
    return QueryResponse(answer=result.answer, sources=result.sources)


# Handle the predictable exceptions we can get

@app.exception_handler(UnsupportedDocumentError)
async def unsupported_document_handler(_: Request, exc: UnsupportedDocumentError):
    return JSONResponse(status_code=400, content=ErrorResponse(detail=str(exc)).model_dump())


@app.exception_handler(EmptyDocumentError)
async def empty_document_handler(_: Request, exc: EmptyDocumentError):
    return JSONResponse(status_code=400, content=ErrorResponse(detail=str(exc)).model_dump())


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content=ErrorResponse(detail="Invalid request payload.", extra={"errors": exc.errors()}).model_dump())


@app.exception_handler(MissingConfigurationError)
async def missing_config_handler(_: Request, exc: MissingConfigurationError):
    return JSONResponse(status_code=503, content=ErrorResponse(detail=str(exc)).model_dump())


@app.exception_handler(LLMInvocationError)
async def llm_invocation_handler(_: Request, exc: LLMInvocationError):
    return JSONResponse(status_code=502, content=ErrorResponse(detail=str(exc)).model_dump())


@app.exception_handler(GroqAPIError)
async def groq_api_handler(_: Request, exc: GroqAPIError):
    logger.exception("Groq API error: %s", exc)
    return JSONResponse(status_code=502, content=ErrorResponse(detail="Groq API request failed.").model_dump())

# ContextFlow

ContextFlow is a lightweight document question-answering app built with FastAPI, Qdrant, Sentence Transformers, Groq, and Streamlit. You upload PDFs or TXT files, the app chunks and embeds the content, stores the chunks in Qdrant, and then answers questions with grounded source citations.

Live app: https://contextflowapi.streamlit.app/

## What It Does

- Upload and index PDF or TXT documents
- Chunk text with sentence-aware overlap
- Generate embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`)
- Store and retrieve chunks from Qdrant using cosine similarity
- Generate grounded answers with Groq
- Show returned sources alongside each answer
- Offer both a FastAPI backend and a Streamlit UI

## UX Highlights

The Streamlit UI is the primary user-facing app:

- Compact deployment/status cards at the top of the page
- Document upload and indexing
- Chat-style question answering
- Source chunk previews for every answer
- Resettable session history

## Architecture

The app is split into three layers:

- `app/main.py`: FastAPI endpoints and lifespan wiring
- `app/rag_pipeline.py`: retrieval-augmented generation orchestration
- `streamlit_app.py`: Streamlit interface for upload and chat

Supporting modules handle configuration, ingestion, embeddings, and vector storage:

- `app/config.py`: environment-driven settings
- `app/ingest.py`: document parsing and chunking
- `app/embeddings.py`: embedding generation
- `app/vectorstore.py`: Qdrant storage and retrieval

## Key Features

- PDF/TXT ingestion with clear validation errors
- Qdrant collection auto-repair when vector dimensions change
- Groq model normalization so deprecated aliases still work
- Streamlit embedded mode for standalone deployment
- Local development mode with a separate FastAPI backend

## Project Layout

```text
app/
  config.py
  embeddings.py
  ingest.py
  main.py
  rag_pipeline.py
  schemas.py
  vectorstore.py
streamlit_app.py
tests/
data/
docs/
```

## Requirements

- Python 3.11 or newer
- A Groq API key
- Optional: Qdrant server or Qdrant Cloud for durable storage

## Installation

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Set environment variables

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
$env:GROQ_MODEL="llama-3.3-70b-versatile"
```

Optional settings:

```powershell
$env:QDRANT_URL="http://127.0.0.1:6333"
$env:QDRANT_API="your_qdrant_api_key"
$env:QDRANT_COLLECTION="contextflow_chunks"
$env:CONTEXTFLOW_API_URL="http://127.0.0.1:8000"
```

## Local Development

### Backend only

```powershell
python -m uvicorn app.main:app --reload
```

FastAPI docs:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

### Streamlit UI with backend

Run the API first, then launch the UI:

```powershell
streamlit run streamlit_app.py
```

If the API is on a different host or port, set `CONTEXTFLOW_API_URL` before launching Streamlit.

## Standalone Deployment

The public deployment is hosted on Streamlit Community Cloud at https://contextflowapi.streamlit.app/.

For Streamlit Community Cloud, use embedded mode so the UI runs the RAG pipeline directly:

```powershell
$env:CONTEXTFLOW_RUNTIME="embedded"
$env:GROQ_API_KEY="your_groq_api_key"
$env:GROQ_MODEL="llama-3.3-70b-versatile"
```

In embedded mode, the Streamlit app can run without a separately hosted FastAPI service. This is the simplest free deployment path.

If you run a separate FastAPI backend, keep `CONTEXTFLOW_API_URL` pointed at that service and switch the UI back to client mode.

## Usage

### Upload a document

Use the Streamlit upload tab or call the API directly:

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@sample.pdf"
```

### Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the key points."}'
```

### Streamlit workflow

1. Upload a PDF or TXT file.
2. Wait for indexing to complete.
3. Ask a question in chat.
4. Review the answer and source chunks.

## API Reference

### `GET /health`

Returns a simple liveness response.

### `POST /upload`

Uploads a document, extracts text, chunks it, embeds it, and stores the chunks in Qdrant.

### `POST /query`

Accepts a JSON payload with a `question` string and returns a grounded answer plus source chunks.

## Configuration

Important environment variables:

- `GROQ_API_KEY`: required for answers
- `GROQ_MODEL`: optional, normalized to a supported model if deprecated aliases are used
- `QDRANT_URL`: optional Qdrant server URL; if empty, local path-backed storage is used
- `QDRANT_API`: optional Qdrant API key
- `QDRANT_COLLECTION`: collection name, defaults to `contextflow_chunks`
- `CONTEXTFLOW_RUNTIME`: set to `embedded` for standalone Streamlit execution
- `CONTEXTFLOW_API_URL`: API base URL for the Streamlit client mode

## Testing

```powershell
pytest
```

Targeted tests cover ingestion, vector search, and the RAG pipeline.

## Troubleshooting

- Upload returns 400: the file may be unreadable or unsupported.
- Query returns a Groq error: confirm `GROQ_API_KEY` and `GROQ_MODEL` are set.
- Upload fails after changing embeddings: the app will now rebuild the Qdrant collection automatically.
- Old local Qdrant data causes odd results: delete the local `data/` directory or use a fresh collection name.

## Deployment Notes

For long-lived production storage, run Qdrant as a service or use Qdrant Cloud. The local `data/` fallback is best for development, testing, and embedded deployments.

The current public deployment is available at https://contextflowapi.streamlit.app/.

If you deploy the Streamlit app publicly, remember to configure:

- `GROQ_API_KEY`
- `GROQ_MODEL`
- `CONTEXTFLOW_RUNTIME=embedded`

## License

No license file is currently included. Add one before distributing the project publicly.

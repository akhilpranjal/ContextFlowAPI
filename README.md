# ContextFlow RAG API

Production-style FastAPI Retrieval-Augmented Generation (RAG) backend for PDF/TXT ingestion and question answering with Groq.

## Features

- Upload and index PDF/TXT documents
- Sentence-aware chunking with overlap
- Embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`)
- Retrieval via Qdrant cosine similarity search
- Grounded answer generation via Groq
- API responses include answer and source chunks

## API Endpoints

- `GET /health` -> liveness check
- `POST /upload` -> upload and index a document
- `POST /query` -> ask a question over indexed docs

## Local Run

### 1) Install dependencies

```powershell
C:/Users/akhil/AppData/Local/Programs/Python/Python313/python.exe -m pip install -r requirements.txt
```

### 2) Set environment variables

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
$env:GROQ_MODEL="llama3-70b-8192"
```

You can also copy `.env.example` values into a local `.env` file.

### 3) Start server

```powershell
C:/Users/akhil/AppData/Local/Programs/Python/Python313/python.exe -m uvicorn app.main:app --reload
```

Docs UI: `http://127.0.0.1:8000/docs`

## Quick Usage

### Upload

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@sample.pdf"
```

### Query

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the key points."}'
```

## Free Deployment (Render)

This repo includes `render.yaml` for one-click blueprint deploy.

### Steps

1. Push this repository to GitHub.
2. In Render, choose **New +** -> **Blueprint**.
3. Select this repository.
4. Set `GROQ_API_KEY` in Render environment variables.
5. Deploy.

Render uses:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## Notes on Retrieval Storage

- Set `QDRANT_URL` to point at a running Qdrant instance, or leave it empty to use the local `data/`-backed Qdrant client for development and tests.
- Set `QDRANT_COLLECTION` if you want to override the default collection name (`contextflow_chunks`).
- For durable storage, run Qdrant as a service or use Qdrant Cloud so indexed chunks survive process restarts.

from __future__ import annotations

import html
import os
from typing import Any

import requests
import streamlit as st

from app.config import Settings, get_settings
from app.embeddings import EmbeddingService
from app.ingest import EmptyDocumentError, chunk_pages, extract_pages
from app.rag_pipeline import RAGPipeline
from app.vectorstore import QdrantVectorStore, StoredChunk

DEFAULT_API_BASE_URL = os.getenv("CONTEXTFLOW_API_URL", "http://127.0.0.1:8000")
RUNTIME_MODE = os.getenv("CONTEXTFLOW_RUNTIME", "auto").strip().lower()
LIVE_APP_URL = "https://contextflowapi.streamlit.app/"


st.set_page_config(
    page_title="ContextFlow Studio",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
    :root {
        --bg: #07111f;
        --panel: rgba(12, 24, 43, 0.82);
        --panel-strong: #0d1b30;
        --panel-soft: rgba(255, 255, 255, 0.04);
        --border: rgba(255, 255, 255, 0.08);
        --text: #edf3fb;
        --muted: #9fb0c7;
        --accent: #f59e0b;
        --accent-2: #22c55e;
        --danger: #fb7185;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(245, 158, 11, 0.18), transparent 34%),
            radial-gradient(circle at top right, rgba(34, 197, 94, 0.12), transparent 30%),
            linear-gradient(180deg, #091120 0%, #07111f 48%, #060c16 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(13, 27, 48, 0.95), rgba(7, 17, 31, 0.92));
        border-radius: 22px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
    }

    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
    }

    .header-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: flex-end;
    }

    .ghost-link {
        display: inline-flex;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 999px;
        padding: 0.45rem 0.8rem;
        color: var(--text);
        text-decoration: none;
        background: rgba(255, 255, 255, 0.03);
    }

    .ghost-link:hover {
        border-color: rgba(245, 158, 11, 0.5);
        color: var(--text);
    }

    .hero-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1rem;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 999px;
        padding: 0.48rem 0.8rem;
        background: rgba(255, 255, 255, 0.03);
        color: var(--muted);
        font-size: 0.9rem;
    }

    .hero-link {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin-top: 1rem;
        padding: 0.55rem 0.9rem;
        border-radius: 999px;
        border: 1px solid rgba(245, 158, 11, 0.35);
        background: rgba(245, 158, 11, 0.12);
        color: var(--text);
        font-weight: 700;
        text-decoration: none;
    }

    .hero-link:hover {
        border-color: rgba(245, 158, 11, 0.7);
        color: var(--text);
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.76rem;
        color: var(--accent);
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .title {
        font-size: 2.25rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 0;
        color: var(--text);
    }

    .subtitle {
        margin-top: 0.75rem;
        color: var(--muted);
        font-size: 1rem;
        max-width: 70ch;
    }

    .glass-card {
        border: 1px solid var(--border);
        border-radius: 18px;
        background: var(--panel);
        padding: 1rem;
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.2);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.18rem;
        font-weight: 700;
        line-height: 1.3;
    }

    .metric-caption {
        color: var(--muted);
        font-size: 0.88rem;
        margin-top: 0.35rem;
    }

    .section-heading {
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .small-note {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.03) !important;
    }

    .stButton button {
        border-radius: 999px;
        border: 1px solid rgba(245, 158, 11, 0.35);
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.94), rgba(251, 146, 60, 0.92));
        color: #07111f;
        font-weight: 800;
    }

    .stButton button:hover {
        border-color: rgba(245, 158, 11, 0.65);
        transform: translateY(-1px);
    }

    div[data-testid="stSidebar"] {
        display: none;
    }
</style>
""",
    unsafe_allow_html=True,
)


def normalize_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    return cleaned or DEFAULT_API_BASE_URL


def use_embedded_runtime() -> bool:
    if RUNTIME_MODE == "embedded":
        return True
    if RUNTIME_MODE == "api":
        return False
    return False


def api_url(base_url: str, path: str) -> str:
    return f"{normalize_base_url(base_url)}{path}"


def response_message(exc: requests.RequestException) -> str:
    if exc.response is not None:
        try:
            payload = exc.response.json()
        except ValueError:
            payload = exc.response.text.strip()
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message") or payload
            return f"{exc.response.status_code}: {detail}"
        if payload:
            return f"{exc.response.status_code}: {payload}"
        return f"{exc.response.status_code}: request failed"
    return str(exc)


def probe_health(base_url: str) -> dict[str, Any]:
    response = requests.get(api_url(base_url, "/health"), timeout=5)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=15)
def cached_health(base_url: str) -> dict[str, Any]:
    return probe_health(base_url)


@st.cache_resource
def get_local_pipeline() -> tuple[Settings, RAGPipeline]:
    settings = get_settings()
    embedding_service = EmbeddingService(settings)
    vector_store = QdrantVectorStore(settings)
    pipeline = RAGPipeline(settings, embedding_service, vector_store)
    return settings, pipeline


def upload_document_embedded(uploaded_file: Any) -> dict[str, Any]:
    settings, pipeline = get_local_pipeline()
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise ValueError("File exceeds the maximum upload size.")

    pages = extract_pages(uploaded_file.name or "uploaded_document", file_bytes, uploaded_file.type)
    chunks = chunk_pages(pages, settings, document_key=uploaded_file.name or "uploaded_document")
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
    return {
        "document_name": uploaded_file.name or "uploaded_document",
        "chunks_indexed": count,
        "message": "Document indexed successfully.",
    }


def ask_question_embedded(question: str) -> dict[str, Any]:
    _, pipeline = get_local_pipeline()
    result = pipeline.answer_question(question)
    return {
        "answer": result.answer,
        "sources": [source.model_dump() for source in result.sources],
    }


def upload_document(base_url: str, uploaded_file: Any) -> dict[str, Any]:
    if use_embedded_runtime():
        return upload_document_embedded(uploaded_file)
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    response = requests.post(api_url(base_url, "/upload"), files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def ask_question(base_url: str, question: str) -> dict[str, Any]:
    if use_embedded_runtime():
        return ask_question_embedded(question)
    response = requests.post(
        api_url(base_url, "/query"),
        json={"question": question},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def render_metric(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-caption">{html.escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def preview_text(text: str, limit: int = 260) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            page_number = source.get("page_number")
            page_label = f"Page {page_number}" if page_number is not None else "TXT source"
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom: 0.75rem;">
                    <div class="metric-label">Source {index}</div>
                    <div class="metric-value" style="font-size: 1rem;">{html.escape(str(source.get('document_name', 'Unknown document')))}</div>
                    <div class="metric-caption">{html.escape(page_label)} · chunk {html.escape(str(source.get('chunk_index', 0)))} · score {float(source.get('score', 0.0)):.3f}</div>
                    <div style="margin-top: 0.55rem; color: var(--text); line-height: 1.55;">{html.escape(preview_text(str(source.get('content', ''))))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload a PDF or TXT file, then ask a question about the indexed content.",
        }
    ]

if "last_upload" not in st.session_state:
    st.session_state.last_upload = None

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_BASE_URL

if use_embedded_runtime():
    api_status = "Embedded runtime"
else:
    try:
        health_data = cached_health(normalize_base_url(st.session_state.api_base_url))
        api_status = f"Connected ({health_data.get('status', 'ok')})"
    except requests.RequestException:
        api_status = "Disconnected"


st.markdown(
    """
    <div class="hero">
        <div class="header-bar">
            <div>
                <div class="eyebrow">Document intelligence</div>
                <h1 class="title">ContextFlow Studio</h1>
            </div>
            <div class="header-actions">
                <a class="ghost-link" href="https://contextflowapi.streamlit.app/" target="_blank" rel="noopener noreferrer">Live app</a>
                <a class="ghost-link" href="#guide">Guide</a>
            </div>
        </div>
        <p class="subtitle">
            Upload documents, index them in Qdrant, and ask grounded questions from a clean Streamlit interface.
            The public deployment runs on Streamlit Cloud and keeps the RAG flow embedded.
        </p>
        <div class="hero-row">
            <div class="pill">Deployment: Streamlit Cloud</div>
            <div class="pill">Runtime: %s</div>
            <div class="pill">Live URL: contextflowapi.streamlit.app</div>
        </div>
        <a class="hero-link" href="https://contextflowapi.streamlit.app/" target="_blank" rel="noopener noreferrer">
            Open the live app
        </a>
    </div>
    """ % html.escape(api_status),
    unsafe_allow_html=True,
)

columns = st.columns(3)
with columns[0]:
    render_metric("API status", api_status, "Health check against the running FastAPI service")
with columns[1]:
    render_metric("Upload types", "PDF / TXT", "The API accepts supported document types and rejects others")
with columns[2]:
    render_metric("Answer mode", "Grounded", "Responses come from retrieval plus Groq-generated synthesis")

top_actions = st.columns([1, 1, 1])
with top_actions[0]:
    if st.button("Reset chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Upload a PDF or TXT file, then ask a question about the indexed content.",
            }
        ]
        st.rerun()
with top_actions[1]:
    st.markdown(f'<a class="ghost-link" href="{LIVE_APP_URL}" target="_blank" rel="noopener noreferrer">Open public deployment</a>', unsafe_allow_html=True)
with top_actions[2]:
    if not use_embedded_runtime():
        st.caption("Client mode still works via CONTEXTFLOW_API_URL for local setups.")

tab_chat, tab_upload, tab_guide = st.tabs(["Chat", "Upload", "Guide"])

with tab_upload:
    st.markdown('<div class="section-heading">Index a document</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">Choose a local file, send it to <code>/upload</code>, and the backend will chunk, embed, and store it in Qdrant.</div>',
        unsafe_allow_html=True,
    )

    upload_col, info_col = st.columns([1.2, 0.8])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a PDF or TXT file",
            type=["pdf", "txt"],
            accept_multiple_files=False,
        )
        upload_clicked = st.button("Upload and index", use_container_width=True, disabled=uploaded_file is None)

    with info_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="metric-label">What happens next</div>
                <div class="metric-value">Chunk, embed, store</div>
                <div class="metric-caption">
                    The backend extracts page text, performs sentence-aware chunking, generates embeddings,
                    and writes the chunks into Qdrant.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if upload_clicked and uploaded_file is not None:
        try:
            with st.spinner("Uploading and indexing document..."):
                result = upload_document(normalize_base_url(st.session_state.api_base_url), uploaded_file)
            st.session_state.last_upload = result
            st.success(result.get("message", "Document indexed."))
            upload_details_col, upload_meta_col = st.columns([1.2, 0.8])
            with upload_details_col:
                st.markdown("**Upload result**")
                st.json(result)
            with upload_meta_col:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="metric-label">Document</div>
                        <div class="metric-value">{html.escape(str(result.get('document_name', uploaded_file.name)))}</div>
                        <div class="metric-caption">{html.escape(str(result.get('chunks_indexed', 0)))} chunks indexed</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except requests.RequestException as exc:
            st.error(response_message(exc))
        except (EmptyDocumentError, ValueError) as exc:
            st.error(str(exc))

    if st.session_state.last_upload:
        st.markdown("### Last indexed document")
        st.json(st.session_state.last_upload)

with tab_chat:
    st.markdown('<div class="section-heading">Ask a question</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">The API will retrieve the most relevant chunks and generate a grounded answer. Source chunks are shown under each response.</div>',
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    prompt = st.chat_input("Ask about your indexed documents")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and generating an answer..."):
                    response = ask_question(normalize_base_url(st.session_state.api_base_url), prompt)
                answer = response.get("answer", "No answer returned.")
                st.markdown(answer)
                render_sources(response.get("sources", []))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": response.get("sources", []),
                }
            )
        except requests.RequestException as exc:
            error_text = response_message(exc)
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_text}"})
            with st.chat_message("assistant"):
                st.error(error_text)

with tab_guide:
    st.markdown('<div id="guide"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">How to run the stack</div>', unsafe_allow_html=True)
    guide_col, notes_col = st.columns([1, 1])
    with guide_col:
        st.markdown("**Live app**")
        st.markdown(f'[Open the live app]({LIVE_APP_URL})')
        st.markdown("**Local backend**")
        st.code("python -m uvicorn app.main:app --reload", language="powershell")
        st.markdown("**Local Streamlit UI**")
        st.code("streamlit run streamlit_app.py", language="powershell")

    with notes_col:
        st.markdown("**Expected environment variables**")
        st.code(
            "\n".join(
                [
                    '$env:GROQ_API_KEY="your_groq_api_key"',
                    '$env:GROQ_MODEL="llama-3.3-70b-versatile"',
                    '$env:CONTEXTFLOW_API_URL="http://127.0.0.1:8000"',
                ]
            ),
            language="powershell",
        )

    st.markdown("**Behavior**")
    st.write("• Uploading uses the `/upload` endpoint and stores chunks in Qdrant.")
    st.write("• Asking a question uses the `/query` endpoint and renders the answer with source chunk previews.")
    st.write("• If the backend returns an error, the UI shows the API response instead of swallowing it.")

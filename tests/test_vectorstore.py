import numpy as np

from app.config import Settings
from app.vectorstore import QdrantVectorStore, StoredChunk


def test_vectorstore_returns_best_match():
    settings = Settings(qdrant_url="", qdrant_collection="test_vectorstore_returns_best_match")
    store = QdrantVectorStore(settings)
    chunks = [
        StoredChunk(source_id="doc:p1:c0", document_name="doc.txt", page_number=None, chunk_index=0, content="alpha beta"),
        StoredChunk(source_id="doc:p1:c1", document_name="doc.txt", page_number=None, chunk_index=1, content="gamma delta"),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.add(chunks, embeddings)

    results = store.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=1)
    assert results[0].payload["source_id"] == "doc:p1:c0"
    assert results[0].score > 0.9

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass

import faiss
import numpy as np

from .config import Settings
from .schemas import SourceChunk


@dataclass(slots=True)
class StoredChunk:
    source_id: str
    document_name: str
    page_number: int | None
    chunk_index: int
    content: str


class FaissVectorStore:
    """Store chunk embeddings in FAISS with cosine similarity search."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self._index: faiss.Index | None = None
        self._records: list[StoredChunk] = []
        self._dimension: int | None = None
        if self.settings.enable_faiss_persistence:
            self._load_if_available()

    def add(self, chunks: list[StoredChunk], embeddings: np.ndarray) -> None:
        """Add chunk records and their normalized embeddings to the FAISS index."""

        if len(chunks) == 0:
            return
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        with self._lock:
            self._ensure_index(embeddings.shape[1])
            assert self._index is not None
            self._index.add(embeddings)
            self._records.extend(chunks)
            if self.settings.enable_faiss_persistence:
                self._persist()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[StoredChunk, float]]:
        """Return top-k records ranked by cosine similarity."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            query = np.asarray(query_embedding, dtype=np.float32)
            if query.ndim == 1:
                query = query.reshape(1, -1)
            scores, indices = self._index.search(query, min(top_k, self._index.ntotal))
            results: list[tuple[StoredChunk, float]] = []
            for score, index in zip(scores[0], indices[0], strict=False):
                if index < 0:
                    continue
                record = self._records[index]
                results.append((record, float(score)))
            return results

    def _ensure_index(self, dimension: int) -> None:
        """Initialize index lazily and prevent mixed embedding dimensions."""

        if self._index is None:
            self._dimension = dimension
            self._index = faiss.IndexFlatIP(dimension)
            return
        if self._dimension != dimension:
            raise ValueError(f"Embedding dimension changed from {self._dimension} to {dimension}")

    def _persist(self) -> None:
        """Persist FAISS index and metadata to local disk."""

        assert self._index is not None
        faiss.write_index(self._index, str(self.settings.faiss_index_path))
        payload = [asdict(record) for record in self._records]
        self.settings.faiss_metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_if_available(self) -> None:
        """Load persisted FAISS index and metadata if both are present."""

        if not self.settings.faiss_index_path.exists() or not self.settings.faiss_metadata_path.exists():
            return
        index = faiss.read_index(str(self.settings.faiss_index_path))
        metadata = json.loads(self.settings.faiss_metadata_path.read_text(encoding="utf-8"))
        self._index = index
        self._dimension = index.d
        self._records = [StoredChunk(**item) for item in metadata]

    @staticmethod
    def to_source_chunks(results: list[tuple[StoredChunk, float]]) -> list[SourceChunk]:
        """Convert internal store records into API response schema objects."""

        return [
            SourceChunk(
                source_id=record.source_id,
                document_name=record.document_name,
                page_number=record.page_number,
                chunk_index=record.chunk_index,
                score=score,
                content=record.content,
            )
            for record, score in results
        ]

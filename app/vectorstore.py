from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import Settings
from .schemas import SourceChunk


@dataclass(slots=True)
class StoredChunk:
    source_id: str
    document_name: str
    page_number: int | None
    chunk_index: int
    content: str


class QdrantVectorStore:
    """Store chunk embeddings in Qdrant with cosine similarity search."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self.collection_name = settings.qdrant_collection
        self.client = client or self._build_client(settings)

    @staticmethod
    def _build_client(settings: Settings) -> QdrantClient:
        if settings.qdrant_url:
            return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api or None)
        return QdrantClient(path="data")

    def ensure_collection(self, dimension: int) -> None:
        collections = self.client.get_collections().collections
        names = {collection.name for collection in collections}

        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )

    def add(self, chunks: list[StoredChunk], embeddings: np.ndarray) -> None:
        """Add chunk records and their normalized embeddings to Qdrant."""

        vector_array = np.asarray(embeddings, dtype=np.float32)
        if vector_array.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(chunks) != len(vector_array):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        self.ensure_collection(vector_array.shape[1])

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "source_id": chunk.source_id,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                },
            )
            for chunk, vector in zip(chunks, vector_array)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[PointStruct]:
        """Return top-k records ranked by cosine similarity."""

        query_array = np.asarray(query_embedding, dtype=np.float32)
        if query_array.ndim == 2:
            if query_array.shape[0] != 1:
                raise ValueError("query_embedding must describe a single query vector")
            query_array = query_array[0]
        elif query_array.ndim != 1:
            raise ValueError("query_embedding must be a 1D or 2D array")

        collections = {collection.name for collection in self.client.get_collections().collections}
        if self.collection_name not in collections:
            return []

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_array.tolist(),
            limit=top_k,
        )
        return list(results.points)

    @staticmethod
    def to_source_chunks(results: list[PointStruct]) -> list[SourceChunk]:
        """Convert Qdrant search results to API source chunks."""

        sources: list[SourceChunk] = []
        for result in results:
            payload = result.payload or {}
            page_number = payload.get("page_number")
            if page_number in {None, ""}:
                normalized_page_number = None
            else:
                normalized_page_number = int(page_number)

            sources.append(
                SourceChunk(
                    source_id=str(payload.get("source_id", "")),
                    document_name=str(payload.get("document_name", "")),
                    page_number=normalized_page_number,
                    chunk_index=int(payload.get("chunk_index", 0)),
                    score=float(result.score or 0.0),
                    content=str(payload.get("content", "")),
                )
            )
        return sources

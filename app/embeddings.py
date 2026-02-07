from __future__ import annotations

from collections import OrderedDict
import threading
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings


class EmbeddingService:
    """Embed text using a locally hosted sentence-transformer model."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: SentenceTransformer | None = None
        self._model_lock = threading.Lock()
        self._cache_lock = threading.RLock()
        # Ordered dict gives a lightweight LRU cache policy for repeated text chunks.
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @property
    def model(self) -> SentenceTransformer:
        """Lazily load and cache the sentence-transformer model."""

        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = SentenceTransformer(self.settings.embedding_model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings for cosine similarity search."""

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: list[np.ndarray | None] = []
        missing: list[str] = []
        missing_positions: list[int] = []

        for index, text in enumerate(texts):
            cached = self._get_cached_embedding(text) if self.settings.enable_embedding_cache else None
            if cached is not None:
                vectors.append(cached)
            else:
                vectors.append(None)
                missing.append(text)
                missing_positions.append(index)

        if missing:
            encoded = self.model.encode(missing, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            for position, text, vector in zip(missing_positions, missing, encoded, strict=False):
                array = np.asarray(vector, dtype=np.float32)
                vectors[position] = array
                if self.settings.enable_embedding_cache:
                    self._put_cached_embedding(text, array)

        return np.vstack([item for item in vectors if item is not None]).astype(np.float32)

    def _get_cached_embedding(self, text: str) -> np.ndarray | None:
        with self._cache_lock:
            value = self._cache.get(text)
            if value is None:
                return None
            self._cache.move_to_end(text)
            return value

    def _put_cached_embedding(self, text: str, value: np.ndarray) -> None:
        with self._cache_lock:
            self._cache[text] = value
            self._cache.move_to_end(text)
            while len(self._cache) > self.settings.embedding_cache_max_items:
                self._cache.popitem(last=False)

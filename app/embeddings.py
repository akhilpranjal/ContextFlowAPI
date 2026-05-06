from __future__ import annotations

from typing import Any
import numpy as np

from .config import Settings


class EmbeddingService:
    """Embed text using a locally hosted sentence-transformer model.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        """Lazily load and cache the sentence-transformer model."""

        if self._model is None:
            # Import lazily to avoid heavy torch/sentence-transformers import cost at app boot.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embedding_model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings for cosine similarity search."""

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        encoded = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        arr = np.asarray(encoded, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

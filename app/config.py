from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_name: str = "ContextFlow RAG API"
    api_prefix: str = "/v1"

    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-70b-8192", validation_alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.2, validation_alias="GROQ_TEMPERATURE")
    groq_max_tokens: int = Field(default=900, validation_alias="GROQ_MAX_TOKENS")

    embedding_model_name: str = Field(default="all-MiniLM-L6-v2", validation_alias="EMBEDDING_MODEL_NAME")

    chunk_size: int = Field(default=1200, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, validation_alias="CHUNK_OVERLAP")
    top_k: int = Field(default=4, validation_alias="TOP_K")

    max_upload_size_mb: int = Field(default=20, validation_alias="MAX_UPLOAD_SIZE_MB")
    enable_embedding_cache: bool = Field(default=True, validation_alias="ENABLE_EMBEDDING_CACHE")
    embedding_cache_max_items: int = Field(default=5000, validation_alias="EMBEDDING_CACHE_MAX_ITEMS")
    enable_faiss_persistence: bool = Field(default=False, validation_alias="ENABLE_FAISS_PERSISTENCE")

    data_dir: Path = Field(default=Path("data"), validation_alias="DATA_DIR")
    faiss_index_path: Path = Field(default=Path("data/faiss.index"), validation_alias="FAISS_INDEX_PATH")
    faiss_metadata_path: Path = Field(default=Path("data/faiss_metadata.json"), validation_alias="FAISS_METADATA_PATH")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @property
    def max_upload_size_bytes(self) -> int:
        """Return maximum upload size in bytes."""

        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance for the application."""

    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.faiss_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    return settings

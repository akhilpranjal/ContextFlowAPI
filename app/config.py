from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# -> To manage application configuration by automatically loading the values from environment 
# variables, .env files and other resources
# -> Automatic Loading: If a field is not provided during initialization, Pydantic searches 
# for a matching environment variable (case-insensitive by default).
# -> Type Validation: Like standard Pydantic models, it enforces type hints and 
# validates data as it is loaded.
# -> Source Priority: It typically prioritizes values in the following order: 
# init arguments > environment variables > .env file > default values. 

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
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
    qdrant_api: str = Field(default="", validation_alias="QDRANT_API_KEY")
    qdrant_url: str = Field(default="", validation_alias="QDRANT_URL")
    qdrant_collection: str = Field(default="contextflow_chunks", validation_alias="QDRANT_COLLECTION")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @property
    def max_upload_size_bytes(self) -> int:
        """Return maximum upload size in bytes."""

        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance for the application."""

    return Settings()

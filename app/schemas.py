# Any is used to tell the static type checkers to skip type validation for this value
from typing import Any

# BaseModel: Container that defines the schema and handles 
# validation/serialization
# Field: Attribute Optimizer customizes specific model attributes, 
# adds validation rules (like length ranges) and defines default values
from pydantic import BaseModel, Field


# Type checking and data validation for requests and responses in the app
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user question to answer.")


class SourceChunk(BaseModel):
    source_id: str
    document_name: str
    page_number: int | None = None
    chunk_index: int
    score: float
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class UploadResponse(BaseModel):
    document_name: str
    chunks_indexed: int
    message: str


class ErrorResponse(BaseModel):
    detail: str
    extra: dict[str, Any] | None = None

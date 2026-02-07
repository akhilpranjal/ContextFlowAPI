import numpy as np

from app.config import Settings
from app.rag_pipeline import MissingConfigurationError, RAGPipeline
from app.vectorstore import FaissVectorStore, StoredChunk


class FakeEmbeddingService:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 1 and texts[0] == "what is alpha":
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeGroqClient:
    def __init__(self):
        class _Completions:
            @staticmethod
            def create(**_: object):
                class _Message:
                    content = "Grounded answer"

                class _Choice:
                    message = _Message()

                class _Response:
                    choices = [_Choice()]

                return _Response()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_answer_question_requires_groq_key():
    settings = Settings(groq_api_key="")
    pipeline = RAGPipeline(settings, FakeEmbeddingService(), FaissVectorStore(settings))
    try:
        pipeline.answer_question("test")
        assert False, "Expected MissingConfigurationError"
    except MissingConfigurationError:
        assert True


def test_answer_question_returns_fallback_when_no_sources():
    settings = Settings(groq_api_key="dummy")
    pipeline = RAGPipeline(settings, FakeEmbeddingService(), FaissVectorStore(settings))
    pipeline.client = FakeGroqClient()

    response = pipeline.answer_question("what is alpha")
    assert response.sources == []
    assert "could not find relevant context" in response.answer.lower()


def test_answer_question_returns_answer_with_sources():
    settings = Settings(groq_api_key="dummy", top_k=1)
    vector_store = FaissVectorStore(settings)
    pipeline = RAGPipeline(settings, FakeEmbeddingService(), vector_store)
    pipeline.client = FakeGroqClient()

    vector_store.add(
        [
            StoredChunk(
                source_id="doc:p1:c0",
                document_name="doc.txt",
                page_number=None,
                chunk_index=0,
                content="alpha is the first item",
            )
        ],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    response = pipeline.answer_question("what is alpha")
    assert response.answer == "Grounded answer"
    assert len(response.sources) == 1

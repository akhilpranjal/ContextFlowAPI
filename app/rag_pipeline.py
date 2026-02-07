from __future__ import annotations

from dataclasses import dataclass

from groq import Groq

from .config import Settings
from .embeddings import EmbeddingService
from .schemas import SourceChunk
from .vectorstore import FaissVectorStore, StoredChunk


@dataclass(slots=True)
class AnswerPayload:
    """Response payload returned by the RAG pipeline."""

    answer: str
    sources: list[SourceChunk]


class MissingConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing."""


class LLMInvocationError(RuntimeError):
    """Raised when the LLM provider call fails."""


class RAGPipeline:
    """Coordinate retrieval and grounded answer generation."""

    def __init__(self, settings: Settings, embedding_service: EmbeddingService, vector_store: FaissVectorStore):
        self.settings = settings
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.client: Groq | None = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None

    def ingest_chunks(self, chunks: list[StoredChunk]) -> int:
        """Embed and index chunk records in the vector store."""

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        self.vector_store.add(chunks, embeddings)
        return len(chunks)

    def answer_question(self, question: str) -> AnswerPayload:
        """Retrieve relevant chunks and generate a grounded answer."""

        if not self.client:
            raise MissingConfigurationError("GROQ_API_KEY is not configured.")

        question_embedding = self.embedding_service.embed_texts([question])
        retrieved = self.vector_store.search(question_embedding, self.settings.top_k)
        sources = self.vector_store.to_source_chunks(retrieved)
        if not sources:
            return AnswerPayload(
                answer="I could not find relevant context in the indexed documents to answer this question.",
                sources=[],
            )
        prompt = self._build_prompt(question, sources)
        answer = self._call_llm(prompt)
        return AnswerPayload(answer=answer, sources=sources)

    def _build_prompt(self, question: str, sources: list[SourceChunk]) -> str:
        """Construct a grounded, instruction-focused prompt for Groq."""

        context_lines = []
        for item in sources:
            page_label = f"page {item.page_number}" if item.page_number is not None else "txt source"
            context_lines.append(f"[{item.source_id} | {item.document_name} | {page_label}] {item.content}")

        context_block = "\n\n".join(context_lines) if context_lines else "No source context was retrieved."
        return (
            "You are ContextFlow, a direct and careful document answerer. "
            "Use only the supplied context. If the answer is not supported, say so plainly. "
            "Prefer concrete wording over generic filler.\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION:\n{question}\n\n"
            "OUTPUT RULES:\n"
            "1. Give a short, grounded answer.\n"
            "2. Mention uncertainty explicitly when evidence is thin.\n"
            "3. Do not invent facts or sources."
        )

    def _call_llm(self, prompt: str) -> str:
        """Call Groq and normalize the returned answer text."""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.groq_model,
                messages=[
                    {"role": "system", "content": "Answer from the provided context only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.groq_temperature,
                max_tokens=self.settings.groq_max_tokens,
            )
        except Exception as exc:  # pragma: no cover - depends on external API behavior
            raise LLMInvocationError("Groq request failed. Please retry.") from exc
        message = response.choices[0].message.content or ""
        return message.strip()

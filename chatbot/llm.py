from collections.abc import Iterator, Sequence

import numpy as np
from openai import OpenAI

from chatbot.config import Settings
from chatbot.index import Embedder, SearchResult

SYSTEM_PROMPT = """You answer questions about the user's documents using only the excerpts provided.
- If the excerpts do not contain the answer, say you could not find it in the documents.
  Do not guess or use outside knowledge.
- Cite the source for every fact in the form [file.pdf p. 3].
- Be concise and use plain language."""

HISTORY_TURNS = 6


def make_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def make_embedder(client: OpenAI, model: str) -> Embedder:
    def embed(texts: Sequence[str]) -> np.ndarray:
        response = client.embeddings.create(model=model, input=list(texts))
        return np.array([item.embedding for item in response.data], dtype=np.float32)

    return embed


def format_context(results: Sequence[SearchResult]) -> str:
    return "\n\n".join(f"[{r.chunk.source} p. {r.chunk.page}]\n{r.chunk.text}" for r in results)


def build_messages(question: str, results: Sequence[SearchResult], history: Sequence[dict]) -> list[dict]:
    """System prompt, recent conversation for follow-ups, then the question with its excerpts."""
    context = format_context(results) or "(no relevant excerpts found)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-HISTORY_TURNS:],
        {"role": "user", "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}"},
    ]


def stream_answer(client: OpenAI, model: str, messages: list[dict]) -> Iterator[str]:
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content

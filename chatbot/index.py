from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from chatbot.chunking import Chunk

Embedder = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class VectorIndex:
    """In-memory cosine-similarity index over document chunks."""

    def __init__(self, chunks: Sequence[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = list(chunks)
        self._vectors = _normalise(np.asarray(vectors, dtype=np.float32))

    @classmethod
    def build(cls, chunks: Sequence[Chunk], embed: Embedder, batch_size: int = 64) -> "VectorIndex":
        if not chunks:
            return cls([], np.zeros((0, 0), dtype=np.float32))
        batches = [
            embed([c.text for c in chunks[i : i + batch_size]]) for i in range(0, len(chunks), batch_size)
        ]
        return cls(chunks, np.vstack(batches))

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        if not self.chunks or k <= 0:
            return []
        query = _normalise(np.asarray(query_vector, dtype=np.float32))
        scores = self._vectors @ query
        top = np.argsort(-scores)[:k]
        return [SearchResult(chunk=self.chunks[i], score=float(scores[i])) for i in top]

import numpy as np

from chatbot.chunking import Chunk
from chatbot.index import VectorIndex

VOCAB = ["refund", "shipping", "warranty", "password"]


def bag_of_words(texts):
    return np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=np.float32)


CHUNKS = [
    Chunk("faq.pdf", 1, "Refund requests are accepted for 30 days. Refund goes to the original card."),
    Chunk("faq.pdf", 2, "Shipping takes 3 to 5 working days."),
    Chunk("faq.pdf", 3, "Reset your password from the account page."),
]


def test_search_returns_most_similar_chunk_first():
    index = VectorIndex.build(CHUNKS, bag_of_words)

    results = index.search(bag_of_words(["How do I get a refund?"])[0], k=2)

    assert results[0].chunk.page == 1
    assert results[0].score > results[1].score
    assert len(results) == 2


def test_build_embeds_in_batches():
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return bag_of_words(texts)

    VectorIndex.build(CHUNKS, embed, batch_size=2)

    assert calls == [2, 1]


def test_empty_index_returns_no_results():
    index = VectorIndex.build([], bag_of_words)

    assert index.search(np.ones(len(VOCAB)), k=3) == []


def test_zero_vector_does_not_crash():
    index = VectorIndex.build(CHUNKS, bag_of_words)

    results = index.search(np.zeros(len(VOCAB)), k=1)

    assert len(results) == 1
    assert results[0].score == 0.0

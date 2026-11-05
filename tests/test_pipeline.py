import io

import numpy as np

from chatbot.chunking import chunk_pages
from chatbot.index import VectorIndex
from chatbot.llm import build_messages
from chatbot.pdf import read_pdf

VOCAB = ["refund", "shipping", "password"]


def bag_of_words(texts):
    return np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=np.float32)


def test_question_retrieves_the_right_page_and_cites_it(pdf_bytes):
    data = pdf_bytes(
        [
            "Shipping takes 3 to 5 working days.",
            "Reset your password from the account page.",
            "A refund is issued within 14 days of the return.",
        ]
    )
    chunks = chunk_pages(read_pdf(io.BytesIO(data), "help.pdf"))
    index = VectorIndex.build(chunks, bag_of_words)

    results = index.search(bag_of_words(["When will I get my refund?"])[0], k=1)
    prompt = build_messages("When will I get my refund?", results, history=[])[-1]["content"]

    assert results[0].chunk.page == 3
    assert "[help.pdf p. 3]" in prompt
    assert "within 14 days" in prompt

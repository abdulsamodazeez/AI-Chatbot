from itertools import pairwise

import pytest

from chatbot.chunking import chunk_pages
from chatbot.pdf import Page


def test_short_page_is_one_chunk():
    chunks = chunk_pages([Page("a.pdf", 3, "  Hello\n world  ")], size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"
    assert (chunks[0].source, chunks[0].page) == ("a.pdf", 3)


def test_long_page_is_split_on_whitespace_with_overlap():
    words = [f"word{i}" for i in range(200)]
    chunks = chunk_pages([Page("a.pdf", 1, " ".join(words))], size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    for chunk in chunks:
        assert all(token in words for token in chunk.text.split())
    for left, right in pairwise(chunks):
        assert left.text.split()[-1] in right.text.split()[:4]


def test_chunks_never_span_pages():
    pages = [Page("a.pdf", 1, "alpha " * 50), Page("a.pdf", 2, "beta " * 50)]

    chunks = chunk_pages(pages, size=80, overlap=10)

    assert {c.page for c in chunks if "alpha" in c.text} == {1}
    assert {c.page for c in chunks if "beta" in c.text} == {2}


def test_text_without_spaces_still_terminates():
    chunks = chunk_pages([Page("a.pdf", 1, "x" * 250)], size=100, overlap=20)

    assert "".join(c.text for c in chunks).count("x") >= 250


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_rejects_invalid_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], size=size, overlap=overlap)

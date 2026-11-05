from collections.abc import Iterable
from dataclasses import dataclass

from chatbot.pdf import Page


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int
    text: str


def chunk_pages(pages: Iterable[Page], size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Split each page into overlapping chunks, breaking on whitespace where possible.

    Chunks never span pages, so every chunk can be cited by a single page number.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")

    chunks = []
    for page in pages:
        text = " ".join(page.text.split())
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                cut = text.rfind(" ", start, end)
                if cut > start:
                    end = cut
            chunks.append(Chunk(source=page.source, page=page.number, text=text[start:end].strip()))
            if end >= len(text):
                break
            boundary = text.find(" ", end - overlap, end)
            start = max(boundary + 1 if boundary != -1 else end, start + 1)
    return chunks

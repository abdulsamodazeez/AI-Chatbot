from dataclasses import dataclass
from typing import BinaryIO

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    source: str
    number: int
    text: str


def read_pdf(file: BinaryIO, source: str) -> list[Page]:
    """Extract text page by page, skipping pages with no extractable text (e.g. scans)."""
    reader = PdfReader(file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(source=source, number=number, text=text))
    return pages

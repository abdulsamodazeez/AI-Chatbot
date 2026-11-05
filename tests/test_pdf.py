import io

from chatbot.pdf import read_pdf


def test_reads_text_with_page_numbers(pdf_bytes):
    data = pdf_bytes(["Refunds are processed within 14 days.", "Shipping is free over 50 euros."])

    pages = read_pdf(io.BytesIO(data), "policy.pdf")

    assert [(p.source, p.number) for p in pages] == [("policy.pdf", 1), ("policy.pdf", 2)]
    assert "14 days" in pages[0].text
    assert "free over 50" in pages[1].text


def test_skips_pages_without_text(pdf_bytes):
    data = pdf_bytes(["First page.", "", "Third page."])

    pages = read_pdf(io.BytesIO(data), "doc.pdf")

    assert [p.number for p in pages] == [1, 3]

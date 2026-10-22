from pathlib import Path

from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_parser import _join_lines, parse_pdf

HI_PDF = Path(__file__).resolve().parents[2] / "hi.pdf"


def test_dehyphenation_and_ligatures():
    assert _join_lines(["a transfor-", "mer model"]) == "a transformer model"
    assert _join_lines(["state-of-the-art", "results"]) == "state-of-the-art results"
    assert _join_lines(["ﬁne ﬂow"]) == "fine flow"


def test_title_and_headings(paper_pdf):
    doc = parse_pdf(paper_pdf)
    assert doc.title == "Sparse Attention for Long Documents"
    headings = [(b.text, b.level) for b in doc.blocks if b.kind == "heading"]
    texts = [h[0] for h in headings]
    assert texts == ["Abstract", "1 Introduction", "2 Method", "2.1 Sparse Pattern", "3 Results", "References"]
    levels = dict(headings)
    assert levels["2 Method"] == 1 and levels["2.1 Sparse Pattern"] == 2


def test_running_header_and_page_numbers_removed(paper_pdf):
    doc = parse_pdf(paper_pdf)
    assert not any("Journal of Testing" in b.text for b in doc.blocks)
    assert not any(b.text.strip() in {"1", "2", "3", "4"} for b in doc.blocks)


def test_cross_page_paragraph_rejoined(paper_pdf):
    doc = parse_pdf(paper_pdf)
    joined = [b for b in doc.blocks if "crucial for long documents." in b.text]
    assert len(joined) == 1 and joined[0].kind == "paragraph"


def test_table_detected_and_caption_kept(paper_pdf):
    doc = parse_pdf(paper_pdf)
    tables = [b for b in doc.blocks if b.kind == "table"]
    assert len(tables) == 1 and "\n" in tables[0].text and "28.9" in tables[0].text
    assert any(b.kind == "caption" and b.text.startswith("Table 1") for b in doc.blocks)


def test_references_flagged(paper_pdf):
    doc = parse_pdf(paper_pdf)
    ref = [b for b in doc.blocks if "Vaswani" in b.text]
    assert ref and all(b.in_references for b in ref)
    assert not any(b.in_references for b in doc.blocks if "variant" in b.text)


def test_two_column_reading_order(two_column_pdf):
    doc = parse_pdf(two_column_pdf)
    order = [b.text[:7].rstrip(".") for b in doc.blocks if b.kind == "paragraph"]
    assert order == ["LEFT-0", "LEFT-1", "LEFT-2", "RIGHT-0", "RIGHT-1", "RIGHT-2"]


def test_end_to_end_chunking_of_pdf(paper_pdf, counter):
    doc = parse_pdf(paper_pdf)
    chunks, stats = chunk_document(doc, counter)
    paths = {c.section_path for c in chunks}
    assert "2 Method > 2.1 Sparse Pattern" in paths
    assert all("References" not in c.section_path for c in chunks)
    assert stats.over_limit == 0
    assert any(c.kind == "table" for c in chunks)
    assert all("Journal of Testing" not in c.text for c in chunks)


def test_scanned_pdf_is_flagged_not_silently_empty():
    if not HI_PDF.exists():
        return
    doc = parse_pdf(HI_PDF)
    assert doc.needs_ocr and doc.blocks == []

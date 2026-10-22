import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.tokens import ApproxTokenCounter  # noqa: E402


@pytest.fixture
def counter():
    return ApproxTokenCounter()


def sentences(n: int, tag: str) -> str:
    """n distinct, realistic-looking sentences (distinct so dedupe never hides a bug)."""
    return " ".join(
        f"The {tag} variant {i} reduces memory by {i + 3}.5 percent on benchmark {i}, as reported by Smith et al. in Fig. {i}."
        for i in range(n)
    )


def _para(page, y, text, size=10, w=480, h=150, font="helv"):
    rc = page.insert_textbox(pymupdf.Rect(57, y, 57 + w, y + h), text, fontsize=size, fontname=font)
    assert rc >= 0, f"textbox overflow at y={y}"


def _furniture(page, n):
    page.insert_text((57, 35), "Journal of Testing 2026", fontsize=8)
    page.insert_text((290, 820), str(n), fontsize=8)


@pytest.fixture(scope="session")
def paper_pdf(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdfs") / "paper.pdf"
    d = pymupdf.open()
    # page 1
    p = d.new_page(); _furniture(p, 1)
    p.insert_text((57, 90), "Sparse Attention for Long Documents", fontsize=22, fontname="hebo")
    p.insert_text((57, 130), "Abstract", fontsize=13, fontname="hebo")
    _para(p, 140, sentences(4, "abstract"), h=100)
    p.insert_text((57, 270), "1 Introduction", fontsize=13, fontname="hebo")
    _para(p, 280, sentences(5, "intro"), h=130)
    # paragraph that deliberately runs onto the next page with no terminal punctuation
    _para(p, 420, "Efficient attention matters because quadratic cost is crucial for", h=40)
    # page 2
    p = d.new_page(); _furniture(p, 2)
    _para(p, 60, "long documents. Later work addresses this limit with learned patterns.", h=40)
    p.insert_text((57, 130), "2 Method", fontsize=13, fontname="hebo")
    p.insert_text((57, 160), "2.1 Sparse Pattern", fontsize=11, fontname="hebo")
    _para(p, 170, sentences(6, "method"), h=160)
    # page 3
    p = d.new_page(); _furniture(p, 3)
    p.insert_text((57, 100), "3 Results", fontsize=13, fontname="hebo")
    _para(p, 110, sentences(3, "results"), h=90)
    rows = ["Model   BLEU   Params", "Base    27.3   65M", "Big     28.4   213M", "Sparse  28.9   120M"]
    for i, r in enumerate(rows):
        p.insert_text((80, 230 + i * 12), r, fontsize=9, fontname="cour")
    _para(p, 300, "Table 1: Translation quality on WMT14 English-German.", h=30)
    # page 4
    p = d.new_page(); _furniture(p, 4)
    p.insert_text((57, 100), "References", fontsize=13, fontname="hebo")
    _para(p, 110, "[1] Vaswani, A. et al. Attention is all you need. NeurIPS 2017. [2] Child, R. Generating long sequences with sparse transformers. 2019.", h=80)
    d.save(path)
    return path


@pytest.fixture(scope="session")
def two_column_pdf(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdfs") / "twocol.pdf"
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((57, 70), "Two Column Study", fontsize=22, fontname="hebo")
    for i in range(3):
        y = 110 + i * 200
        for col, x in (("LEFT", 57), ("RIGHT", 315)):
            rc = p.insert_textbox(pymupdf.Rect(x, y, x + 230, y + 180),
                                  f"{col}-{i}. " + sentences(3, col.lower() + str(i)), fontsize=9, fontname="helv")
            assert rc >= 0
    d.save(path)
    return path

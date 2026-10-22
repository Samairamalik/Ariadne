"""Regression tests for failures found by running real papers (BERT/arXiv, IEEE Transactions) through the parser.

Synthetic PDFs reproduce each failure mode in isolation; the last section runs the real papers when present."""
import os
from collections import Counter
from pathlib import Path

import pymupdf
import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.pdf_parser import _join_lines, parse_pdf
from conftest import sentences


def _save(d, tmp_path, name) -> Path:
    p = tmp_path / name
    d.save(p)
    return p


def _tb(page, y, text, size=10, x=57, w=480, h=120, font="tiro"):
    assert page.insert_textbox(pymupdf.Rect(x, y, x + w, y + h), text, fontsize=size, fontname=font) >= 0


def _kinds(doc):
    return [(b.kind, b.text) for b in doc.blocks]


# ── hyphenation ─────────────────────────────────────────────────────

def test_hyphen_kept_for_real_compounds_dropped_for_line_breaks():
    vocab = Counter({"task-specific": 3, "transformer": 5})
    assert _join_lines(["a task-", "specific head"], vocab) == "a task-specific head"
    assert _join_lines(["the transfor-", "mer model"], vocab) == "the transformer model"
    assert _join_lines(["an unseen-", "word here"], vocab) == "an unseenword here"  # default: typesetter hyphen


# ── page furniture ──────────────────────────────────────────────────

def test_rotated_margin_text_is_ignored(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((20, 600), "arXiv:1810.04805v2 [cs.CL] 24 May 2019", fontsize=20, rotate=90)
    _tb(p, 100, sentences(4, "body"), font="helv", h=150)
    doc = parse_pdf(_save(d, tmp_path, "rot.pdf"))
    assert not any("arXiv" in b.text for b in doc.blocks)
    assert not any(b.kind == "heading" for b in doc.blocks)


def test_licence_banner_dropped_even_on_a_single_page(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    _tb(p, 100, sentences(4, "body"), font="helv", h=150)
    p.insert_text((60, 780), "1558-0016 © 2023 IEEE. Personal use is permitted, but republication requires permission.", fontsize=8)
    doc = parse_pdf(_save(d, tmp_path, "banner.pdf"))
    assert not any("Personal use" in b.text for b in doc.blocks)


def test_centred_label_straddling_the_gutter_is_not_duplicated(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((290, 60), "TABLE II", fontsize=8, fontname="tiro")  # centred over both columns
    for i, y in enumerate((100, 300, 500)):
        _tb(p, y, f"LEFT{i}. " + sentences(2, f"l{i}"), x=57, w=230, h=150, font="helv", size=9)
        _tb(p, y, f"RIGHT{i}. " + sentences(2, f"r{i}"), x=315, w=230, h=150, font="helv", size=9)
    doc = parse_pdf(_save(d, tmp_path, "gutter.pdf"))
    assert sum(1 for b in doc.blocks if "TABLE II" in b.text) == 1


# ── titles and headings ─────────────────────────────────────────────

def test_multiline_title_merged_and_not_a_heading(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((80, 80), "Scene-Adaptive Real-Time Fast Dehazing and", fontsize=22, fontname="tibo")
    p.insert_text((120, 108), "Detection in Driving Environment", fontsize=22, fontname="tibo")
    _tb(p, 160, sentences(5, "body"), h=150)
    doc = parse_pdf(_save(d, tmp_path, "title.pdf"))
    assert doc.title == "Scene-Adaptive Real-Time Fast Dehazing and Detection in Driving Environment"
    assert not any(b.kind == "heading" for b in doc.blocks)


def test_ieee_roman_numeral_and_lettered_italic_headings_with_runin_abstract(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((80, 80), "A Study of Fog", fontsize=22, fontname="tibo")
    # (PyMuPDF's built-in fonts render an em dash as '?', so use the colon variant of the run-in heading.)
    _tb(p, 120, "Abstract: " + sentences(3, "abs"), size=9, font="tibi", h=90)
    p.insert_text((57, 240), "I. INTRODUCTION", fontsize=10, fontname="tiro")
    _tb(p, 250, sentences(4, "intro"), h=100)
    p.insert_text((57, 370), "A. Atmospheric Light", fontsize=10, fontname="tiit")
    _tb(p, 380, sentences(4, "sub"), h=100)
    p.insert_text((57, 500), "II. RELATED WORKS", fontsize=10, fontname="tiro")
    _tb(p, 510, sentences(3, "rel"), h=80)
    doc = parse_pdf(_save(d, tmp_path, "ieee.pdf"))
    heads = [(b.text, b.level) for b in doc.blocks if b.kind == "heading"]
    assert heads == [("Abstract", 1), ("I. INTRODUCTION", 1), ("A. Atmospheric Light", 2), ("II. RELATED WORKS", 1)]
    chunks, _ = chunk_document(doc, __import__("app.ingestion.tokens", fromlist=["x"]).ApproxTokenCounter())
    assert "I. INTRODUCTION > A. Atmospheric Light" in {c.section_path for c in chunks}


def test_equation_folded_into_preceding_paragraph_not_a_table(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((57, 80), "1 Method", fontsize=12, fontname="tibo")
    _tb(p, 90, "The scattering model used throughout this work is given by the following expression:", h=40)
    p.insert_text((200, 150), "I(x) = J(x)t(x) + A(1 - t(x))        (1)", fontsize=10, fontname="tiro")
    _tb(p, 165, "where I(x) is the observed frame and J(x) is the clean frame. " + sentences(3, "eq"), h=100)
    doc = parse_pdf(_save(d, tmp_path, "eq.pdf"))
    assert not any(b.kind in ("table", "equation") for b in doc.blocks)
    paras = [b.text for b in doc.blocks if b.kind == "paragraph"]
    assert len(paras) == 1 and "A(1 - t(x)) (1) where I(x) is the observed frame" in paras[0]


# ── tables ──────────────────────────────────────────────────────────

def _table_page(p, top, caption, rows, caption_gap=14):
    for i, cells in enumerate(rows):
        for x, txt in zip((57, 220, 320), cells):
            p.insert_text((x, top + i * 12), txt, fontsize=9, fontname="tiro")
    p.insert_text((57, top + len(rows) * 12 + caption_gap), caption, fontsize=9, fontname="tiro")


def test_table_rows_rebuilt_and_caption_attached(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    _tb(p, 60, sentences(4, "lead"), h=100)
    _table_page(p, 200, "Table 3: Results on the development set.",
                [("System", "MNLI", "QQP"), ("BERT-base", "84.6", "71.2"), ("BERT-large", "86.7", "72.1"), ("GPT", "82.1", "70.3")])
    _tb(p, 300, sentences(4, "after"), h=100)
    doc = parse_pdf(_save(d, tmp_path, "table.pdf"))
    tables = [b for b in doc.blocks if b.kind == "table"]
    assert len(tables) == 1
    lines = tables[0].text.split("\n")
    assert lines[0].startswith("Table 3: Results")
    assert "BERT-large | 86.7 | 72.1" in lines  # a row survives as a row, not one cell per line
    assert lines[1] == "System | MNLI | QQP"  # header row is part of the table, not a stray paragraph


def test_two_adjacent_captioned_tables_stay_separate(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    _table_page(p, 100, "Table 1: First.", [("A", "1.1", "2.2"), ("B", "3.3", "4.4")])
    _table_page(p, 160, "Table 2: Second.", [("C", "5.5", "6.6"), ("D", "7.7", "8.8")])
    _tb(p, 400, sentences(4, "filler"), h=120)  # a page with almost no text would be (correctly) treated as scanned
    doc = parse_pdf(_save(d, tmp_path, "two.pdf"))
    tables = [b.text for b in doc.blocks if b.kind == "table"]
    assert len(tables) == 2 and tables[0].startswith("Table 1:") and tables[1].startswith("Table 2:")
    assert "5.5" not in tables[0] and "1.1" not in tables[1]


def test_caption_without_text_layer_body_produces_a_warning(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((57, 100), "TABLE II", fontsize=8, fontname="tiro")
    p.insert_text((57, 112), "TEST RESULTS ON THE DEVELOPMENT SET", fontsize=8, fontname="tiro")
    _tb(p, 400, sentences(5, "prose"), h=150)
    doc = parse_pdf(_save(d, tmp_path, "imgtable.pdf"))
    assert any("TABLE II" in w and "no text-layer content" in w for w in doc.warnings)
    assert any(b.kind == "caption" and "TEST RESULTS" in b.text for b in doc.blocks)  # caption lines absorbed


def test_figure_label_text_is_discarded_not_indexed(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    _tb(p, 60, sentences(4, "lead"), h=100)
    for i, lab in enumerate(["T1", "T2", "T3", "E1", "E2", "E3", "Tok 1", "Tok 2"]):
        p.insert_text((100 + (i % 4) * 60, 300 + (i // 4) * 14), lab, fontsize=7, fontname="helv")
    p.insert_text((57, 360), "Figure 1: Overview of the architecture.", fontsize=9, fontname="tiro")
    doc = parse_pdf(_save(d, tmp_path, "fig.pdf"))
    assert not any("Tok 2" in b.text for b in doc.blocks)


# ── footnotes & continuations ───────────────────────────────────────

def test_footnote_does_not_break_a_paragraph_continued_in_the_next_column(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    _tb(p, 80, "Complete first paragraph in the left column. " + sentences(2, "a"), x=57, w=230, h=120, font="tiro", size=10)
    _tb(p, 210, "The unfinished paragraph says that the results depend strongly on the", x=57, w=230, h=60, font="tiro", size=10)
    p.insert_textbox(pymupdf.Rect(57, 640, 287, 700),
                     "1 This footnote explains in some detail how the thresholds were selected for these experiments.",
                     fontsize=8, fontname="tiro")
    _tb(p, 80, "chosen threshold value. Then the right column goes on with another full sentence.", x=315, w=230, h=80, size=10)
    _tb(p, 200, "Final paragraph of the right column. " + sentences(2, "z"), x=315, w=230, h=120, size=10)
    doc = parse_pdf(_save(d, tmp_path, "fn.pdf"))
    merged = [b for b in doc.blocks if b.kind == "paragraph" and "depend strongly on the chosen threshold value" in b.text]
    assert len(merged) == 1
    assert any(b.kind == "footnote" and "thresholds were selected" in b.text for b in doc.blocks)


# ── run-in paragraph titles, captions, floats ───────────────────────

def test_runin_bold_paragraph_title_becomes_a_subheading(tmp_path):
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((57, 80), "3 Model", fontsize=12, fontname="tibo")
    body = ("The pre-training procedure largely follows the existing literature on language model pre-training and "
            "uses a large unlabeled corpus of books and encyclopedia articles for the first phase of training.")
    rc = p.insert_htmlbox(pymupdf.Rect(57, 90, 540, 200),
                          f"<p style='font-family:serif;font-size:10pt'><b>Pre-training data</b> {body}</p>")
    assert rc[0] >= 0
    _tb(p, 220, sentences(5, "more"), h=130)
    doc = parse_pdf(_save(d, tmp_path, "runin.pdf"))
    heads = [(b.text, b.level) for b in doc.blocks if b.kind == "heading"]
    assert ("Pre-training data", -1) in heads
    para = next(b for b in doc.blocks if b.kind == "paragraph" and "largely follows" in b.text)
    assert not para.text.startswith("Pre-training data")  # the title was split off, not duplicated


def test_runin_titles_are_siblings_under_the_real_heading(counter):
    from app.ingestion.models import Block, ParsedDocument

    blocks = [Block("heading", "3 BERT", 1, level=1),
              Block("heading", "Task #1: Masked LM", 1, level=-1), Block("paragraph", sentences(6, "mlm"), 1),
              Block("heading", "Task #2: Next Sentence Prediction", 1, level=-1), Block("paragraph", sentences(6, "nsp"), 1),
              Block("heading", "3.1 Fine-tuning", 1, level=2), Block("paragraph", sentences(6, "ft"), 1)]
    chunks, _ = chunk_document(ParsedDocument("T", None, 1, blocks), counter)
    assert {c.section_path for c in chunks} == {
        "3 BERT > Task #1: Masked LM", "3 BERT > Task #2: Next Sentence Prediction", "3 BERT > 3.1 Fine-tuning"}


def test_figure_caption_is_its_own_chunk_not_glued_to_prose(counter):
    from app.ingestion.models import Block, ParsedDocument

    blocks = [Block("heading", "1 Intro", 1, level=1), Block("paragraph", sentences(6, "before"), 1),
              Block("caption", "Figure 1: Overall pre-training and fine-tuning procedures for the model.", 1),
              Block("paragraph", sentences(6, "after"), 1)]
    chunks, _ = chunk_document(ParsedDocument("T", None, 1, blocks), counter)
    cap = [c for c in chunks if c.kind == "caption"]
    assert len(cap) == 1 and cap[0].text.startswith("Figure 1:")
    assert not any("Figure 1" in c.text for c in chunks if c.kind == "prose")


def test_floats_do_not_stop_a_paragraph_from_being_rejoined():
    from app.ingestion.models import Block
    from app.ingestion.pdf_parser import _merge_continuations

    blocks = [Block("paragraph", "The results depend strongly on the", 3),
              Block("caption", "Figure 2: A float placed mid-paragraph.", 3),
              Block("table", "Table 1: x\nA | 1.1", 3),
              Block("paragraph", "chosen threshold. The next sentence follows.", 4)]
    out = _merge_continuations(blocks)
    assert [b.kind for b in out] == ["paragraph", "caption", "table"]
    assert out[0].text == "The results depend strongly on the chosen threshold. The next sentence follows."


# ── real papers (skipped when the files aren't on this machine) ─────

BERT = Path(os.environ.get("ARIADNE_TEST_BERT", Path.home() / "Downloads" / "1810.04805v2.pdf"))
IEEE = Path(os.environ.get("ARIADNE_TEST_IEEE", Path.home() / "Desktop" / "Dehazing Papers" /
                           "Scene-Adaptive_Real-Time_Fast_Dehazing_and_Detection_in_Driving_Environment.pdf"))


@pytest.fixture
def counter():  # real papers are checked with the real embedder tokenizer, not the 1.3x-inflated approximation
    from app.ingestion.tokens import get_token_counter
    return get_token_counter()


@pytest.mark.skipif(not BERT.exists(), reason="BERT paper not on this machine")
def test_real_bert_paper(counter):
    doc = parse_pdf(BERT)
    assert doc.title == "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    heads = [(b.text, b.level) for b in doc.blocks if b.kind == "heading"]
    assert ("2.1 Unsupervised Feature-based Approaches", 2) in heads
    assert ("A.1 Illustration of the Pre-training Tasks", 2) in heads and ("A Additional Details for BERT", 1) in heads
    assert not any("arXiv:1810.04805" in b.text for b in doc.blocks)  # rotated sidebar (refs may cite other arXiv ids)
    assert sum(b.in_references for b in doc.blocks) > 50
    chunks, stats = chunk_document(doc, counter)
    assert stats.over_limit == 0
    t1 = next(c for c in chunks if c.kind == "table" and c.text.startswith("Table 1: GLUE"))
    assert "BERTLARGE | 86.7/85.9 | 72.1 | 92.7 | 94.9 | 60.5 | 86.5 | 89.3 | 70.1 | 82.1" in t1.text
    t3 = next(c for c in chunks if c.kind == "table" and c.text.startswith("Table 3: SQuAD 2.0"))
    assert "Table 2" not in t3.text and "BERTLARGE (Single) | 78.7 81.9 80.0 83.1" in t3.text
    assert not any("Vaswani, Noam Shazeer" in c.text for c in chunks)  # reference list excluded
    assert any(c.kind == "footnote" and "GLUE data set distribution" in c.text for c in chunks)


@pytest.mark.skipif(not IEEE.exists(), reason="IEEE paper not on this machine")
def test_real_ieee_paper(counter):
    doc = parse_pdf(IEEE)
    assert doc.title == "Scene-Adaptive Real-Time Fast Dehazing and Detection in Driving Environment"
    heads = {b.text for b in doc.blocks if b.kind == "heading"}
    assert {"I. INTRODUCTION", "II. RELATED WORKS", "V. CONCLUSION", "A. Efficiency Analysis", "REFERENCES"} <= heads
    # tables II-VI are vector drawings with no text layer: must be reported, never silently dropped
    assert sum("no text-layer content" in w for w in doc.warnings) >= 5
    chunks, _ = chunk_document(doc, counter)
    assert not any("Light Perception" in c.section_path and "Zaghloul" in c.text for c in chunks)
    assert not any("Authorized licensed use" in c.text for c in chunks)
    scatter = next(c for c in chunks if "atmospheric scattering model is" in c.text)
    assert "I (x) = J(x)t(x)" in scatter.text and "where I (x) represents" in scatter.text  # equation inline

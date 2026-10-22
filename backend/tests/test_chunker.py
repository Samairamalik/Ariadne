import re

from app.ingestion.chunker import chunk_document
from app.ingestion.models import Block, ParsedDocument
from app.ingestion.sentences import split_sentences
from conftest import sentences

KW = dict(target=300, max_tokens=400, min_tokens=100, overlap_sentences=1, embed_max_tokens=512)


def doc(*blocks, title="Test Paper"):
    return ParsedDocument(title, None, 3, list(blocks))


def H(text, level=1, page=1):
    return Block("heading", text, page, level=level)


def P(text, page=1):
    return Block("paragraph", text, page)


def test_never_exceeds_embedder_limit(counter):
    d = doc(H("1 Intro"), *[P(sentences(12, f"p{i}")) for i in range(8)])
    chunks, stats = chunk_document(d, counter, **KW)
    assert chunks and stats.over_limit == 0
    assert all(c.token_count <= 512 for c in chunks)


def test_chunks_never_cross_sections(counter):
    d = doc(H("1 A"), P(sentences(10, "alpha")), H("2 B"), P(sentences(10, "beta")))
    chunks, _ = chunk_document(d, counter, **KW)
    for c in chunks:
        assert not ("alpha" in c.text and "beta" in c.text)
        assert c.section_path in ("1 A", "2 B")


def test_no_text_is_lost(counter):
    paras = [sentences(9, f"cov{i}") for i in range(6)]
    d = doc(H("1 Intro"), *[P(p) for p in paras])
    chunks, _ = chunk_document(d, counter, **KW)
    joined = " ".join(c.text for c in chunks)
    for p in paras:
        for s in split_sentences(p):
            assert s in joined, f"lost sentence: {s}"


def test_chunks_start_and_end_on_sentence_boundaries(counter):
    d = doc(H("1 Intro"), *[P(sentences(10, f"b{i}")) for i in range(5)])
    chunks, _ = chunk_document(d, counter, **KW)
    for c in chunks:
        assert c.text[0].isupper()
        assert c.text.rstrip().endswith(".")


def test_overlap_is_one_whole_sentence_within_section_only(counter):
    d = doc(H("1 A"), P(sentences(30, "ov")), H("2 B"), P(sentences(30, "other")))
    chunks, _ = chunk_document(d, counter, **KW)
    a = [c for c in chunks if c.section_path == "1 A"]
    b = [c for c in chunks if c.section_path == "2 B"]
    assert len(a) >= 2
    assert a[0].overlap_sentences == 0 and all(c.overlap_sentences == 1 for c in a[1:])
    assert b[0].overlap_sentences == 0  # fresh topic, no bleed from section A
    first_tail = split_sentences(a[0].text)[-1]
    assert a[1].text.startswith(first_tail)


def test_table_is_atomic_and_not_merged_with_prose(counter):
    table = "Model BLEU\nBase 27.3\nBig 28.4\nSparse 28.9\nXL 29.5"
    d = doc(H("3 Results"), P(sentences(4, "res")), Block("table", table, 2), P(sentences(4, "after")))
    chunks, _ = chunk_document(d, counter, **KW)
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1 and tables[0].text == table
    assert all("Base 27.3" not in c.text for c in chunks if c.kind == "prose")


def test_references_excluded(counter):
    ref = Block("paragraph", "Vaswani, A. et al. Attention is all you need. NeurIPS 2017. " * 3, 4, in_references=True)
    d = doc(H("1 Intro"), P(sentences(5, "body")), Block("heading", "References", 4, level=1, in_references=True), ref)
    chunks, stats = chunk_document(d, counter, **KW)
    assert stats.skipped_reference_blocks == 2
    assert all("Vaswani" not in c.text for c in chunks)


def test_heading_path_and_contextual_header(counter):
    d = doc(H("2 Method", 1), H("2.1 Sparse Pattern", 2), P(sentences(5, "sp")))
    chunks, _ = chunk_document(d, counter, **KW)
    assert chunks[0].section_path == "2 Method > 2.1 Sparse Pattern"
    assert chunks[0].embed_text.startswith("Test Paper > 2 Method > 2.1 Sparse Pattern\n\n")
    assert not chunks[0].text.startswith("Test Paper")  # header must not leak into the citeable text


def test_unpunctuated_giant_sentence_is_split_under_limit(counter):
    soup = " ".join(f"token{i}" for i in range(2000))
    d = doc(H("1 A"), P(soup))
    chunks, stats = chunk_document(d, counter, **KW)
    assert stats.long_sentence_splits == 1 and len(chunks) > 1
    assert all(c.token_count <= 512 for c in chunks)


def test_small_tail_folds_into_previous_chunk(counter):
    d = doc(H("1 A"), P(sentences(14, "tail")))
    chunks, _ = chunk_document(d, counter, **KW)
    # no tiny orphan chunk at the end of the section
    assert all(c.token_count >= 60 for c in chunks)


def test_exact_duplicate_chunks_dropped(counter):
    # Boilerplate repeated verbatim as two chunks (e.g. a disclaimer on every section) is indexed once.
    body = sentences(14, "dup")  # long enough that each section produces identical multi-chunk output
    d = doc(H("1 A"), P(body), H("2 B"), P(body))
    chunks, stats = chunk_document(d, counter, **KW)
    single, _ = chunk_document(doc(H("1 A"), P(body)), counter, **KW)
    assert stats.dropped_duplicates == len(single)
    assert len(chunks) == len(single)


def test_junk_dropped(counter):
    d = doc(H("1 A"), P("12 34 56 78 90 11 22 33 44 55 66 77 88 99 00 12 34 56 78 90"), P(sentences(5, "real")))
    chunks, stats = chunk_document(d, counter, **KW)
    assert stats.dropped_junk >= 1
    assert all("12 34 56" not in c.text for c in chunks)


def test_page_range_tracked(counter):
    d = doc(H("1 A"), P(sentences(8, "pg"), page=2), P(sentences(8, "pg2"), page=3))
    chunks, _ = chunk_document(d, counter, **KW)
    assert chunks[0].page_start == 2 and max(c.page_end for c in chunks) == 3
    assert re.match(r"\d+", str(chunks[0].page_start))

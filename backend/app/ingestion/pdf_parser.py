"""Layout-aware PDF -> typed blocks.

What this fixes versus `page.get_text()` (each item was found by running real papers through it):
  * running headers/footers/page numbers/licence banners are removed; rotated margin text (arXiv sidebar) is ignored
  * two-column pages are read column-by-column, not zig-zagged across the gutter
  * hyphenation is repaired using the document's own vocabulary ("transfor-mer" -> transformer, but
    "task-specific" keeps its hyphen); ligatures are normalised
  * headings: numbered ("3.2"), lettered appendices ("A.1"), IEEE roman numerals ("II. RELATED WORKS") with lettered
    italic subsections ("A. Method"), run-in "Abstract—", multi-line headings, drop-cap clean-up
  * multi-line titles are merged into one title
  * equations are folded into the paragraph that introduces them instead of becoming fake tables/chunks
  * tables are rebuilt row-by-row from word coordinates (PyMuPDF flattens them to one cell per line), merged with
    their caption, and figure-label text is discarded; tables that exist only as vector graphics are reported
  * footnotes are tagged so they don't split a paragraph that continues in the next column or page
  * paragraphs split across columns/pages are re-joined
  * the References section (and anything after it, e.g. author bios) is flagged so it can be excluded
Scanned PDFs (no text layer) are detected and reported; OCR is not implemented.
"""
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.ingestion.models import Block, ParsedDocument

_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "­": "", " ": " "})
_NUMBERED_HEADING = re.compile(r"^((?:\d+|[A-Z])(?:\.\d+)*)\.?\s+\S")
_ROMAN_HEADING = re.compile(r"^[IVX]{1,4}\.\s+\S")
_LETTER_SUB = re.compile(r"^[A-Z]\.\s+[A-Z]")
_RUNIN = re.compile(r"^(abstract|index terms|keywords?)\s*[—–:.\-]+\s*", re.I)
_KNOWN_SECTIONS = re.compile(
    r"^(abstract|introduction|background|related works?|method(s|ology)?|approach|model|experiments?|results?|"
    r"discussion|conclusions?|limitations?|acknowledg(e)?ments?|references|bibliography|appendix.*|"
    r"supplementary.*|future work)$",
    re.I,
)
_REFERENCES_HEADING = re.compile(r"^(?:\d+\.?\s+|[IVX]+\.\s+)?(references|bibliography|works cited)$", re.I)
_CAPTION_NUM = re.compile(r"^(figure|fig\.|table|algorithm|listing)\s+\d+\s*[.:]", re.I)
_CAPTION_ROMAN = re.compile(r"^TABLE\s+[IVXLC]+\b")  # IEEE style, literal upper case
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d{1,5}(?:\s*(?:of|/)\s*\d{1,4})?$", re.I)
_BOILERPLATE = re.compile(
    r"©|\(c\)\s*\d{4}|all rights reserved|personal use is permitted|authorized licensed use|downloaded on|"
    r"restrictions apply|republication/redistribution|digital object identifier|^arxiv:\d", re.I)
_MATH_CHARS = set("=∈→≥≤∑∏√∫|−<>±×÷≈∞∂∇")
_FOOTNOTE_MARK = re.compile(r"^[\d*†‡§]+\s*\S")
_SENT_END = re.compile(r"[.!?][\"'”’)\]]*\d*$")
_DECIMAL = re.compile(r"\d+\.\d+")


@dataclass
class _RawBlock:
    page: int
    page_h: float
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    size: float
    bold: float  # fraction of chars in bold
    italic: float
    mono: float  # fraction of chars in a monospace font
    n_lines: int
    avg_line_len: float
    lines_raw: list[str]  # original line breaks, kept for hyphenation repair and code
    lead_bold: str = ""  # bold run at the very start of the block ("Pre-training data" in a run-in paragraph title)


def _is_bold(span: dict) -> bool:
    f = span["font"].lower()
    return bool(span["flags"] & 16) or "bold" in f or f.endswith("-bd") or "medi" in f


def _is_italic(span: dict) -> bool:
    f = span["font"].lower()
    return bool(span["flags"] & 2) or "italic" in f or "oblique" in f


def _is_mono(span: dict) -> bool:
    f = span["font"].lower()
    return bool(span["flags"] & 8) or any(k in f for k in ("courier", "mono", "consolas", "cmtt"))


# ── hyphenation ─────────────────────────────────────────────────────

_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def _build_vocab(raws: list[_RawBlock]) -> Counter:
    vocab: Counter = Counter()
    for r in raws:
        for line in r.lines_raw:
            toks = _WORD.findall(line)
            if line.rstrip().endswith("-") and toks:
                toks = toks[:-1]  # trailing fragment is ambiguous by definition
            vocab.update(t.lower() for t in toks)
    return vocab


_BULLET = re.compile(r"^\s*(?:[•◦▪‣▫·●○■□]\s*|[-–—]\s+|\d{1,2}[.)]\s+)")


def _block_text(lines: list[str], vocab: Counter | None = None) -> str:
    """Join a block's lines. A block that is a bullet/numbered list keeps one item per line ("- item"), so list
    items don't fuse into one run-on sentence; a lone leading bullet glyph is simply stripped."""
    n_bullets = sum(bool(_BULLET.match(l)) for l in lines)
    if n_bullets == 0:
        return _join_lines(lines, vocab)
    if n_bullets == 1 and _BULLET.match(lines[0]):
        return _join_lines([_BULLET.sub("", lines[0], count=1)] + lines[1:], vocab)
    items: list[list[str]] = []
    for ln in lines:
        if _BULLET.match(ln) or not items:
            items.append([_BULLET.sub("", ln, count=1)])
        else:
            items[-1].append(ln)
    return "\n".join("- " + _join_lines(it, vocab) for it in items if any(s.strip() for s in it))


def _join_lines(lines: list[str], vocab: Counter | None = None) -> str:
    """Join physical lines. A line ending in 'xx-' is a real compound ("task-specific") if that spelling occurs
    elsewhere in the document, a line-break hyphen ("transfor-mer") if the closed spelling occurs, and by default the
    latter (typesetters hyphenate far more often than authors coin compounds)."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
            continue
        m = re.search(r"([A-Za-z]{2,})-$", out)
        nxt = re.match(r"([a-z]+)", ln)
        if m and nxt:
            left, right = m.group(1).lower(), nxt.group(1)
            hyph, closed = f"{left}-{right}", left + right
            keep_hyphen = vocab is not None and vocab[hyph] > 0 and vocab[closed] == 0
            out = (out if keep_hyphen else out[:-1]) + ln
        else:
            out += " " + ln
    return re.sub(r"\s+", " ", out).translate(_LIGATURES).strip()


def _extract_raw_blocks(page: pymupdf.Page, page_no: int) -> list[_RawBlock]:
    raws: list[_RawBlock] = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        line_texts, sizes, bold, ital, mono, total = [], Counter(), 0, 0, 0, 0
        for line in b["lines"]:
            if abs(line["dir"][1]) > 0.1:  # rotated text: arXiv sidebar, axis labels
                continue
            parts = []
            for sp in line["spans"]:
                t = sp["text"]
                n = len(t.strip())
                if n:
                    sizes[round(sp["size"], 1)] += n
                    total += n
                    bold += n if _is_bold(sp) else 0
                    ital += n if _is_italic(sp) else 0
                    mono += n if _is_mono(sp) else 0
                parts.append(t)
            line_texts.append("".join(parts))
        lines_raw = [t.strip() for t in line_texts if t.strip()]
        if total == 0 or not lines_raw:
            continue
        x0, y0, x1, y1 = b["bbox"]
        raws.append(_RawBlock(
            page_no, page.rect.height, x0, y0, x1, y1, _block_text(lines_raw), sizes.most_common(1)[0][0],
            bold / total, ital / total, mono / total, len(lines_raw),
            statistics.mean(len(t) for t in lines_raw), lines_raw, _leading_bold(b),
        ))
    return raws


def _leading_bold(block: dict) -> str:
    """Text of the bold run at the start of a block, if the rest of the first line is not bold. This is how
    'run-in' paragraph titles look: **Pre-training data** The pre-training procedure largely follows..."""
    lead, started = "", False
    for line in block["lines"]:
        if abs(line["dir"][1]) > 0.1:
            return ""
        for sp in line["spans"]:
            if not sp["text"].strip():
                lead += sp["text"]
                continue
            if _is_bold(sp):
                lead += sp["text"]
                started = True
            else:
                return re.sub(r"\s+", " ", lead).translate(_LIGATURES).strip() if started else ""
        break  # only ever look at the first line
    return ""


def _reading_order(raws: list[_RawBlock], page_width: float) -> list[_RawBlock]:
    """Order blocks for single- or two-column pages. Full-width blocks split the page into bands; inside a band
    the left column is read fully before the right column."""
    fallback = sorted(raws, key=lambda r: (r.y0, r.x0))
    narrow = [r for r in raws if (r.x1 - r.x0) < 0.55 * page_width]
    if len(raws) < 4 or len(narrow) < 4:
        return fallback

    # Find vertical gutters: x-ranges almost no narrow block covers. A block straddling a gutter (a centred label
    # such as "TABLE II") counts as full-width, never as a member of either column.
    step = 2.0
    n_bins = int(page_width // step) + 1
    cover = [0] * n_bins
    for r in narrow:
        for b in range(int(r.x0 // step), min(int(r.x1 // step) + 1, n_bins)):
            cover[b] += 1
    thresh = max(1, int(0.05 * len(narrow)))
    occupied = [i for i, c in enumerate(cover) if c > 0]
    lo, hi = occupied[0], occupied[-1]
    gutters, i = [], lo
    while i <= hi:
        if cover[i] <= thresh:
            j = i
            while j <= hi and cover[j] <= thresh:
                j += 1
            if (j - i) * step >= 0.01 * page_width and i > lo and j <= hi:  # IEEE gutters are only ~12pt
                gutters.append((i * step, j * step))
            i = j
        else:
            i += 1
    if not gutters:
        return fallback
    edges = [lo * step] + [x for g in gutters for x in g] + [(hi + 1) * step]
    segments = [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2)]
    midpoints = [(g[0] + g[1]) / 2 for g in gutters]
    cols: list[list[_RawBlock]] = [[] for _ in segments]
    for r in narrow:
        if any(r.x0 < m < r.x1 for m in midpoints):
            continue  # spans a gutter: full-width (e.g. a centred "TABLE II" label)
        cx = (r.x0 + r.x1) / 2
        for k, (a, b) in enumerate(segments):
            if a <= cx <= b:
                cols[k].append(r)
                break
    if any(len(c) < 2 for c in cols) or sum(len(c) for c in cols) < 0.5 * len(raws):
        return fallback

    in_cols = {id(r) for c in cols for r in c}
    wide = sorted((r for r in raws if id(r) not in in_cols), key=lambda r: r.y0)
    ordered: list[_RawBlock] = []
    cursor = -1.0
    for y_limit, w in [(w.y0, w) for w in wide] + [(float("inf"), None)]:
        for col in cols:
            ordered += sorted((r for r in col if cursor <= r.y0 < y_limit), key=lambda r: r.y0)
        if w is not None:
            ordered.append(w)
            cursor = w.y0
    return ordered


def _normalise_margin(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _drop_running_furniture(pages: list[list[_RawBlock]], n_pages: int) -> list[list[_RawBlock]]:
    def in_margin(r: _RawBlock) -> bool:
        return r.y1 < 0.10 * r.page_h or r.y0 > 0.90 * r.page_h

    seen: Counter[str] = Counter()
    for raws in pages:
        seen.update({_normalise_margin(r.text) for r in raws if in_margin(r)})
    kept_pages = []
    for raws in pages:
        kept = []
        for r in raws:
            if in_margin(r):
                if _PAGE_NUMBER.match(r.text.strip()) or _BOILERPLATE.search(r.text):
                    continue
                if n_pages >= 3 and seen[_normalise_margin(r.text)] >= max(3, 0.5 * n_pages):
                    continue
            kept.append(r)
        kept_pages.append(kept)
    return kept_pages


# ── block classification helpers ────────────────────────────────────

def _alpha_ratio(text: str) -> float:
    compact = re.sub(r"\s", "", text)
    return sum(c.isalpha() for c in compact) / max(len(compact), 1)


def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / max(len(letters), 1)


def _looks_tabular(r: _RawBlock) -> bool:
    if r.n_lines < 3:
        return False
    compact = r.text.replace(" ", "")
    digit_ratio = sum(c.isdigit() for c in compact) / max(len(compact), 1)
    return r.avg_line_len < 40 and digit_ratio > 0.25


def _is_equation(r: _RawBlock) -> bool:
    words = r.text.split()
    if r.n_lines > 5 or len(words) > 30:
        return False
    has_pua = any("" <= c <= "" for c in r.text)  # brace/bracket glyph pieces of big operators
    circled = any("①" <= c <= "⑳" for c in r.text)  # ① ② ③ — numbered conditions of a piecewise definition
    has_math = any(c in _MATH_CHARS for c in r.text)
    tagged = bool(re.search(r"\(\d+\)$", r.text.strip()))
    if has_pua and _alpha_ratio(r.text) < 0.2:
        return True
    if circled and has_math:
        return True
    return (has_math or tagged) and _alpha_ratio(r.text) < 0.6 and len(_DECIMAL.findall(r.text)) < 3


def _aligned_table_rows(page: pymupdf.Page, r: _RawBlock) -> str | None:
    """Detect an uncaptioned table (slide tables, key-value grids) by column alignment: >=3 rows that share the same
    number of cells whose x-starts line up (within 8pt) down the block. Justified prose also has wide gaps, but its
    word starts drift randomly, so it fails the alignment test."""
    if r.n_lines < 3:
        return None
    words = page.get_text("words", clip=pymupdf.Rect(r.x0 - 1, r.y0 - 1, r.x1 + 1, r.y1 + 1))
    if len(words) < 6:
        return None
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    h = statistics.median(w[3] - w[1] for w in words)
    rows: list[list[tuple]] = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if rows and abs(yc - statistics.mean((x[1] + x[3]) / 2 for x in rows[-1])) < 0.45 * max(h, 1):
            rows[-1].append(w)
        else:
            rows.append([w])
    parsed = []  # per row: list of (cell_x0, cell_text)
    for row in rows:
        row.sort(key=lambda w: w[0])
        cells, cur, x0, prev_x1 = [], [row[0][4]], row[0][0], row[0][2]
        for w in row[1:]:
            if w[0] - prev_x1 > 1.2 * h:
                cells.append((x0, " ".join(cur)))
                cur, x0 = [], w[0]
            cur.append(w[4])
            prev_x1 = w[2]
        cells.append((x0, " ".join(cur)))
        parsed.append(cells)
    modal, count = Counter(len(c) for c in parsed).most_common(1)[0]
    if modal < 2 or count < 3 or count < 0.7 * len(parsed):
        return None
    full = [c for c in parsed if len(c) == modal]
    for k in range(modal):
        xs = [c[k][0] for c in full]
        if max(xs) - min(xs) > 8:
            return None
    return "\n".join(" | ".join(t for _, t in c) for c in parsed).translate(_LIGATURES)


def _rows_from_words(page: pymupdf.Page, r: _RawBlock) -> str:
    """Rebuild table rows from word coordinates: cluster by baseline, order by x, ' | ' at column-sized gaps."""
    words = page.get_text("words", clip=pymupdf.Rect(r.x0 - 1, r.y0 - 1, r.x1 + 1, r.y1 + 1))
    if not words:
        return "\n".join(r.lines_raw)
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list[tuple]] = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if rows and abs(yc - statistics.mean((x[1] + x[3]) / 2 for x in rows[-1])) < 0.45 * max(w[3] - w[1], 1):
            rows[-1].append(w)
        else:
            rows.append([w])
    gap_limit = 0.9 * statistics.median(w[3] - w[1] for w in words)
    lines = []
    for row in rows:
        row.sort(key=lambda w: w[0])
        cells, cur, prev_x1 = [], [row[0][4]], row[0][2]
        for w in row[1:]:
            if w[0] - prev_x1 > gap_limit:
                cells.append(" ".join(cur))
                cur = []
            cur.append(w[4])
            prev_x1 = w[2]
        cells.append(" ".join(cur))
        lines.append(" | ".join(cells))
    return "\n".join(lines).translate(_LIGATURES)


def _figure_regions(page: pymupdf.Page, cell: float = 18.0) -> list[pymupdf.Rect]:
    """Regions of the page occupied by vector diagrams/plots. Text inside them (node labels, axis ticks, legends)
    is not prose and, left in, poisons both the embedding and the reranker's view of the chunk.

    Drawings are rasterised onto a coarse grid and grouped into connected components. A component is a figure only
    if it has many items AND curves/diagonal strokes: table rules and cell shading are axis-aligned rects/lines, so
    tables are never mistaken for figures."""
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 — a malformed content stream must not fail ingestion
        return []
    W, H = page.rect.width, page.rect.height
    gw, gh = int(W // cell) + 1, int(H // cell) + 1
    items: dict[tuple[int, int], int] = {}
    curvy: dict[tuple[int, int], int] = {}
    for d in drawings:
        r = d["rect"]
        if min(r.width, r.height) < 1.5 or (r.width > 0.9 * W and r.height > 0.5 * H):
            continue  # hairline rule or page-sized background
        is_curvy = any(it[0] == "c" or (it[0] == "l" and abs(it[1].x - it[2].x) > 1 and abs(it[1].y - it[2].y) > 1)
                       for it in d["items"])
        x0, x1 = max(int(r.x0 // cell), 0), min(int(r.x1 // cell), gw - 1)
        y0, y1 = max(int(r.y0 // cell), 0), min(int(r.y1 // cell), gh - 1)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 400:
            continue
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                items[(gx, gy)] = items.get((gx, gy), 0) + 1
                curvy[(gx, gy)] = curvy.get((gx, gy), 0) + int(is_curvy)
    regions, seen = [], set()
    for start in items:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            gx, gy = stack.pop()
            comp.append((gx, gy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (gx + dx, gy + dy)
                    if nb in items and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(comp) >= 8 and sum(items[c] for c in comp) >= 20 and sum(curvy[c] for c in comp) >= 5:
            xs, ys = [c[0] for c in comp], [c[1] for c in comp]
            regions.append(pymupdf.Rect(min(xs) * cell - 2, min(ys) * cell - 2, (max(xs) + 1) * cell + 2, (max(ys) + 1) * cell + 2))
    return regions


def _in_figure(r: _RawBlock, regions: list[pymupdf.Rect]) -> bool:
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    return any(reg.x0 <= cx <= reg.x1 and reg.y0 <= cy <= reg.y1 for reg in regions)


# ── main entry ──────────────────────────────────────────────────────

def parse_pdf(path: str | Path) -> ParsedDocument:
    path = Path(path)
    doc = pymupdf.open(path)
    warnings: list[str] = []

    pages_raw: list[list[_RawBlock]] = []
    total_chars = 0
    for i, page in enumerate(doc, start=1):
        raws = _extract_raw_blocks(page, i)
        total_chars += sum(len(r.text) for r in raws)
        pages_raw.append(_reading_order(raws, page.rect.width))

    if doc.page_count == 0 or total_chars / doc.page_count < 100:
        return ParsedDocument(None, None, doc.page_count, [], needs_ocr=True,
                              warnings=["No text layer found (scanned/image PDF). OCR is not implemented yet."])

    pages = _drop_running_furniture(pages_raw, doc.page_count)
    vocab = _build_vocab([r for raws in pages for r in raws])
    for raws in pages:
        for r in raws:
            if any(ln.endswith("-") for ln in r.lines_raw[:-1]):
                r.text = _block_text(r.lines_raw, vocab)

    size_chars: Counter[float] = Counter()
    for raws in pages:
        for r in raws:
            size_chars[r.size] += len(r.text)
    body = size_chars.most_common(1)[0][0]

    blocks: list[Block] = []
    title_ids = _title_cluster(pages[0] if pages else [], body)
    figure_blocks = 0
    for raws in pages:
        regions = _figure_regions(doc[raws[0].page - 1]) if raws else []
        for r in raws:
            if id(r) in title_ids:
                blocks.append(Block("title", r.text, r.page, font_size=r.size, y0=r.y0, y1=r.y1))
                continue
            is_caption = bool(_CAPTION_NUM.match(r.text) or _CAPTION_ROMAN.match(r.text))
            if regions and not is_caption and len(r.text.split()) <= 20 and _in_figure(r, regions):
                figure_blocks += 1
                continue
            blocks.extend(_classify(r, body, doc[r.page - 1]))
    if figure_blocks:
        warnings.append(f"Discarded {figure_blocks} text block(s) inside vector figures.")

    blocks = _merge_title(blocks)
    blocks = _fix_dropcaps(blocks)
    blocks = _merge_multiline_headings(blocks)
    blocks = _absorb_caption_lines(blocks, body)
    blocks = _fold_equations(blocks)
    blocks = _assemble_tables(blocks, warnings)
    blocks = _merge_footnotes(blocks)
    deck = sum(1 for p in doc if p.rect.width > 1.2 * p.rect.height) >= 0.8 * doc.page_count
    _assign_heading_levels(blocks, body, deck)
    blocks = _merge_continuations(blocks)
    warnings.extend(_image_page_warnings(doc, pages))
    _flag_references(blocks)

    meta = doc.metadata or {}
    title = next((b.text for b in blocks if b.kind == "title"), None)
    if not title:
        mt = (meta.get("title") or "").strip()
        title = mt if len(mt) > 8 and "print to pdf" not in mt.lower() and not mt.lower().endswith((".doc", ".docx", ".tex")) else None
    return ParsedDocument(title, (meta.get("author") or "").strip() or None, doc.page_count, blocks, False, warnings)


def _image_page_warnings(doc: pymupdf.Document, pages: list[list[_RawBlock]]) -> list[str]:
    """Text inside raster images (screenshots, infographics, scanned figures) has no text layer and is not indexed.
    Say so, per page, instead of letting those pages silently look empty in search."""
    out = []
    for i, raws in enumerate(pages):
        page = doc[i]
        big_images = [im for im in page.get_image_info() if (im["bbox"][2] - im["bbox"][0]) * (im["bbox"][3] - im["bbox"][1])
                      > 0.15 * page.rect.width * page.rect.height]
        chars = sum(len(r.text) for r in raws)
        if big_images and chars < 400:
            out.append(f"Page {i + 1}: large image with only {chars} characters of text; text inside the image is not "
                       f"indexed (needs OCR / a vision model).")
        elif not big_images and chars < 40:
            out.append(f"Page {i + 1}: almost no extractable text ({chars} chars).")
    return out


def _title_cluster(page1: list[_RawBlock], body: float) -> set[int]:
    cands = [r for r in page1 if r.n_lines <= 3 and r.y0 < 0.45 * r.page_h and r.size >= 1.3 * body]
    if not cands:
        return set()
    biggest = max(r.size for r in cands)
    ids: set[int] = set()
    prev: _RawBlock | None = None
    for r in page1:
        ok = r in cands and abs(r.size - biggest) <= 0.06 * biggest
        if ok and (prev is None or r.y0 - prev.y1 < 2.0 * r.size):
            ids.add(id(r))
            prev = r
        elif ids and not ok:
            break
    return ids


def _classify(r: _RawBlock, body: float, page: pymupdf.Page) -> list[Block]:
    text = r.text
    if not text or _PAGE_NUMBER.match(text):
        return []
    mk = lambda kind, t=None, **kw: Block(kind, t if t is not None else text, r.page, font_size=r.size, y0=r.y0, y1=r.y1, **kw)  # noqa: E731
    words = text.split()

    m = _RUNIN.match(text)
    if m and r.n_lines >= 1 and (r.bold > 0.5 or len(words) <= 3 or text[m.end():m.end() + 1].isupper()):
        rest = text[m.end():].strip()
        head = m.group(1).strip().title() if m.group(1).lower() != "index terms" else "Index Terms"
        out = [mk("heading", head)]
        if rest:
            out.append(mk("paragraph", rest))
        return out

    if _CAPTION_NUM.match(text) or (_CAPTION_ROMAN.match(text) and len(words) <= 40):
        return [mk("caption")]

    if r.n_lines >= 3 and sum(bool(_BULLET.match(l)) for l in r.lines_raw) < 0.5 * r.n_lines:
        aligned = _aligned_table_rows(page, r)
        if aligned:
            return [mk("table", aligned)]

    numbered = _NUMBERED_HEADING.match(text)
    bigger = r.size >= body * 1.15
    boldish = r.bold > 0.8 and r.size >= body * 0.97
    known = bool(_KNOWN_SECTIONS.match(text.rstrip(". :")))
    short = len(words) <= 14 and r.n_lines <= 2 and not text.endswith((".", ",", ";"))
    roman = bool(_ROMAN_HEADING.match(text)) and len(words) <= 12 and r.n_lines <= 2 and (_upper_ratio(text) > 0.7 or r.bold > 0.8)
    letter_sub = bool(_LETTER_SUB.match(text)) and len(words) <= 20 and r.n_lines <= 3 \
        and (r.italic > 0.8 or r.bold > 0.8) and not text.endswith(".")
    known_small = known and r.n_lines == 1 and len(words) <= 4 and (_upper_ratio(text) > 0.7 or r.bold > 0.5 or r.size >= 0.8 * body)
    if roman or letter_sub or known_small or (
            short and (bigger or (boldish and (numbered or known or len(words) <= 8)) or (known and r.size >= body))):
        return [mk("heading")]

    if _is_equation(r):
        return [mk("equation")]

    if _looks_tabular(r) or (_alpha_ratio(text) < 0.45 and r.n_lines >= 2):
        rows = _rows_from_words(page, r)
        if len(_DECIMAL.findall(rows)) >= 3:
            return [mk("table", rows)]
        # Weak candidate: either a fragment of a captioned table (kept only if it sits next to a 'Table N' caption,
        # see _assemble_tables) or figure labels ("T1 T2 ... Tok N") / axis ticks, which are discarded there.
        return [mk("tablecand", rows)]

    if r.mono > 0.8 and r.n_lines >= 2:
        return [mk("code", "\n".join(r.lines_raw).translate(_LIGATURES))]

    if r.size <= 0.86 * body and r.y0 > 0.6 * r.page_h and (len(words) >= 12 or _FOOTNOTE_MARK.match(text)):
        return [mk("footnote")]

    lead = r.lead_bold
    lead_words = lead.split()
    if (lead and text.startswith(lead) and 2 <= len(lead_words) <= 8 and lead[0].isupper() and not lead.endswith((".", ",", ";"))
            and r.bold < 0.6 and len(text) - len(lead) >= 40):
        # run-in paragraph title: sub-heading (level -1 = "one below the current real heading", see chunker)
        return [mk("heading", lead, level=-1), mk("paragraph", text[len(lead):].strip())]

    blk = mk("paragraph")
    n_numeric = sum(bool(re.search(r"\d", w)) for w in words)
    short_fragment = r.n_lines <= 10 and len(words) <= 45 and (n_numeric >= 2 or len(words) <= 12)
    # Only ever consumed when it sits right against a 'Table N' caption, so a loose density threshold is safe.
    numeric_body = r.n_lines <= 40 and len(words) <= 140 and n_numeric >= 5 and n_numeric / max(len(words), 1) >= 0.15
    if short_fragment or numeric_body:
        blk.rows = _rows_from_words(page, r)  # may be a header/footer row of a table; decided by its caption
    return [blk]


# ── post passes ─────────────────────────────────────────────────────

def _merge_title(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        if b.kind == "title" and out and out[-1].kind == "title":
            out[-1].text += " " + b.text
        else:
            out.append(b)
    return out


def _fix_dropcaps(blocks: list[Block]) -> list[Block]:
    """'I. INTRODUCTION' + a lone drop-cap letter on a second line: the letter belongs to the next paragraph."""
    for i, b in enumerate(blocks):
        m = re.match(r"^(.*\S)\s+([A-Z])$", b.text)
        if b.kind == "heading" and m and _ROMAN_HEADING.match(b.text) and i + 1 < len(blocks) and blocks[i + 1].kind == "paragraph":
            b.text = m.group(1)
            blocks[i + 1].text = m.group(2) + blocks[i + 1].text
    return blocks


def _merge_multiline_headings(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        p = out[-1] if out else None
        starts_new = bool(_NUMBERED_HEADING.match(b.text) or _ROMAN_HEADING.match(b.text) or _LETTER_SUB.match(b.text)
                          or _KNOWN_SECTIONS.match(b.text.rstrip(". :")))
        if (p and p.kind == "heading" and b.kind == "heading" and p.page == b.page and not starts_new
                and p.level != -1 and b.level != -1  # run-in titles are never continuations of a real heading
                and abs(p.font_size - b.font_size) < 0.3 and -0.4 * b.font_size <= b.y0 - p.y1 < 0.9 * b.font_size):
            p.text += " " + b.text
            p.y1 = b.y1
        else:
            out.append(b)
    return out


def _absorb_caption_lines(blocks: list[Block], body: float) -> list[Block]:
    """IEEE 'TABLE II' captions arrive as a label block followed by one block per ALL-CAPS line."""
    out: list[Block] = []
    absorbing: Block | None = None
    for b in blocks:
        if b.kind == "caption" and re.fullmatch(r"TABLE\s+[IVXLC]+", b.text.strip()):
            absorbing = b
            out.append(b)
        elif absorbing and b.kind == "paragraph" and b.page == absorbing.page and b.font_size <= 0.9 * body \
                and _upper_ratio(b.text) > 0.7:
            absorbing.text += " " + b.text
            absorbing.y1 = b.y1
        else:
            absorbing = None
            out.append(b)
    return out


def _fold_equations(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        if b.kind != "equation":
            out.append(b)
            continue
        prev = next((x for x in reversed(out) if x.kind != "footnote"), None)
        if prev is not None and prev.kind == "paragraph" and prev.page == b.page:
            prev.text += " " + re.sub(r"\s+", " ", b.text)
    return out


def _table_fragment(b: Block, numeric_only: bool) -> bool:
    """Could this block be part of a table body? Table pieces are tables, weak table candidates, or short
    paragraphs (header rows like 'System Dev Test EM F1', footer rows like 'Human (expert) - 85.0')."""
    if b.kind in ("table", "tablecand"):
        return True
    if b.kind != "paragraph" or b.rows is None:
        return False
    return not numeric_only or sum(bool(re.search(r"\d", w)) for w in b.rows.split()) >= 2


def _vertically_adjacent(above: Block, below: Block, max_gap: float = 30) -> bool:
    """`below` starts within max_gap of where `above` ends. Cell blocks sitting side by side in one row overlap
    vertically and still count; a block from the *other column* (starting far below `below`'s end) does not."""
    return above.y0 <= below.y1 + 2 and below.y0 - above.y1 <= max_gap


def _assemble_tables(blocks: list[Block], warnings: list[str]) -> list[Block]:
    """Group each 'Table N' caption with the contiguous table-like blocks next to it. PyMuPDF splits one visual
    table into several blocks (header row, ruled sections, footer rows); the caption is the reliable anchor.
    Captions below the table (ACL style) are tried first, then above (IEEE style)."""
    owner: dict[int, int] = {}  # block index -> index of its caption
    members: dict[int, list[int]] = {}
    for ci, cap in enumerate(blocks):
        if cap.kind != "caption" or not re.match(r"table", cap.text, re.I):
            continue
        chain: list[int] = []
        j, nxt = ci - 1, cap
        while j >= 0 and j not in owner and blocks[j].page == cap.page and _table_fragment(blocks[j], False) \
                and _vertically_adjacent(blocks[j], nxt):
            chain.append(j)
            j, nxt = j - 1, blocks[j]
        chain.reverse()
        if not chain:
            j, prev = ci + 1, cap
            while j < len(blocks) and j not in owner and blocks[j].page == cap.page and _table_fragment(blocks[j], True) \
                    and _vertically_adjacent(prev, blocks[j]):
                chain.append(j)
                j, prev = j + 1, blocks[j]
        if not chain:
            label = re.match(r"(table\s+[0-9IVXLC]+)", cap.text, re.I)
            warnings.append(f"{label.group(1) if label else 'Table'}: caption found but no text-layer content "
                            f"(table is an image/vector drawing); only the caption is indexed.")
            continue
        members[ci] = chain
        owner.update({k: ci for k in chain})

    out: list[Block] = []
    dropped = 0
    emitted_chain: set[int] = set()
    for i, b in enumerate(blocks):
        if i in members:  # the caption itself: its table is emitted at the position of the first body block
            continue
        if i in owner:
            ci = owner[i]
            if ci in emitted_chain:
                continue
            emitted_chain.add(ci)
            chain = members[ci]
            body = "\n".join((blocks[k].rows or blocks[k].text) if blocks[k].kind != "table" else blocks[k].text
                             for k in chain)
            out.append(Block("table", blocks[ci].text + "\n" + body, blocks[ci].page, font_size=blocks[chain[0]].font_size,
                             y0=blocks[chain[0]].y0, y1=blocks[ci].y1))
        elif b.kind == "tablecand":
            dropped += 1
        else:
            out.append(b)
    if dropped:
        warnings.append(f"Discarded {dropped} figure-label/axis text block(s).")
    # uncaptioned strong tables that touch vertically are one table
    merged: list[Block] = []
    for b in out:
        p = merged[-1] if merged else None
        # A block that starts with its own caption is a different table, however close it sits.
        if p and p.kind == "table" and b.kind == "table" and p.page == b.page and 0 <= b.y0 - p.y1 < 24 \
                and not re.match(r"table\b", b.text, re.I):
            p.text += "\n" + b.text
            p.y1 = b.y1
        else:
            merged.append(b)
    return merged


def _merge_footnotes(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        if b.kind == "footnote" and out and out[-1].kind == "footnote" and out[-1].page == b.page:
            out[-1].text += " " + b.text
        else:
            out.append(b)
    return out


def _numbering_level(text: str, seen_roman: bool) -> int | None:
    if _ROMAN_HEADING.match(text):
        return 1
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s", text)
    if m:
        return m.group(1).count(".") + 1
    m = re.match(r"^([A-Z])(\.\d+)+\s", text)
    if m:
        return text.split()[0].rstrip(".").count(".") + 1
    if re.match(r"^[A-Z]\.\s", text):
        return 2 if seen_roman else 1
    if re.match(r"^[A-Z]\s+[A-Z]", text):
        return 1
    return None


def _assign_heading_levels(blocks: list[Block], body: float = 10.0, deck: bool = False) -> None:
    headings = [b for b in blocks if b.kind == "heading" and b.level != -1]  # -1 = run-in title, kept as is
    if deck:
        # Slides: every slide title is the same logical level even though auto-shrink gives them different font
        # sizes (rank-by-size produced titles at levels 1, 2 and 3 and nonsense section paths). Anything smaller
        # inside a slide is a sub-heading.
        for h in headings:
            h.level = 1 if h.font_size >= 1.5 * body else 2
        return
    seen_roman = any(_ROMAN_HEADING.match(h.text) for h in headings)
    unnumbered = [h for h in headings if _numbering_level(h.text, seen_roman) is None]
    sizes = sorted({h.font_size for h in unnumbered}, reverse=True)
    rank = {s: i + 1 for i, s in enumerate(sizes)}
    for h in headings:
        lvl = _numbering_level(h.text, seen_roman)
        if lvl is None and _KNOWN_SECTIONS.match(h.text.rstrip(". :")):
            lvl = 1  # Abstract / References / Conclusion are top-level whatever their font size
        h.level = lvl if lvl is not None else min(rank[h.font_size], 3)


def _merge_continuations(blocks: list[Block]) -> list[Block]:
    """A paragraph that doesn't end a sentence, followed by one starting in lower case, is one paragraph split by a
    column or page break. Footnotes and equations sitting in between don't interrupt it."""
    out: list[Block] = []
    last_para: Block | None = None
    for b in blocks:
        if b.kind == "paragraph":
            if last_para is not None and not _SENT_END.search(last_para.text.rstrip()) and b.text[:1].islower():
                closed = re.search(r"[A-Za-z]{2}-$", last_para.text)
                last_para.text = (last_para.text[:-1] if closed else last_para.text + " ") + b.text
                continue
            last_para = b
        elif b.kind in ("heading", "title"):
            last_para = None  # floats (figures, tables, code) are placed wherever TeX likes; they don't end a paragraph
        out.append(b)
    return out


def _flag_references(blocks: list[Block]) -> None:
    active_level: int | None = None
    for b in blocks:
        if b.kind == "heading" and b.level != -1:
            if _REFERENCES_HEADING.match(b.text.strip()):
                active_level = b.level
                b.in_references = True
                continue
            if active_level is not None and (b.level <= active_level or re.match(r"^(appendix|supplementary)", b.text, re.I)):
                active_level = None
        if active_level is not None:
            b.in_references = True
    # Safety net for reference lists with no recognisable heading: a run of 5+ "[n] ..." entries.
    run = [i for i, b in enumerate(blocks) if b.kind in ("paragraph", "footnote") and re.match(r"^\[\d+\]\s", b.text)]
    if len(run) >= 5:
        for i in run:
            blocks[i].in_references = True


# ── Plain text & Markdown ────────────────────────────────────────────

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def parse_text(path: str | Path, markdown: bool) -> ParsedDocument:
    raw = Path(path).read_text(encoding="utf-8", errors="replace").translate(_LIGATURES)
    blocks: list[Block] = []
    title = None
    in_code, code_lines, para_lines = False, [], []

    def flush_para():
        if para_lines:
            blocks.append(Block("paragraph", re.sub(r"\s+", " ", " ".join(para_lines)).strip(), 1))
            para_lines.clear()

    for line in raw.splitlines():
        if markdown and line.strip().startswith("```"):
            if in_code:
                blocks.append(Block("code", "\n".join(code_lines), 1))
                code_lines.clear()
            else:
                flush_para()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        m = _MD_HEADING.match(line) if markdown else None
        if m:
            flush_para()
            level, text = len(m.group(1)), m.group(2)
            if level == 1 and title is None and not blocks:
                title = text
                blocks.append(Block("title", text, 1))
            else:
                blocks.append(Block("heading", text, 1, level=level - 1 if title else level))
        elif not line.strip():
            flush_para()
        else:
            para_lines.append(line.strip())
    flush_para()
    if code_lines:  # unterminated fence
        blocks.append(Block("code", "\n".join(code_lines), 1))
    _flag_references(blocks)
    return ParsedDocument(title, None, 1, blocks)

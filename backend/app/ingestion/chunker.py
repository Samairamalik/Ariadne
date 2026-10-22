"""Structure-aware, sentence-packed chunking.

Design (and why it differs from the pseudo-code in ariadne-design.md §3):
  * Chunks never cross section boundaries and are built per *section across pages*, not per page, so a
    section that spans a page break is not cut at the break.
  * Packing happens on sentence boundaries; paragraph boundaries are preferred split points.
  * Overlap is whole sentences (default 1), only inside a section — never splicing half a sentence, never
    bleeding one topic into the next.
  * Tables and code are atomic: never merged into prose and never split mid-row.
  * Token budget is measured on what the embedder actually sees (contextual header + text) with the embedder's
    own tokenizer. bge-small silently truncates at 512 tokens; the design doc's 600-token max would lose text.
  * Each chunk is embedded with a contextual header ("Doc title > Section > Subsection") so a paragraph that
    says "the model" still carries which model/section it belongs to. The clean text is stored separately
    for display and citation.
"""
import re
from dataclasses import dataclass, field

from app.ingestion.models import Block, Chunk, ParsedDocument
from app.ingestion.sentences import split_sentences
from app.ingestion.tokens import TokenCounter

_SAFETY = 8  # tokens of slack for tokenizer differences when concatenating pieces


@dataclass
class ChunkStats:
    n_chunks: int = 0
    dropped_junk: int = 0
    dropped_duplicates: int = 0
    skipped_reference_blocks: int = 0
    long_sentence_splits: int = 0
    over_limit: int = 0
    token_counts: list[int] = field(default_factory=list)

    def summary(self) -> dict:
        tc = self.token_counts or [0]
        return {
            "chunks": self.n_chunks, "tokens_min": min(tc), "tokens_max": max(tc),
            "tokens_mean": round(sum(tc) / len(tc), 1), "dropped_junk": self.dropped_junk,
            "dropped_duplicates": self.dropped_duplicates,
            "skipped_reference_blocks": self.skipped_reference_blocks,
            "long_sentence_splits": self.long_sentence_splits, "over_embedder_limit": self.over_limit,
        }


@dataclass
class _Unit:
    text: str
    tokens: int
    page: int
    para_id: int
    kind: str = "prose"  # prose | table | code
    overlap: bool = False
    newline: bool = False  # first sentence of a new line inside a bullet list

    @property
    def atomic(self) -> bool:
        return self.kind != "prose"


@dataclass
class _Section:
    path: str
    blocks: list[Block]


def _split_long(text: str, limit: int, counter: TokenCounter) -> list[str]:
    """Last resort for a 'sentence' longer than the budget (no punctuation, giant equation, OCR soup)."""
    words, pieces, cur = text.split(), [], []
    for w in words:
        cur.append(w)
        if counter.count(" ".join(cur)) > limit:
            if len(cur) > 1:
                cur.pop()
                pieces.append(" ".join(cur))
                cur = [w]
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _split_lines(text: str, limit: int, counter: TokenCounter, repeat_first_line: bool = False) -> list[str]:
    """Split on line boundaries. For tables, every piece after the first repeats the first line (the caption), so
    a lone block of rows is still identifiable as 'Table 1: GLUE results'."""
    lines = text.split("\n")
    head = lines[0][:160] if repeat_first_line and len(lines) > 1 else None
    pieces, cur = [], []
    for line in lines:
        if cur and counter.count("\n".join(cur + [line])) > limit:
            pieces.append("\n".join(cur))
            cur = [head] if head else []
        cur.append(line)
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def _sections(doc: ParsedDocument, stats: ChunkStats) -> list[_Section]:
    sections: list[_Section] = [_Section("", [])]
    stack: list[str] = []  # path of the last *real* heading
    for b in doc.blocks:
        if b.kind == "title":
            continue
        if b.in_references:
            stats.skipped_reference_blocks += 1
            continue
        if b.kind == "heading":
            if b.level == -1:  # run-in paragraph title: one level below the current real heading, replaces siblings
                sections.append(_Section(" > ".join(stack + [b.text.strip()]), []))
                continue
            level = max(b.level, 1)
            stack = stack[: level - 1] + [b.text.strip()]
            sections.append(_Section(" > ".join(stack), []))
        else:
            sections[-1].blocks.append(b)
    return [s for s in sections if s.blocks]


def _fold_tiny_parents(sections: list[_Section], counter: TokenCounter, min_tokens: int) -> list[_Section]:
    """A parent section that is only a sentence or two of lead-in ("In this section we present results on 11
    tasks.") becomes a near-empty chunk that matches generic queries and answers nothing. Prepend it to its first
    child subsection instead; the child keeps its more specific path."""
    out: list[_Section] = []
    for i, sec in enumerate(sections):
        nxt = sections[i + 1] if i + 1 < len(sections) else None
        prose = all(b.kind == "paragraph" for b in sec.blocks)
        if (nxt is not None and sec.path and nxt.path.startswith(sec.path + " > ") and prose
                and counter.count(" ".join(b.text for b in sec.blocks)) < min_tokens):
            nxt.blocks = sec.blocks + nxt.blocks
        else:
            out.append(sec)
    return out


def _units_for(section: _Section, budget: int, counter: TokenCounter, stats: ChunkStats,
               atomic_budget: int | None = None) -> list[_Unit]:
    units: list[_Unit] = []
    atomic_budget = atomic_budget or budget
    for pid, b in enumerate(section.blocks):
        if b.kind in ("table", "code"):
            text, tail = b.text, None
            first, _, rest = text.partition("\n")
            if b.kind == "table" and rest and len(first) > 240 and re.match(r"table\b", first, re.I):
                # A long caption ("Table 1: ... F1 scores are reported for ... We exclude entries that ...") would
                # crowd the rows out of the embedding window. Keep the first sentence with the rows and index the
                # remainder of the caption as its own chunk so nothing is lost.
                sents = split_sentences(first)
                if len(sents) > 1:
                    text, tail = sents[0] + "\n" + rest, " ".join(sents[1:])
            for piece in _split_lines(text, atomic_budget, counter, repeat_first_line=b.kind == "table"):
                units.append(_Unit(piece, counter.count(piece), b.page, pid, kind=b.kind))
            if tail:
                units.append(_Unit(f"{sents[0].split(':')[0]} (continued): {tail}", counter.count(tail) + 6, b.page, pid, kind="caption"))
            continue
        if b.kind in ("footnote", "caption"):
            # Kept apart from prose so they never dilute a paragraph's chunk: a figure caption glued to the end of an
            # unrelated paragraph made that chunk match neither topic well (found with the reranker eval).
            n = counter.count(b.text)
            for piece in ([b.text] if n <= budget else _split_long(b.text, budget, counter)):
                units.append(_Unit(piece, counter.count(piece), b.page, pid, kind=b.kind))
            continue
        # paragraph
        compact = re.sub(r"\s", "", b.text)
        if sum(c.isalpha() for c in compact) / max(len(compact), 1) < 0.4:
            stats.dropped_junk += 1  # stray axis labels, numeric soup, OCR garbage
            continue
        for li, line in enumerate(b.text.split("\n")):  # bullet items stay on their own lines
            for si, sent in enumerate(split_sentences(line)):
                nl = li > 0 and si == 0
                n = counter.count(sent)
                if n > budget:
                    stats.long_sentence_splits += 1
                    for piece in _split_long(sent, budget, counter):
                        units.append(_Unit(piece, counter.count(piece), b.page, pid, newline=nl))
                else:
                    units.append(_Unit(sent, n, b.page, pid, newline=nl))
    return units


def _render(units: list[_Unit]) -> str:
    out = ""
    for i, u in enumerate(units):
        if i == 0:
            out = u.text
        elif u.atomic or units[i - 1].atomic or u.para_id != units[i - 1].para_id:
            out += "\n\n" + u.text
        elif u.newline:
            out += "\n" + u.text
        else:
            out += " " + u.text
    return out


def _pack(units: list[_Unit], target: int, max_tokens: int, min_tokens: int, overlap_k: int) -> list[list[_Unit]]:
    chunks: list[list[_Unit]] = []
    cur: list[_Unit] = []
    # footnote/table/code units are atomic: each is its own chunk, so a footnote never shares a chunk with prose

    def body_tokens(us: list[_Unit]) -> int:
        return sum(u.tokens for u in us)

    def flush():
        nonlocal cur
        fresh = [u for u in cur if not u.overlap]
        if fresh:
            chunks.append(cur)
        carry: list[_Unit] = []
        if overlap_k and fresh and not any(u.atomic for u in cur):
            tail = [u for u in cur if not u.atomic][-overlap_k:]
            if body_tokens(tail) <= 0.25 * target:
                carry = [_Unit(u.text, u.tokens, u.page, u.para_id, u.kind, overlap=True) for u in tail]
        cur = carry

    for u in units:
        if u.atomic:
            flush()
            cur = []  # no overlap around tables/code
            chunks.append([u])
            continue
        if cur:
            cur_tok = body_tokens(cur)
            new_para = u.para_id != cur[-1].para_id
            if cur_tok + u.tokens > max_tokens:
                flush()
            elif cur_tok >= target and (new_para or cur_tok + u.tokens > target * 1.15):
                flush()
            elif new_para and cur_tok >= 0.75 * target and cur_tok + u.tokens > target:
                flush()
        cur.append(u)
    flush()

    # Fold a tiny trailing chunk back into its predecessor when it fits.
    if len(chunks) >= 2 and not chunks[-1][0].atomic and not chunks[-2][-1].atomic:
        last_fresh = [u for u in chunks[-1] if not u.overlap]
        if body_tokens(last_fresh) < min_tokens and body_tokens(chunks[-2]) + body_tokens(last_fresh) <= max_tokens:
            chunks[-2] = chunks[-2] + last_fresh
            chunks.pop()
    return chunks


def _header(title: str | None, path: str, budget_cap: int, counter: TokenCounter) -> str:
    parts = [p for p in (title, path) if p]
    header = " > ".join(parts)
    if counter.count(header) > budget_cap and path:  # keep only the deepest two section levels
        header = " > ".join(([title] if title else []) + path.split(" > ")[-2:])
    if counter.count(header) > budget_cap:
        header = path.split(" > ")[-1]
    return header[:300]


def chunk_document(
    doc: ParsedDocument,
    counter: TokenCounter,
    *,
    target: int = 300,
    max_tokens: int = 400,
    min_tokens: int = 100,
    overlap_sentences: int = 1,
    embed_max_tokens: int = 512,
) -> tuple[list[Chunk], ChunkStats]:
    stats = ChunkStats()
    out: list[Chunk] = []
    seen: set[str] = set()

    for section in _fold_tiny_parents(_sections(doc, stats), counter, min_tokens):
        header = _header(doc.title, section.path, 100, counter)
        header_tokens = counter.count(header) + 2
        sec_max = min(max_tokens, embed_max_tokens - header_tokens - _SAFETY)
        sec_target = min(target, int(sec_max * 0.8))
        # Tables/code are never packed with anything else, so they may use the whole embedder window.
        units = _units_for(section, sec_max, counter, stats, atomic_budget=embed_max_tokens - header_tokens - _SAFETY)

        for group in _pack(units, sec_target, sec_max, min_tokens, overlap_sentences):
            text = _render(group)
            fresh = [u for u in group if not u.overlap]
            kind = group[0].kind
            alpha = sum(c.isalpha() for c in text) / max(len(re.sub(r"\s", "", text)), 1)
            if sum(u.tokens for u in fresh) < 12 or (kind == "prose" and alpha < 0.4):
                stats.dropped_junk += 1
                continue
            key = re.sub(r"\W+", " ", text.lower()).strip()
            if key in seen:
                stats.dropped_duplicates += 1
                continue
            seen.add(key)

            embed_text = f"{header}\n\n{text}" if header else text
            n_tokens = counter.count(embed_text)
            if n_tokens > embed_max_tokens:
                stats.over_limit += 1
            out.append(Chunk(
                index=len(out), text=text, embed_text=embed_text, section_path=section.path,
                page_start=min(u.page for u in group), page_end=max(u.page for u in group),
                token_count=n_tokens, kind=kind,
                overlap_sentences=sum(1 for u in group if u.overlap),
            ))
            stats.token_counts.append(n_tokens)

    stats.n_chunks = len(out)
    return out, stats

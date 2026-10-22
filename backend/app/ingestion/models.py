from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["title", "heading", "paragraph", "caption", "table", "code", "footnote", "equation"]


@dataclass
class Block:
    kind: BlockKind
    text: str
    page: int  # 1-indexed
    level: int = 0  # headings only: 1 = top
    font_size: float = 0.0
    in_references: bool = False
    y0: float = 0.0  # vertical extent on the page, used to merge neighbouring blocks
    y1: float = 0.0
    rows: str | None = None  # row-per-line reconstruction, set for short/numeric blocks that may be table fragments


@dataclass
class ParsedDocument:
    title: str | None
    author: str | None
    page_count: int
    blocks: list[Block]
    needs_ocr: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    index: int
    text: str  # clean text, what the user sees and what citations quote
    embed_text: str  # "Title > Section\n\ntext" — what actually gets embedded
    section_path: str
    page_start: int
    page_end: int
    token_count: int  # of embed_text, i.e. what the embedder sees
    kind: str = "prose"  # prose | table | code | footnote | caption
    overlap_sentences: int = 0

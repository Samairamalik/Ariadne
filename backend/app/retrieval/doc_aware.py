"""Document-aware retrieval: a question that names a paper should prefer that paper's chunks.

Failure this fixes: with BERT and RoBERTa both in the library, "What vocabulary size does BERT use?" was answered
from RoBERTa's chunk about its 50K byte-level vocabulary, because the RoBERTa paper talks about BERT constantly.

A document's anchor is the short name in its title (the part before the colon: "BERT", "RoBERTa", "RepVGG"), or the
whole title when it has no colon and is short ("Attention Is All You Need"). A question "names" a document when an
anchor appears in it as a whole word, case-insensitive ("RoBERTa" does not match the anchor "BERT").

This is a PREFERENCE, not a filter: targeted documents get their best chunks added to the candidate pool and a score
bonus at rerank time, but chunks from other documents still compete, so a question that also needs another paper
(and the graph walk, which pulls in linked documents) is not cut off.
"""
import re
from uuid import UUID

import asyncpg

MAX_ANCHOR_WORDS = 4
MAX_WHOLE_TITLE_WORDS = 6


def anchors(title: str | None) -> list[str]:
    if not title:
        return []
    title = " ".join(title.split())
    head, sep, _ = title.partition(":")
    cand = head if sep else title
    words = len(cand.split())
    limit = MAX_ANCHOR_WORDS if sep else MAX_WHOLE_TITLE_WORDS
    return [cand] if 0 < words <= limit else []


def names_anchor(question: str, anchor: str) -> bool:
    pat = rf"(?<![A-Za-z0-9]){re.escape(anchor)}(?![A-Za-z0-9])"
    return re.search(pat, question, re.IGNORECASE) is not None


def match_documents(question: str, docs: list[tuple[UUID, str | None]]) -> list[UUID]:
    """Ids of the documents the question names. Empty if it names none, or names every document (no preference)."""
    hit = [d for d, title in docs if any(names_anchor(question, a) for a in anchors(title))]
    return [] if len(hit) == len(docs) else hit


async def target_documents(conn: asyncpg.Connection, question: str) -> list[UUID]:
    rows = await conn.fetch("SELECT id, title FROM documents WHERE status = 'done'")
    return match_documents(question, [(r["id"], r["title"]) for r in rows])

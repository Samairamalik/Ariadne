"""Question -> graph nodes: which entities in the graph is this question about?

Three signals; fuzzy+embed are fused with the same RRF used for chunks, and the chunk signal is only a fallback:
  fuzzy  - phrases from the question vs node names (typo/spacing tolerant, rapidfuzz)
  embed  - question vector vs node-name vectors (catches paraphrases)
  chunks - nodes whose edges point at the chunks hybrid search already retrieved
No match anywhere -> empty list, and the caller falls back to plain chunk RAG.
"""
import re
from dataclasses import dataclass, field
from uuid import UUID

import asyncpg
import numpy as np
from rapidfuzz import fuzz

from app.config import settings
from app.retrieval.fusion import _STOPWORDS, rrf

_TOKEN = re.compile(r"[A-Za-z0-9]+")
MAX_PHRASE_WORDS = 4
CHUNK_FALLBACK_MAX = 3


def is_specific_name(name: str) -> bool:
    """A node worth starting a graph walk from: a multi-word name, or a single word that looks like a name
    (has a capital letter or digit: BERT, Transformer, SQuAD2). Plain lowercase words ("cost", "size",
    "architecture") match half the corpus and only add noise."""
    words = name.split()
    if len(words) >= 2:
        return True
    return bool(words) and any(ch.isupper() or ch.isdigit() for ch in words[0])


@dataclass
class LinkedNode:
    node_id: UUID
    name: str
    node_type: str
    score: float
    found_by: dict[str, int] = field(default_factory=dict)  # signal -> rank, like Candidate.ranks


def candidate_phrases(question: str, max_words: int = MAX_PHRASE_WORDS) -> list[str]:
    """1..max_words consecutive-word phrases from the question, skipping phrases that start or end on a stopword."""
    toks = _TOKEN.findall(question.lower())
    seen, out = set(), []
    for n in range(max_words, 0, -1):
        for i in range(len(toks) - n + 1):
            gram = toks[i : i + n]
            if gram[0] in _STOPWORDS or gram[-1] in _STOPWORDS:
                continue
            ph = " ".join(gram)
            if ph not in seen:
                seen.add(ph)
                out.append(ph)
    return out


def fuzzy_rank(phrases: list[str], names: dict[str, str], cutoff: int) -> list[str]:
    """Node ids whose name matches some phrase at >= cutoff, best match first (ties: longer name first)."""
    best: dict[str, float] = {}
    for nid, name in names.items():
        low = name.lower()
        score = max((fuzz.ratio(ph, low) for ph in phrases), default=0)
        if score >= cutoff:
            best[nid] = score
    return sorted(best, key=lambda i: (-best[i], -len(names[i]), i))


async def link_entities(
    conn: asyncpg.Connection, question: str, query_vec: np.ndarray, chunk_ids: list[UUID],
) -> list[LinkedNode]:
    rows = await conn.fetch("SELECT id, canonical_name, node_type FROM nodes")
    if not rows:
        return []
    info = {str(r["id"]): (r["canonical_name"], r["node_type"]) for r in rows}

    specific = {i: v for i, v in info.items() if is_specific_name(v[0])}
    fuzzy = fuzzy_rank(candidate_phrases(question), {i: v[0] for i, v in specific.items()}, settings.link_fuzzy_cutoff)

    emb_rows = await conn.fetch(
        "SELECT id, 1 - (embedding <=> $1) AS cos FROM nodes WHERE embedding IS NOT NULL "
        "ORDER BY embedding <=> $1 LIMIT $2", query_vec, settings.link_embed_k)
    embed = [str(r["id"]) for r in emb_rows if r["cos"] >= settings.link_embed_cutoff and str(r["id"]) in specific]

    chunk_rows = await conn.fetch(
        """SELECT n.id, count(*) AS c FROM edges e JOIN nodes n ON n.id IN (e.source_id, e.target_id)
           WHERE e.chunk_id = ANY($1::uuid[]) GROUP BY n.id ORDER BY c DESC, n.id LIMIT 20""", chunk_ids)
    from_chunks = [str(r["id"]) for r in chunk_rows if str(r["id"]) in specific]

    if fuzzy or embed:  # the question names (or paraphrases) real nodes: trust those signals only
        fused = rrf({"fuzzy": fuzzy, "embed": embed}, k=settings.rrf_k)
    else:  # nothing named: the retrieved chunks are the only lead, and a weak one, so keep few
        fused = rrf({"chunks": from_chunks[:CHUNK_FALLBACK_MAX]}, k=settings.rrf_k)
    return [LinkedNode(UUID(f.id), info[f.id][0], info[f.id][1], f.score, f.ranks)
            for f in fused[: settings.link_max_nodes]]

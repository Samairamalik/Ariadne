"""Step 7: community summaries, for questions no single chunk can answer ("what are the main themes?").

Louvain splits the graph into groups of densely connected nodes; an LLM writes a short title and summary from each
group's facts; the summary is embedded so a question can be matched to it. Building costs one LLM call per
community, so it runs on demand (POST /api/graph/communities/build), not at ingestion.

Summaries are background for the answer, never a citation: the answer prompt tells the model not to cite them, and
every claim still has to be backed by a numbered source chunk.
"""
import asyncio
import json
import logging
import re
from uuid import UUID

import asyncpg
import networkx as nx

from app import db
from app.config import settings
from app.embeddings import get_embedder
from app.llm import get_router

log = logging.getLogger("ariadne.communities")

_GLOBAL = re.compile(
    r"\b(main|key|major|overall|big[- ]picture)\b.*\b(themes?|topics?|ideas?|points?|takeaways?|findings?)\b"
    r"|\b(overview|summari[sz]e|summary)\b|\bwhat (is|are) (this|these|the) (document|documents|papers?|corpus)\b"
    r"|\bacross (all|the|these) (documents|papers)\b", re.I)


def is_global_question(question: str) -> bool:
    return bool(_GLOBAL.search(question))


def detect(edges: list[tuple[UUID, UUID]], min_size: int) -> list[list[UUID]]:
    """Louvain communities of at least min_size nodes, largest first (deterministic)."""
    g = nx.Graph()
    g.add_edges_from((a, b) for a, b in edges if a != b)
    if g.number_of_edges() == 0:
        return []
    comms = nx.community.louvain_communities(g, seed=42)
    out = [sorted(c, key=str) for c in comms if len(c) >= min_size]
    return sorted(out, key=lambda c: (-len(c), str(c[0])))


PROMPT = (
    "You summarize a group of related facts from the user's documents. Return JSON "
    '{"title": "<=6 words", "summary": "2-4 sentences covering what the group is about"}. '
    "Use only the facts given."
)


async def _summarize(facts: list[str]) -> tuple[str, str] | None:
    try:
        res = await get_router().chat(
            [{"role": "system", "content": PROMPT}, {"role": "user", "content": "\n".join(facts)}],
            json_mode=True)
        d = json.loads(res.text)
        title, summary = str(d["title"]).strip()[:80], str(d["summary"]).strip()[:800]
        return (title, summary) if title and summary else None
    except Exception as e:  # noqa: BLE001 - one bad community must not abort the build
        log.warning("community summary failed: %s", e)
        return None


async def build_communities() -> dict:
    async with db.pool().acquire() as conn:
        edge_rows = await conn.fetch("SELECT source_id, target_id FROM edges")
        comms = detect([(r["source_id"], r["target_id"]) for r in edge_rows], settings.community_min_size)
        built = []
        for ids in comms:
            facts = await conn.fetch(
                """SELECT s.canonical_name s, e.predicate p, t.canonical_name o FROM edges e
                   JOIN nodes s ON s.id = e.source_id JOIN nodes t ON t.id = e.target_id
                   WHERE e.source_id = ANY($1::uuid[]) AND e.target_id = ANY($1::uuid[]) LIMIT 40""", ids)
            got = await _summarize([f"({f['s']}) -[{f['p']}]-> ({f['o']})" for f in facts])
            if got:
                built.append((got[0], got[1], ids))
        if not built:
            return {"communities_found": len(comms), "built": 0}
        vecs = await asyncio.to_thread(get_embedder().embed_passages, [f"{t}. {s}" for t, s, _ in built])
        async with conn.transaction():
            await conn.execute("DELETE FROM communities")
            await conn.executemany(
                "INSERT INTO communities (title, summary, node_ids, size, embedding) VALUES ($1,$2,$3,$4,$5)",
                [(t, s, ids, len(ids), v) for (t, s, ids), v in zip(built, vecs)])
    return {"communities_found": len(comms), "built": len(built)}


async def best_overviews(conn: asyncpg.Connection, query_vec) -> list[asyncpg.Record]:
    rows = await conn.fetch(
        "SELECT title, summary, size, 1 - (embedding <=> $1) AS cos FROM communities WHERE embedding IS NOT NULL "
        "ORDER BY embedding <=> $1 LIMIT $2", query_vec, settings.community_top_k)
    return [r for r in rows if r["cos"] >= settings.community_cutoff]

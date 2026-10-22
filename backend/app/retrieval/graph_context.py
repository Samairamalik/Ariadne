"""Step 9: graph-augmented retrieval.

question -> linked nodes (seeds) -> walk edges up to `graph_max_hops` -> the chunks those edges came from.

Chunk search finds passages that LOOK like the question; the walk finds passages that are CONNECTED to what the
question names, including ones in other documents that share an entity. The extra chunks are appended after the
normal top-k (never replacing them), and the traversed facts are returned so the answer can connect them.

Every Fact carries its evidence (exact quote, paper, section, node ids), which is what the evidence trail, the entity
profile and path finding are built from.

Two ways to order the graph-found chunks (settings.graph_scoring):
  "hops"  count the facts pointing at a chunk, weighted 1/hop;
  "ppr"   personalized PageRank: a random walk that restarts at the question's entities; a chunk scores the walk
          mass on the endpoints of its edges. Handles several routes to one chunk and damps hub nodes by itself.
"""
import asyncio
from dataclasses import dataclass, field
from uuid import UUID

import asyncpg
import networkx as nx
import numpy as np

from app.config import settings
from app.retrieval.graph import is_specific_name, link_entities
from app.retrieval.hybrid import Candidate
from app.retrieval.rerank import rerank


@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    chunk_id: UUID
    hop: int
    evidence: str = ""            # exact quote the fact was extracted from
    document_id: UUID | None = None
    doc_title: str = ""
    section: str = ""
    subject_id: UUID | None = None
    object_id: UUID | None = None


@dataclass
class GraphContext:
    seeds: list = field(default_factory=list)          # LinkedNode
    facts: list[Fact] = field(default_factory=list)
    extra: list[Candidate] = field(default_factory=list)  # chunks reached only through the graph
    hops: int = 0


async def walk(conn: asyncpg.Connection, seed_ids: list[UUID]) -> list[Fact]:
    """Breadth-first over edges (either direction). Hub nodes are not expanded past hop 1: a node touching dozens of
    edges (like the paper's own model name) would flood the context with loosely related facts."""
    facts, seen_edges, frontier = [], set(), list(seed_ids)
    for hop in range(1, settings.graph_max_hops + 1):
        if not frontier:
            break
        rows = await conn.fetch(
            """SELECT e.id, e.chunk_id, e.source_id, e.target_id, e.predicate, e.evidence_text, e.document_id,
                      s.canonical_name AS s, t.canonical_name AS t, coalesce(d.title, d.filename) AS doc_title,
                      c.section_path
               FROM edges e JOIN nodes s ON s.id = e.source_id JOIN nodes t ON t.id = e.target_id
                    LEFT JOIN documents d ON d.id = e.document_id LEFT JOIN chunks c ON c.id = e.chunk_id
               WHERE (e.source_id = ANY($1::uuid[]) OR e.target_id = ANY($1::uuid[])) AND e.chunk_id IS NOT NULL
               ORDER BY e.created_at, e.id LIMIT $2""", frontier, settings.graph_hop_edge_cap)
        nxt = set()
        for r in rows:
            if r["id"] in seen_edges:
                continue
            seen_edges.add(r["id"])
            facts.append(Fact(r["s"], r["predicate"], r["t"], r["chunk_id"], hop, r["evidence_text"] or "",
                              r["document_id"], r["doc_title"] or "", r["section_path"] or "", r["source_id"], r["target_id"]))
            nxt.update((r["source_id"], r["target_id"]))
        nxt -= set(frontier) | set(seed_ids)
        if nxt:
            deg = await conn.fetch(
                "SELECT n.id, count(*) c FROM nodes n JOIN edges e ON n.id IN (e.source_id, e.target_id) "
                "WHERE n.id = ANY($1::uuid[]) GROUP BY n.id", list(nxt))
            hubs = {d["id"] for d in deg if d["c"] > settings.graph_hub_degree}
            nxt -= hubs
        frontier = list(nxt)
    return facts


def rank_graph_chunks(facts: list[Fact], exclude: set[UUID]) -> list[tuple[UUID, int]]:
    """(chunk_id, best hop) ordered by how many facts point at the chunk, weighted 1/hop; known chunks excluded."""
    score: dict[UUID, float] = {}
    best_hop: dict[UUID, int] = {}
    for f in facts:
        if f.chunk_id in exclude:
            continue
        score[f.chunk_id] = score.get(f.chunk_id, 0.0) + 1.0 / f.hop
        best_hop[f.chunk_id] = min(best_hop.get(f.chunk_id, f.hop), f.hop)
    order = sorted(score, key=lambda c: (-score[c], str(c)))
    return [(c, best_hop[c]) for c in order]


def rank_graph_chunks_ppr(facts: list[Fact], seed_ids: list[UUID], exclude: set[UUID],
                          alpha: float = 0.85) -> list[tuple[UUID, int]]:
    """Order chunks by personalized PageRank over the walked subgraph.

    The walk restarts at the seed nodes with probability 1 - alpha, so mass concentrates near the question's entities,
    flows along every route, and is spread thin across hubs. A chunk's score is the mass on its edges' endpoints.
    Facts without node ids (old data) are skipped; if nothing is usable the hop-count ranking is used instead."""
    usable = [f for f in facts if f.subject_id is not None and f.object_id is not None]
    seeds = [s for s in dict.fromkeys(seed_ids) if any(s in (f.subject_id, f.object_id) for f in usable)]
    if not usable or not seeds:
        return rank_graph_chunks(facts, exclude)
    g = nx.Graph()
    for f in usable:
        g.add_edge(f.subject_id, f.object_id)
    pr = nx.pagerank(g, alpha=alpha, personalization={s: 1.0 / len(seeds) for s in seeds})
    score: dict[UUID, float] = {}
    best_hop: dict[UUID, int] = {}
    for f in usable:
        if f.chunk_id in exclude:
            continue
        score[f.chunk_id] = score.get(f.chunk_id, 0.0) + pr[f.subject_id] + pr[f.object_id]
        best_hop[f.chunk_id] = min(best_hop.get(f.chunk_id, f.hop), f.hop)
    order = sorted(score, key=lambda c: (-score[c], str(c)))
    return [(c, best_hop[c]) for c in order]


async def fetch_candidates(conn: asyncpg.Connection, chunk_ids: list[UUID], query_vec: np.ndarray,
                           ranks: dict[UUID, dict] | None = None) -> list[Candidate]:
    """Chunks as retrieval Candidates, in the order given, tagged with how they were found (`ranks`)."""
    if not chunk_ids:
        return []
    rows = await conn.fetch(
        """SELECT c.id, c.document_id, coalesce(d.title, d.filename) AS doc_title, d.filename, c.section_path,
                  c.page_start, c.page_end, c.chunk_index, c.text, 1 - (c.embedding <=> $1) AS cosine
           FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ANY($2::uuid[])""",
        query_vec, list(chunk_ids))
    by_id = {r["id"]: r for r in rows}
    out = []
    for cid in chunk_ids:
        r = by_id.get(cid)
        if r:
            out.append(Candidate(r["id"], r["document_id"], r["doc_title"], r["filename"], r["section_path"] or "",
                                 r["page_start"], r["page_end"], r["chunk_index"], r["text"], float(r["cosine"]),
                                 0.0, (ranks or {}).get(cid, {})))
    return out


def evidence_trail(facts: list[Fact], source_index: dict[UUID, int], limit: int = 12) -> list[dict]:
    """The chain of facts behind an answer, nearest hop first, each with its quote, paper and the source it appears as.

    Only facts whose chunk is actually in the answer's sources are listed, so every step can be checked there."""
    ordered = sorted((f for f in facts if f.chunk_id in source_index), key=lambda f: (f.hop, f.doc_title, f.subject))
    seen, out = set(), []
    for f in ordered:
        key = (f.subject, f.predicate, f.object, f.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        out.append({"hop": f.hop, "fact": f"({f.subject}) -[{f.predicate}]-> ({f.object})", "quote": f.evidence,
                    "document": f.doc_title, "section": f.section, "source": source_index[f.chunk_id]})
        if len(out) >= limit:
            break
    return out


async def build_graph_context(
    conn: asyncpg.Connection, question: str, query_vec: np.ndarray, top: list[Candidate],
) -> GraphContext:
    seeds = await link_entities(conn, question, query_vec, [c.chunk_id for c in top])
    # Multi-hop: the answer's entities are often inside the best chunk, not in the question ("what are the three
    # families?" -> the intro chunk names them). So also walk from the entities of the top retrieved chunks.
    seed_ids = [s.node_id for s in seeds]
    rows = await conn.fetch(
        """SELECT n.id, n.canonical_name, count(*) AS c FROM edges e JOIN nodes n ON n.id IN (e.source_id, e.target_id)
           WHERE e.chunk_id = ANY($1::uuid[]) GROUP BY n.id, n.canonical_name ORDER BY c DESC, n.id LIMIT 20""",
        [c.chunk_id for c in top[:2]])
    chunk_seeds = [r["id"] for r in rows if is_specific_name(r["canonical_name"]) and r["id"] not in seed_ids][:4]
    if not seed_ids and not chunk_seeds:
        return GraphContext()
    facts = await walk(conn, seed_ids + chunk_seeds)
    # Over-fetch, then keep only what the cross-encoder says actually bears on THIS question: being linked to the
    # same entity is a lead, not evidence, and unfiltered leads made answers wander off the question.
    known = {c.chunk_id for c in top}
    ranked = (rank_graph_chunks_ppr(facts, seed_ids + chunk_seeds, known) if settings.graph_scoring == "ppr"
              else rank_graph_chunks(facts, known))
    picked = ranked[: settings.graph_extra_chunks * 3]
    extra = await fetch_candidates(conn, [c for c, _ in picked], query_vec, {c: {"graph": h} for c, h in picked})
    if extra:
        scored, reranked = await asyncio.to_thread(rerank, question, extra, len(extra))
        if reranked:
            extra = [c for c in scored if c.rerank_score >= settings.graph_extra_min_rerank][: settings.graph_extra_chunks]
        else:
            extra = extra[: settings.graph_extra_chunks]
    used = {c.chunk_id for c in top} | {c.chunk_id for c in extra}
    facts = [f for f in facts if f.chunk_id in used]
    return GraphContext(seeds, facts, extra, max((f.hop for f in facts), default=0))

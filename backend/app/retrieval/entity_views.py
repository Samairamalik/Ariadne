"""Structured views of the knowledge graph: what plain chunk retrieval cannot give.

  entity_profile      everything the library says about ONE entity, gathered across papers, every fact with its quote
  find_paths          the shortest chains of facts between TWO entities, each step with its quote and paper
  aggregate_documents which papers say something about an entity (graph facts, plus papers that mention the name
                      even where extraction produced no fact)
  entities_in_question  which graph nodes a question names, with the phrase that matched, in question order

The database reads are thin; the logic (grouping, graph building, path search) is in plain functions so it is tested
without a database.
"""
import re
from collections import Counter, defaultdict
from itertools import islice
from uuid import UUID

import networkx as nx
from rapidfuzz import fuzz

from app.retrieval.graph import candidate_phrases, is_specific_name

EDGE_SQL = """
    SELECT e.id, e.predicate, e.evidence_text, e.chunk_id, e.document_id, e.source_id, e.target_id,
           s.canonical_name AS s, t.canonical_name AS t, coalesce(d.title, d.filename) AS doc_title, d.filename,
           c.section_path, c.page_start, c.chunk_index
    FROM edges e JOIN nodes s ON s.id = e.source_id JOIN nodes t ON t.id = e.target_id
         JOIN documents d ON d.id = e.document_id LEFT JOIN chunks c ON c.id = e.chunk_id
"""


def fact_dict(r, node_id: UUID | None = None) -> dict:
    """One edge row as a plain, JSON-ready fact."""
    d = {"subject": r["s"], "predicate": r["predicate"], "object": r["t"], "quote": r["evidence_text"] or "",
         "document": r["doc_title"], "document_id": str(r["document_id"]), "section": r["section_path"] or "",
         "page": r["page_start"], "chunk_id": str(r["chunk_id"]) if r["chunk_id"] else None}
    if node_id is not None:
        d["direction"] = "out" if r["source_id"] == node_id else "in"
    return d


def group_profile(node: dict, rows: list, top_related: int = 10) -> dict:
    """Group an entity's edges by paper (in the order papers were added), with predicate and neighbour counts."""
    by_doc: dict = defaultdict(lambda: {"facts": []})
    related, predicates = Counter(), Counter()
    for r in rows:
        doc = by_doc[r["document_id"]]
        doc.update(document=r["doc_title"], document_id=str(r["document_id"]), filename=r["filename"])
        doc["facts"].append(fact_dict(r, node["id"]))
        predicates[r["predicate"]] += 1
        related[r["t"] if r["source_id"] == node["id"] else r["s"]] += 1
    return {"entity": {"id": str(node["id"]), "name": node["canonical_name"], "type": node["node_type"],
                       "aliases": list(node.get("aliases") or []), "documents": len(by_doc), "facts": len(rows)},
            "documents": list(by_doc.values()),
            "predicates": dict(predicates.most_common(8)),
            "related": [{"name": n, "facts": c} for n, c in related.most_common(top_related)]}


def build_graph(rows: list, allow: set) -> nx.Graph:
    """Undirected graph of nodes joined by facts. Intermediate nodes must look like real entities (a plain lowercase
    word such as 'cost' would create meaningless shortcuts), except the endpoints named in `allow`."""
    g = nx.Graph()
    ok = lambda nid, name: nid in allow or is_specific_name(name)  # noqa: E731
    for r in rows:
        if r["source_id"] != r["target_id"] and ok(r["source_id"], r["s"]) and ok(r["target_id"], r["t"]):
            g.add_node(r["source_id"], name=r["s"])
            g.add_node(r["target_id"], name=r["t"])
            g.add_edge(r["source_id"], r["target_id"])
            g.edges[r["source_id"], r["target_id"]].setdefault("rows", []).append(r)
    return g


def steps_for_path(g: nx.Graph, path: list) -> list[dict]:
    """The facts along one path, one step per hop, each with the quote and paper of its best edge."""
    steps = []
    for a, b in zip(path, path[1:]):
        r = sorted(g.edges[a, b]["rows"], key=lambda x: (x["doc_title"], x["chunk_index"] or 0))[0]
        steps.append({**fact_dict(r), "from": g.nodes[a]["name"], "to": g.nodes[b]["name"],
                      "direction": "forward" if r["source_id"] == a else "reverse"})
    return steps


def paths_between(g: nx.Graph, a, b, max_hops: int = 3, k: int = 3) -> list[dict]:
    """Up to k shortest paths of at most max_hops facts. Empty if the entities are not connected that closely."""
    if a == b or a not in g or b not in g:
        return []
    try:
        found = [p for p in islice(nx.shortest_simple_paths(g, a, b), k) if len(p) - 1 <= max_hops]
    except nx.NetworkXNoPath:
        return []
    return [{"hops": len(p) - 1, "steps": steps_for_path(g, p)} for p in found]


def phrase_matches(phrases: list[str], nodes: list[dict], cutoff: int = 88) -> list[dict]:
    """Greedy longest-phrase-first matching of question phrases to node names, no word used twice.

    Several nodes may match one phrase (BERT, BERTBASE): the highest score wins, then the node that appears in more
    papers, then the more often mentioned. Returns matches in the order their phrases appear in the question."""
    taken: set[str] = set()
    out = []
    for ph in phrases:
        words = ph.split()
        if taken & set(words):
            continue
        best, best_key = None, None
        for n in nodes:
            score = max([fuzz.ratio(ph, n["canonical_name"].lower()), *(fuzz.ratio(ph, a.lower()) for a in n.get("aliases") or [])])
            if score >= cutoff:
                key = (score, n["documents"], n["mention_count"])
                if best_key is None or key > best_key:
                    best, best_key = n, key
        if best is not None:
            out.append({"node": best, "phrase": ph, "score": best_key[0]})
            taken |= set(words)
    return out


async def entities_in_question(conn, question: str, cutoff: int = 88) -> list[dict]:
    rows = await conn.fetch(
        """SELECT n.id, n.canonical_name, n.node_type, n.mention_count, n.aliases,
                  (SELECT count(DISTINCT e.document_id) FROM edges e WHERE n.id IN (e.source_id, e.target_id)) AS documents
           FROM nodes n""")
    nodes = [dict(r) for r in rows if is_specific_name(r["canonical_name"]) and r["documents"] > 0]
    found = phrase_matches(candidate_phrases(question), nodes, cutoff)
    low = question.lower()
    return sorted(found, key=lambda m: low.find(m["phrase"]))


async def find_nodes(conn, name: str, limit: int = 5) -> list[dict]:
    """Nodes whose name or alias matches `name` (exact first, then close spellings)."""
    q = name.lower().strip()
    rows = await conn.fetch(
        """SELECT n.id, n.canonical_name, n.node_type, n.mention_count, n.aliases,
                  (SELECT count(DISTINCT e.document_id) FROM edges e WHERE n.id IN (e.source_id, e.target_id)) AS documents
           FROM nodes n""")
    scored = []
    for r in rows:
        score = max([fuzz.ratio(q, r["canonical_name"].lower()), *(fuzz.ratio(q, a.lower()) for a in r["aliases"] or [])])
        if score >= 85 and r["documents"] > 0:
            scored.append((score, r["documents"], r["mention_count"], dict(r)))
    scored.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    return [{**x[3], "match": x[0]} for x in scored[:limit]]


async def entity_profile(conn, node_id: UUID) -> dict | None:
    node = await conn.fetchrow(
        "SELECT id, canonical_name, node_type, aliases FROM nodes WHERE id = $1", node_id)
    if not node:
        return None
    rows = await conn.fetch(EDGE_SQL + " WHERE e.source_id = $1 OR e.target_id = $1 ORDER BY d.ingested_at, c.chunk_index", node_id)
    return group_profile(dict(node), rows)


async def find_paths(conn, a_id: UUID, b_id: UUID, max_hops: int = 3, k: int = 3) -> list[dict]:
    rows = await conn.fetch(EDGE_SQL)
    return paths_between(build_graph(rows, {a_id, b_id}), a_id, b_id, max_hops, k)


async def aggregate_documents(conn, node_id: UUID, predicate: str | None = None) -> list[dict]:
    """Per paper: the graph facts about the entity (optionally only predicates containing `predicate`), plus the best
    chunk that mentions the entity by name, so a paper where extraction missed the fact still appears."""
    node = await conn.fetchrow("SELECT canonical_name, aliases FROM nodes WHERE id = $1", node_id)
    if not node:
        return []
    names = [node["canonical_name"], *(node["aliases"] or [])]
    rows = await conn.fetch(EDGE_SQL + " WHERE e.source_id = $1 OR e.target_id = $1 ORDER BY d.ingested_at, c.chunk_index", node_id)
    if predicate:
        rows = [r for r in rows if predicate.lower() in r["predicate"].lower()]
    docs: dict = {}
    for r in rows:
        d = docs.setdefault(r["document_id"], {"document": r["doc_title"], "document_id": str(r["document_id"]),
                                               "facts": [], "chunk_id": str(r["chunk_id"]) if r["chunk_id"] else None})
        d["facts"].append(fact_dict(r, node_id))
    patt = "|".join(re.escape(n) for n in names)
    for m in await conn.fetch(
            """SELECT DISTINCT ON (c.document_id) c.document_id, c.id, coalesce(d.title, d.filename) AS doc_title
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.text ~* $1 ORDER BY c.document_id, c.chunk_index""", r"\y(" + patt + r")\y"):
        d = docs.setdefault(m["document_id"], {"document": m["doc_title"], "document_id": str(m["document_id"]),
                                               "facts": [], "chunk_id": str(m["id"])})
        d["mentioned"] = True
    return list(docs.values())

"""Entity merging: nodes that are the same thing but exist as separate rows.

A shared name is not enough to merge ("bank" the river vs "bank" the lender). The learner's rule: merge only when
the CONTEXTS are close, where a node's context is the mean embedding of the chunks its edges came from. Every merge
is logged in `merges` (method='auto') with its similarity, so it can be reviewed.

Two tiers, both gated by context similarity:
  same-name  names equal once case, spaces and hyphens are ignored ("re-parameterization" = "reparameterization",
             "RepVGG block" = "RepVGGBlock"); also covers one name stored under several type labels.
  contained  one name's words are all inside the other's, across different documents, and the longer name adds ONLY
             a generic suffix (function, method, technique, approach) or a paper's own name ("structural
             reparameterization" inside "RepVGG structural reparameterization"; "GELU activation" inside "GELU
             activation function"). Context similarity alone cannot police this tier: in testing, wrong pairs
             ("learning rate" / "learning rate optimizer", 0.89) scored higher than the right one (0.77). So the shorter
             name needs >= 2 words, and "attention" never merges with "self-attention".
Anything fuzzier (abbreviations, paraphrases) is NOT merged here.
"""
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

import asyncpg
import numpy as np

from app import db
from app.config import settings
from app.retrieval.doc_aware import anchors

log = logging.getLogger("ariadne.resolve")
MAX_EXTRA_WORDS = 2
GENERIC_SUFFIXES = frozenset({"function", "method", "technique", "approach"})


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def compact_key(name: str) -> str:
    """Name with everything but letters and digits removed: spelling variants collapse to the same key."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def name_tokens(name: str) -> frozenset[str]:
    """Words of a name; hyphens inside a word are dropped so 're-parameterization' is one token 'reparameterization'."""
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", name.lower())
    return frozenset(w.replace("-", "") for w in words)


def is_contained_variant(short: frozenset[str], long: frozenset[str], allowed_extra: frozenset[str] = GENERIC_SUFFIXES) -> bool:
    """`short` is inside `long` and everything `long` adds is allowed (generic suffix or a document's own name)."""
    extra = long - short
    return len(short) >= 2 and bool(extra) and short < long and len(extra) <= MAX_EXTRA_WORDS and extra <= allowed_extra


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


@dataclass
class Pair:
    name: str
    kept_id: UUID
    kept_type: str
    other_id: UUID
    other_type: str
    similarity: float | None  # None = a node has no source chunks to compare
    kind: str = "same-name"   # same-name | contained
    other_name: str = ""
    merged: bool = False


def _cutoff(p: Pair) -> float:
    return settings.resolve_subset_cutoff if p.kind == "contained" else settings.resolve_context_cutoff


async def _node_info(conn: asyncpg.Connection):
    rows = await conn.fetch("SELECT id, canonical_name, node_type, mention_count FROM nodes")
    ctx_rows = await conn.fetch(
        """SELECT n.id, avg(c.embedding) AS v FROM nodes n
           JOIN edges e ON n.id IN (e.source_id, e.target_id) JOIN chunks c ON c.id = e.chunk_id GROUP BY n.id""")
    doc_rows = await conn.fetch(
        """SELECT n.id, array_agg(DISTINCT e.document_id) AS docs FROM nodes n
           JOIN edges e ON n.id IN (e.source_id, e.target_id) GROUP BY n.id""")
    ctx = {r["id"]: np.asarray(r["v"], dtype=np.float32) for r in ctx_rows}
    docs = {r["id"]: set(r["docs"]) for r in doc_rows}
    return rows, ctx, docs


def _pair(keep, other, ctx, kind: str) -> Pair:
    sim = cosine(ctx[keep["id"]], ctx[other["id"]]) if keep["id"] in ctx and other["id"] in ctx else None
    return Pair(keep["canonical_name"], keep["id"], keep["node_type"], other["id"], other["node_type"], sim,
                kind, other["canonical_name"])


def _order(a, b):
    """Survivor first: the node mentioned more, ties broken by id (deterministic)."""
    return (a, b) if (-a["mention_count"], str(a["id"])) <= (-b["mention_count"], str(b["id"])) else (b, a)


async def find_pairs(conn: asyncpg.Connection) -> list[Pair]:
    """Every merge candidate with its context similarity. Changes nothing."""
    rows, ctx, docs = await _node_info(conn)
    titles = await conn.fetch("SELECT title FROM documents WHERE status = 'done'")
    # a paper's own name ("repvgg", "bert") may prefix a concept: "RepVGG structural reparameterization"
    allowed = GENERIC_SUFFIXES | frozenset(
        a.lower() for t in titles for a in anchors(t["title"]) if len(a.split()) == 1 and a.isalnum())
    by_id = {r["id"]: r for r in rows}
    pairs: list[Pair] = []
    seen: set[frozenset] = set()

    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        k = compact_key(r["canonical_name"])
        if k:
            groups[k].append(r)
    for g in groups.values():
        if len(g) < 2:
            continue
        g.sort(key=lambda r: (-r["mention_count"], str(r["id"])))
        for other in g[1:]:
            pairs.append(_pair(g[0], other, ctx, "same-name"))
            seen.add(frozenset((g[0]["id"], other["id"])))

    toks = {r["id"]: name_tokens(r["canonical_name"]) for r in rows}
    inv: dict[str, set] = defaultdict(set)
    for nid, ts in toks.items():
        for t in ts:
            inv[t].add(nid)
    for a, ta in toks.items():
        if len(ta) < 2:
            continue
        for b in set.intersection(*(inv[t] for t in ta)) - {a}:
            if not is_contained_variant(ta, toks[b], allowed) or frozenset((a, b)) in seen:
                continue
            da, db_ = docs.get(a, set()), docs.get(b, set())
            if len(da | db_) < 2:  # both live in one single document: nothing to connect across papers
                continue
            keep, other = _order(by_id[a], by_id[b])
            pairs.append(_pair(keep, other, ctx, "contained"))
            seen.add(frozenset((a, b)))
    return pairs


async def _merge(conn: asyncpg.Connection, p: Pair, kept_id: UUID, other_id: UUID) -> None:
    edges = await conn.fetch(
        "SELECT id, source_id, target_id FROM edges WHERE source_id = $1 OR target_id = $1", other_id)
    for e in edges:
        src = kept_id if e["source_id"] == other_id else e["source_id"]
        tgt = kept_id if e["target_id"] == other_id else e["target_id"]
        if src == tgt:  # a fact about "X relates to X" after merging carries no information
            await conn.execute("DELETE FROM edges WHERE id = $1", e["id"])
            continue
        try:
            async with conn.transaction():  # savepoint: the re-pointed edge may already exist on the survivor
                await conn.execute("UPDATE edges SET source_id = $2, target_id = $3 WHERE id = $1", e["id"], src, tgt)
        except asyncpg.UniqueViolationError:
            await conn.execute("DELETE FROM edges WHERE id = $1", e["id"])
    other_name = await conn.fetchval("SELECT canonical_name FROM nodes WHERE id = $1", other_id)
    await conn.execute(
        "UPDATE nodes SET aliases = CASE WHEN $2 = ANY(aliases) OR $2 = canonical_name THEN aliases "
        "ELSE array_append(aliases, $2) END WHERE id = $1", kept_id, other_name)
    await conn.execute(
        "INSERT INTO merges (kept_id, merged_id, merged_name, similarity, method) VALUES ($1,$2,$3,$4,'auto')",
        kept_id, other_id, other_name, p.similarity)
    await conn.execute("DELETE FROM nodes WHERE id = $1", other_id)
    await conn.execute(
        """UPDATE nodes n SET mention_count = GREATEST(1, (SELECT count(*) FROM edges e
                                                           WHERE e.source_id = n.id OR e.target_id = n.id))
           WHERE n.id = $1""", kept_id)


async def resolve_duplicates(cutoff: float | None = None, apply: bool = True) -> list[Pair]:
    """Merge candidate pairs whose context similarity >= their tier's cutoff (`cutoff` overrides the same-name
    tier's). apply=False reports without changing anything."""
    async with db.pool().acquire() as conn:
        async with conn.transaction():
            pairs = await find_pairs(conn)
            replaced: dict[UUID, UUID] = {}

            def final(i: UUID) -> UUID:
                while i in replaced:
                    i = replaced[i]
                return i

            for p in pairs:
                limit = cutoff if (cutoff is not None and p.kind == "same-name") else _cutoff(p)
                if not apply or p.similarity is None or p.similarity < limit:
                    continue
                kept, other = final(p.kept_id), final(p.other_id)
                if kept == other:
                    p.merged = True  # already merged through another pair
                    continue
                await _merge(conn, p, kept, other)
                replaced[other] = kept
                p.merged = True
    if apply:
        log.info("entity merge: %d of %d candidate pairs merged", sum(p.merged for p in pairs), len(pairs))
    return pairs

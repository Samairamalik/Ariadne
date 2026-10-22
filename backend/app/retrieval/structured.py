"""Turn a graph intent (profile / path / aggregate) into evidence for the answer: facts, quotes, and the chunks to cite.

The graph decides WHICH passages matter (gathered across papers, not just the nearest to the question); the language
model still writes the answer, and still has to cite numbered source chunks. If the graph cannot support the intent
(an entity is not in it, or no chain of facts connects two entities) this returns None and the caller falls back to
ordinary retrieval, so an unusual question never gets a worse answer than before.
"""
import re
from dataclasses import dataclass, field
from uuid import UUID

from app.retrieval.bm25 import bm25_search
from app.retrieval.entity_views import (aggregate_documents, entities_in_question, entity_profile, find_paths)
from app.retrieval.graph_context import Fact

MAX_CHUNKS = 8
FACTS_PER_DOC = 3


@dataclass
class Structured:
    intent: str
    entities: list[str]
    facts: list[Fact] = field(default_factory=list)
    chunk_ids: list[UUID] = field(default_factory=list)
    detail: dict = field(default_factory=dict)  # the full structured result, returned to the client


def to_fact(d: dict, hop: int = 1) -> Fact:
    return Fact(d["subject"], d["predicate"], d["object"], UUID(d["chunk_id"]), hop, d["quote"],
                UUID(d["document_id"]), d["document"], d["section"])


def pick_per_document(docs: list[dict], per_doc: int = FACTS_PER_DOC, limit: int = MAX_CHUNKS) -> tuple[list[Fact], list[UUID]]:
    """Up to per_doc facts per paper (distinct chunks first), papers taken in turn so every paper is represented,
    capped at `limit` chunks overall."""
    facts, chunk_ids, seen = [], [], set()
    queues = [[f for f in d["facts"] if f["chunk_id"]] for d in docs]
    taken = [0] * len(queues)
    progress = True
    while progress and len(chunk_ids) < limit:
        progress = False
        for i, q in enumerate(queues):
            while q and taken[i] < per_doc:
                f = q.pop(0)
                if f["chunk_id"] in seen:
                    continue
                seen.add(f["chunk_id"])
                facts.append(to_fact(f))
                chunk_ids.append(UUID(f["chunk_id"]))
                taken[i] += 1
                progress = True
                break
            if len(chunk_ids) >= limit:
                break
    return facts, chunk_ids


async def best_mention_chunk(conn, doc_id: str, names: list[str], question: str) -> UUID | None:
    """The chunk of one paper that names the entity AND best matches the question (BM25 inside the paper)."""
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.I)
    for cid in await bm25_search(conn, question, 8, [UUID(doc_id)]):
        text = await conn.fetchval("SELECT text FROM chunks WHERE id = $1", cid)
        if pattern.search(text or ""):
            return cid
    row = await conn.fetchrow("SELECT id FROM chunks WHERE document_id = $1 AND text ~* $2 ORDER BY chunk_index LIMIT 1",
                              UUID(doc_id), r"\y(" + "|".join(re.escape(n) for n in names) + r")\y")
    return row["id"] if row else None


async def build_structured(conn, intent: str, question: str) -> Structured | None:
    ents = await entities_in_question(conn, question)
    if not ents:
        return None
    if intent in ("profile", "aggregate"):
        node = ents[0]["node"]
        if intent == "profile":
            prof = await entity_profile(conn, node["id"])
            if not prof or not prof["documents"]:
                return None
            facts, chunk_ids = pick_per_document(prof["documents"])
            return Structured("profile", [node["canonical_name"]], facts, chunk_ids, prof)
        docs = await aggregate_documents(conn, node["id"])
        if not docs:
            return None
        names = [node["canonical_name"], *(node.get("aliases") or [])]
        facts, chunk_ids = pick_per_document([d for d in docs if d["facts"]], per_doc=1, limit=MAX_CHUNKS)
        have = {str(c) for c in chunk_ids}
        covered = {f.document_id for f in facts}
        for d in docs:  # papers with no extracted fact still count if their text names the entity
            if UUID(d["document_id"]) not in covered and len(chunk_ids) < MAX_CHUNKS:
                cid = await best_mention_chunk(conn, d["document_id"], names, question)
                if cid and str(cid) not in have:
                    chunk_ids.append(cid)
                    have.add(str(cid))
        return Structured("aggregate", [node["canonical_name"]], facts, chunk_ids,
                          {"entity": node["canonical_name"], "documents": docs})
    if intent == "path":
        if len(ents) < 2:
            return None
        a, b = ents[0]["node"], ents[1]["node"]
        if a["id"] == b["id"]:
            return None
        paths = await find_paths(conn, a["id"], b["id"])
        if not paths:
            return None
        best = paths[0]
        facts = [to_fact(s, hop=i + 1) for i, s in enumerate(best["steps"]) if s["chunk_id"]]
        chunk_ids = list(dict.fromkeys(f.chunk_id for f in facts))
        return Structured("path", [a["canonical_name"], b["canonical_name"]], facts, chunk_ids,
                          {"from": a["canonical_name"], "to": b["canonical_name"], "paths": paths})
    return None


def trim_detail(st: Structured, max_facts_per_doc: int = 6) -> dict:
    """The structured result as returned by the API: at most a few facts per paper, so responses stay small."""
    d = dict(st.detail)
    if "documents" in d:
        d["documents"] = [{**doc, "facts": doc["facts"][:max_facts_per_doc]} for doc in d["documents"]]
    return {"intent": st.intent, "entities": st.entities, **d}

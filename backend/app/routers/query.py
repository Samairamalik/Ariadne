import asyncio
import time
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app import db
from app import nli_worker
from app.config import settings
from app.embeddings import get_embedder
from app.generation import NO_ANSWER, Answer, generate_answer, generate_general
from app.retrieval.communities import best_overviews, build_communities, is_global_question
from app.retrieval.decompose import interleave, split_question
from app.retrieval.doc_aware import target_documents
from app.retrieval.graph import link_entities
from app.retrieval.graph_context import GraphContext, build_graph_context, evidence_trail, fetch_candidates
from app.retrieval.intent import classify_intent
from app.retrieval.structured import build_structured, trim_detail
from app.retrieval.hybrid import Candidate, hybrid_search
from app.retrieval.calibrate import load_thresholds
from app.retrieval.relevance import is_relevant
from app.retrieval.rerank import rerank

router = APIRouter(prefix="/api")


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[UUID] | None = None
    top_k: int = Field(6, ge=1, le=15)
    generate: bool = True
    rerank: bool = True  # False = RRF order only; lets you ablate the cross-encoder
    use_graph: bool = True  # False = plain hybrid RAG only; lets you compare with and without the graph
    verify_claims: bool = False  # check every cited sentence against its cited sources with an NLI model
    general_knowledge: bool = True  # if the documents don't cover the question, let the LLM answer (clearly marked)


def _dump(c: Candidate, n: int) -> dict:
    return {
        "n": n, "chunk_id": c.chunk_id, "document_id": c.document_id, "doc_title": c.doc_title,
        "section_path": c.section_path, "pages": [c.page_start, c.page_end], "text": c.text,
        "scores": {"cosine": round(c.cosine, 4), "rrf": round(c.rrf_score, 5), "rerank": c.rerank_score},
        "found_by": c.ranks,  # e.g. {"vector": 3, "lexical": 1} — why this passage was retrieved
    }


async def _retrieve_part(q: QueryIn, text: str, k: int):
    """Hybrid search + paper preference + rerank for one (sub-)question. Returns (top, reranked, targets, n_cands, vec)."""
    vec = await asyncio.to_thread(get_embedder().embed_query, text)
    targets: list[UUID] = []
    async with db.pool().acquire() as conn:
        cands = await hybrid_search(conn, text, vec, document_ids=q.document_ids)
        if not q.document_ids:  # an explicit document filter from the caller always wins
            targets = await target_documents(conn, text)
        if targets:  # the (sub-)question names a paper: make sure that paper's best chunks are in the rerank pool
            named = (await hybrid_search(conn, text, vec, document_ids=targets))[: settings.doc_prefer_extra_candidates]
            taken = {c.chunk_id for c in named}
            cands = named + [c for c in cands if c.chunk_id not in taken]
    if q.rerank:
        top, reranked = await asyncio.to_thread(rerank, text, cands, k, set(targets), settings.doc_prefer_bonus)
    else:
        top, reranked = cands[:k], False
    return top, reranked, targets, len(cands), vec


async def _retrieve(q: QueryIn) -> tuple[list[Candidate], dict]:
    t0 = time.perf_counter()
    parts = split_question(q.question)
    per_part = -(-q.top_k // len(parts))  # ceil: every part keeps room in the context
    results = [await _retrieve_part(q, part, per_part) for part in parts]
    top = interleave([r[0] for r in results], q.top_k, key=lambda c: c.chunk_id) if len(parts) > 1 else results[0][0]
    targets = list({t for r in results for t in r[2]})
    focus = {}
    async with db.pool().acquire() as conn:
        thresholds = await load_thresholds(conn)  # per-document relevance cutoffs (empty until calibrated)
        if targets:
            focus = {str(r["id"]): r["title"] for r in
                     await conn.fetch("SELECT id, title FROM documents WHERE id = ANY($1::uuid[])", targets)}
    return top, {"total_retrieval_ms": int((time.perf_counter() - t0) * 1e3), "candidates": sum(r[3] for r in results),
                 "reranked": all(r[1] for r in results), "targeted_docs": len(targets), "focus": focus, "thresholds": thresholds,
                 "parts": parts if len(parts) > 1 else [], "_vec": results[0][4] if len(parts) == 1 else None}


@router.post("/search")
async def search(q: QueryIn):
    """Retrieval only (no LLM): for evaluating chunking/retrieval quality in isolation."""
    top, timing = await _retrieve(q)
    timing.pop("_vec", None)
    return {"results": [_dump(c, i) for i, c in enumerate(top, 1)], "relevant": is_relevant(top, timing["reranked"], timing.get("thresholds")),
            "timing": timing}


@router.post("/graph/link")
async def graph_link(q: QueryIn):
    """Debug view of step 8: which graph nodes does this question link to, and which signal found each?"""
    top, timing = await _retrieve(q)
    vec = timing.pop("_vec", None)
    if not is_relevant(top, timing["reranked"], timing.get("thresholds")):
        return {"nodes": [], "relevant": False}
    if vec is None:
        vec = await asyncio.to_thread(get_embedder().embed_query, q.question)
    async with db.pool().acquire() as conn:
        nodes = await link_entities(conn, q.question, vec, [c.chunk_id for c in top])
    return {"relevant": True, "nodes": [{"id": n.node_id, "name": n.name, "type": n.node_type, "score": round(n.score, 5),
                       "found_by": n.found_by} for n in nodes]}


@router.post("/graph/communities/build")
async def communities_build():
    """Step 7: cluster the graph and write a summary per community (one LLM call each; run on demand)."""
    return await build_communities()


@router.get("/graph/communities")
async def communities_list():
    async with db.pool().acquire() as conn:
        rows = await conn.fetch("SELECT title, summary, size FROM communities ORDER BY size DESC")
    return [dict(r) for r in rows]


@router.post("/query")
async def query(q: QueryIn):
    t0 = time.perf_counter()
    intent, structured = "chunk", None
    vec = None
    top: list[Candidate] = []
    timing: dict = {}

    # Fast path for structured graph intents (profile / path / aggregate): check graph first (~25ms)
    if q.use_graph and not q.document_ids and classify_intent(q.question) != "chunk":
        t_ret = time.perf_counter()
        vec = await asyncio.to_thread(get_embedder().embed_query, q.question)
        async with db.pool().acquire() as conn:
            structured = await build_structured(conn, classify_intent(q.question), q.question)
            if structured is not None:
                sources = await fetch_candidates(conn, structured.chunk_ids, vec, {c: {"graph": 1} for c in structured.chunk_ids})
                structured = structured if sources else None
        if structured is not None:
            intent, relevant = structured.intent, True
            top = sources
            gctx = GraphContext(facts=structured.facts, hops=max((f.hop for f in structured.facts), default=1))
            timing = {"total_retrieval_ms": int((time.perf_counter() - t_ret) * 1e3), "candidates": len(sources),
                      "reranked": False, "targeted_docs": 0, "focus": {}, "thresholds": {}, "parts": []}

    if structured is None:
        top, timing = await _retrieve(q)
        vec = timing.pop("_vec", None)
        relevant = is_relevant(top, timing["reranked"], timing.get("thresholds"))
        focus = timing.get("focus", {})
        top = [c for c in top if str(c.document_id) in focus] + [c for c in top if str(c.document_id) not in focus]
        gctx = GraphContext()
    else:
        focus = {}

    answer, llm_error = None, None
    overview = []
    if q.use_graph and is_global_question(q.question):
        if vec is None:
            vec = await asyncio.to_thread(get_embedder().embed_query, q.question)
        async with db.pool().acquire() as conn:
            overview = [(r["title"], r["summary"]) for r in await best_overviews(conn, vec)]
        relevant = relevant or bool(overview)

    if structured is not None:
        pass
    elif not relevant:
        top = []  # nearest chunks to an unrelated question are noise: show none, cite none
    elif q.use_graph:
        if vec is None:
            vec = await asyncio.to_thread(get_embedder().embed_query, q.question)
        async with db.pool().acquire() as conn:
            gctx = await build_graph_context(conn, q.question, vec, top)
        top = top + gctx.extra

    used_general = False
    if q.generate:
        try:
            if relevant:
                answer = await generate_answer(q.question, top, gctx.facts, overview, list(focus.values()),
                                              timing.get("parts"), intent if structured is not None else None)
                if answer.refused and structured is not None:
                    # Structured graph fast-path lacked the specific passage; fall back to full hybrid retrieval
                    structured, intent = None, "chunk"
                    top, timing = await _retrieve(q)
                    vec = timing.pop("_vec", None)
                    relevant = is_relevant(top, timing["reranked"], timing.get("thresholds"))
                    focus = timing.get("focus", {})
                    top = [c for c in top if str(c.document_id) in focus] + [c for c in top if str(c.document_id) not in focus]
                    gctx = GraphContext()
                    if relevant and q.use_graph:
                        if vec is None:
                            vec = await asyncio.to_thread(get_embedder().embed_query, q.question)
                        async with db.pool().acquire() as conn:
                            gctx = await build_graph_context(conn, q.question, vec, top)
                        top = top + gctx.extra
                    if relevant:
                        answer = await generate_answer(q.question, top, gctx.facts, overview, list(focus.values()),
                                                      timing.get("parts"), None)
                if answer.refused and q.general_knowledge:
                    answer = await generate_general(q.question)
                    used_general = True
            elif q.general_knowledge:
                answer = await generate_general(q.question)
                used_general = True
            else:
                answer = Answer(NO_ANSWER, [], [], [], True)
        except Exception as e:  # noqa: BLE001 — never lose the retrieved passages because the LLM is down
            llm_error = f"{type(e).__name__}: {e}"[:400]

    verification: dict = {"enabled": False}
    if q.verify_claims and relevant and not used_general and answer is not None and not answer.refused and top:
        verification = await nli_worker.verify(answer.text, [c.text for c in top])

    total_ms = int((time.perf_counter() - t0) * 1e3)
    async with db.pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO queries (query_text, path_taken, nodes_visited, chunk_ids, hops, answer, latency_ms) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7)",
            q.question, "graph" if structured is not None else ("hybrid" if gctx.extra or gctx.facts else "chunk"), [n.node_id for n in gctx.seeds],
            [c.chunk_id for c in top], gctx.hops, answer.text if answer else None, total_ms)
    return {
        "answer": answer.text if answer else None,
        "llm_error": llm_error,  # set when generation failed; sources are still valid
        "llm_provider": answer.provider if answer else None,  # who saw your passages: ollama = local, else cloud
        "llm_model": answer.model if answer else None,
        "refused": answer.refused if answer else None,
        "relevant": relevant,
        "source": "general_knowledge" if used_general else ("documents" if relevant else "none"),
        "cited": answer.cited if answer else [],
        "invalid_citations": answer.invalid_citations if answer else [],
        "uncited_sentences": answer.uncited_sentences if answer else [],
        "verification": verification,
        "sources": [_dump(c, i) for i, c in enumerate(top, 1)],
        "path_taken": "graph" if structured is not None else ("hybrid" if gctx.extra or gctx.facts else "chunk"),
        "intent": intent,
        "evidence_trail": evidence_trail(gctx.facts, {c.chunk_id: i for i, c in enumerate(top, 1)}),
        "structured": trim_detail(structured) if structured is not None else None,
        "question_parts": timing.get("parts", []),
        "overview_used": [t for t, _ in overview],
        "graph": {"seeds": [n.name for n in gctx.seeds], "hops": gctx.hops, "extra_chunks": len(gctx.extra),
                  "extras": [{"doc": c.doc_title[:40], "section": c.section_path[:50],
                              "rerank": None if c.rerank_score is None else round(c.rerank_score, 2)}
                             for c in gctx.extra],
                  "facts": [{"subject": f.subject, "predicate": f.predicate, "object": f.object, "hop": f.hop}
                            for f in gctx.facts[:40]]},
        "timing": {**timing, "total_ms": total_ms},
    }

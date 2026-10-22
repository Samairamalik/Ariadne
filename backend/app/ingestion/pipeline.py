"""Ingestion orchestration: parse -> chunk -> embed -> store, with persisted progress.

DEVIATION from the design: no Celery/Redis. Single-user local app, Redis on Windows is awkward, and the heavy
steps are CPU-bound work pushed to a thread. Job state lives in `ingestion_jobs`, so progress survives restarts and
the SSE endpoint just reads the row. Embedding is serialised with a semaphore so concurrent uploads don't
oversubscribe a laptop CPU.
"""
import asyncio
import json
import logging
from pathlib import Path
from uuid import UUID

from app import db
from app.config import settings
from app.embeddings import get_embedder
from app.ingestion.chunker import chunk_document
from app.ingestion.resolve import resolve_duplicates
from app.retrieval.calibrate import calibrate_document
from app.retrieval.communities import build_communities
from app.ingestion.extraction import BATCH_SIZE, LLMError, LLMUnavailable, extract_batch
from app.llm.router import get_router
from app.ingestion.pdf_parser import parse_pdf, parse_text
from app.ingestion.tokens import get_token_counter

log = logging.getLogger("ariadne.ingest")
_gate = asyncio.Semaphore(1)
EMBED_BATCH = 32
MAX_CONSECUTIVE_FAILURES = 3


class IngestError(Exception):
    pass


async def _stage(doc_id: UUID, job_id: UUID, stage: str, status: str | None = None, **counters) -> None:
    sets, args = ["stage = $2"], [job_id, stage]
    for k, v in counters.items():
        args.append(v)
        sets.append(f"{k} = ${len(args)}")
    async with db.pool().acquire() as conn:
        await conn.execute(f"UPDATE ingestion_jobs SET {', '.join(sets)} WHERE id = $1", *args)
        if status:
            await conn.execute("UPDATE documents SET status = $2 WHERE id = $1", doc_id, status)


async def run_ingestion(doc_id: UUID, job_id: UUID, path: Path, doc_type: str) -> None:
    try:
        async with _gate:
            chunks = await _run(doc_id, job_id, path, doc_type)
    except Exception as e:  # noqa: BLE001 — surface any failure on the document row
        log.exception("ingestion failed for %s", doc_id)
        msg = str(e) if isinstance(e, IngestError) else f"{type(e).__name__}: {e}"
        async with db.pool().acquire() as conn:
            await conn.execute("UPDATE documents SET status = 'failed', error = $2 WHERE id = $1", doc_id, msg)
            await conn.execute("UPDATE ingestion_jobs SET stage = 'failed', finished_at = now() WHERE id = $1", job_id)
        return
    await _build_graph(doc_id, job_id, chunks)


async def recover_interrupted() -> int:
    """At startup, close graph jobs a restart left on 'extracting'; their documents are already searchable."""
    async with db.pool().acquire() as conn:
        rows = await conn.fetch(
            "UPDATE ingestion_jobs SET stage = 'done', finished_at = now() "
            "WHERE stage = 'extracting' AND finished_at IS NULL RETURNING document_id")
        await conn.execute(
            "UPDATE documents SET metadata = jsonb_set(metadata, '{graph_warning}', "
            "'\"graph extraction was interrupted by a restart; delete and re-upload to rebuild\"') WHERE id = ANY($1::uuid[])",
            [r["document_id"] for r in rows])
    return len(rows)


async def _build_graph(doc_id: UUID, job_id: UUID, chunks: list[tuple[UUID, str]],
                       rebuild_communities: bool = True) -> None:
    """Runs after the document is already searchable. A failure here is recorded, never fatal."""
    warning = None
    try:
        if settings.calibrate_on_ingest:  # first: it gives this document its own relevance threshold
            await _stage(doc_id, job_id, "calibrating")
            await calibrate_document(doc_id)
        warning = await _extract_graph(doc_id, job_id, chunks)
        if settings.resolve_on_ingest:
            await _stage(doc_id, job_id, "resolving")
            await resolve_duplicates()
        if rebuild_communities and settings.community_rebuild_on_ingest:
            await _stage(doc_id, job_id, "summarizing")  # community summaries describe the whole graph: now stale
            await build_communities()
    except Exception as e:  # noqa: BLE001
        log.exception("graph extraction failed for %s", doc_id)
        warning = f"{type(e).__name__}: {e}"[:500]
    async with db.pool().acquire() as conn:
        if warning:
            await conn.execute(
                "UPDATE documents SET metadata = jsonb_set(metadata, '{graph_warning}', to_jsonb($2::text)) WHERE id = $1",
                doc_id, warning)
        await conn.execute("UPDATE ingestion_jobs SET stage = 'done', finished_at = now() WHERE id = $1", job_id)


async def _run(doc_id: UUID, job_id: UUID, path: Path, doc_type: str) -> list[tuple[UUID, str]]:
    await _stage(doc_id, job_id, "chunking", "chunking")

    def parse_and_chunk():
        parsed = parse_pdf(path) if doc_type == "pdf" else parse_text(path, markdown=doc_type == "markdown")
        if parsed.needs_ocr:
            raise IngestError(parsed.warnings[0])
        chunks, stats = chunk_document(
            parsed, get_token_counter(),
            target=settings.chunk_target_tokens, max_tokens=settings.chunk_max_tokens,
            min_tokens=settings.chunk_min_tokens, overlap_sentences=settings.chunk_overlap_sentences,
            embed_max_tokens=settings.embed_max_tokens,
        )
        return parsed, chunks, stats

    parsed, chunks, stats = await asyncio.to_thread(parse_and_chunk)
    if not chunks:
        raise IngestError("No indexable text found in document.")

    async with db.pool().acquire() as conn:
        await conn.execute(
            "UPDATE documents SET title = $2, author = $3, page_count = $4, token_count = $5, metadata = $6::jsonb WHERE id = $1",
            doc_id, parsed.title, parsed.author, parsed.page_count, sum(c.token_count for c in chunks),
            json.dumps({"chunking": stats.summary(), "warnings": parsed.warnings}),
        )
    await _stage(doc_id, job_id, "embedding", "embedding", chunks_total=len(chunks), chunks_done=0)

    embedder = get_embedder()
    vectors = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = [c.embed_text for c in chunks[i : i + EMBED_BATCH]]
        vectors.extend(await asyncio.to_thread(embedder.embed_passages, batch))
        await _stage(doc_id, job_id, "embedding", chunks_done=min(i + EMBED_BATCH, len(chunks)))

    async with db.pool().acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM chunks WHERE document_id = $1", doc_id)
            await conn.executemany(
                """INSERT INTO chunks (document_id, chunk_index, kind, section_path, page_start, page_end, text, token_count, embedding)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
                [(doc_id, c.index, c.kind, c.section_path, c.page_start, c.page_end, c.text, c.token_count, v)
                 for c, v in zip(chunks, vectors)],
            )
        await conn.execute(  # chunk length in lexemes (repeats counted): BM25's length normalisation needs it
            "UPDATE chunks SET doc_len = GREATEST(1, (SELECT coalesce(sum(coalesce(array_length(u.positions, 1), 1)), 1) "
            "FROM unnest(chunks.tsv) u)) WHERE document_id = $1", doc_id)
        rows = await conn.fetch("SELECT id, text FROM chunks WHERE document_id = $1 ORDER BY chunk_index", doc_id)

    async with db.pool().acquire() as conn:
        await conn.execute("UPDATE documents SET status = 'done', error = NULL WHERE id = $1", doc_id)
        await conn.execute("UPDATE ingestion_jobs SET stage = 'extracting' WHERE id = $1", job_id)
    return [(r["id"], r["text"]) for r in rows]


async def embed_nodes(names_by_id: dict[UUID, str]) -> None:
    """Store a vector for each node name so a question can be matched to nodes by meaning."""
    if not names_by_id:
        return
    ids = list(names_by_id)
    vecs = await asyncio.to_thread(get_embedder().embed_passages, [names_by_id[i] for i in ids])
    async with db.pool().acquire() as conn:
        await conn.executemany("UPDATE nodes SET embedding = $2 WHERE id = $1", list(zip(ids, vecs)))


async def _extract_triples(doc_id: UUID, job_id: UUID, chunks: list[tuple[UUID, str]]):
    """LLM phase only: returns ([(chunk_id, Triple)], warning). Writes NOTHING to the graph, so a failure here can
    never damage an existing graph."""
    await _stage(doc_id, job_id, "extracting", chunks_done=0)
    router = get_router()
    async with db.pool().acquire() as conn:  # paper-aware extraction: the title and each chunk's section heading
        doc_title = await conn.fetchval("SELECT title FROM documents WHERE id = $1", doc_id)
        section_of = {r["id"]: r["section_path"] for r in
                      await conn.fetch("SELECT id, section_path FROM chunks WHERE document_id = $1", doc_id)}
    found: list = []
    warning = None
    consecutive_failures, failed_batches = 0, []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            triples = await extract_batch(router, [text for _, text in batch], doc_title,
                                          [section_of.get(cid) for cid, _ in batch])
            consecutive_failures = 0
        except (LLMError, LLMUnavailable) as e:
            # One bad batch (e.g. a 3B model stuck repeating tokens) must not cost the rest of the document;
            # only a run of failures means the LLM is really down.
            consecutive_failures += 1
            failed_batches.append(i)
            warning = f"triple extraction failed for batches starting at chunk {failed_batches}: {e}"[:500]
            log.warning("%s (document %s)", warning, doc_id)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                warning = f"triple extraction stopped at chunk {i} after {consecutive_failures} failures in a row: {e}"[:500]
                break
            continue
        found += [(batch[tr.chunk][0], tr) for tr in triples]
        await _stage(doc_id, job_id, "extracting", chunks_done=min(i + BATCH_SIZE, len(chunks)), triples_found=len(found))
    return found, warning


async def _store_triples(doc_id: UUID, job_id: UUID, found: list, replace: bool = False) -> dict:
    """Write triples as nodes (exact name+type) and edges that point back at their source chunk, in ONE transaction.

    replace=True first removes this document's old edges inside the same transaction, so the swap is all-or-nothing.
    Node-name embeddings are computed BEFORE the transaction: if the embedder cannot load, nothing has changed."""
    names = sorted({t.subject for _, t in found} | {t.object for _, t in found})
    vecs = await asyncio.to_thread(get_embedder().embed_passages, names) if names else []
    vec_of = dict(zip(names, vecs))
    touched: set[UUID] = set()
    nodes_created = edges_created = 0
    async with db.pool().acquire() as conn:
        async with conn.transaction():
            if replace:
                await conn.execute("DELETE FROM edges WHERE document_id = $1", doc_id)
            for chunk_id, tr in found:
                ids = []
                for name, ntype in ((tr.subject, tr.subject_type), (tr.object, tr.object_type)):
                    row = await conn.fetchrow(
                        """INSERT INTO nodes (canonical_name, node_type, first_seen_doc, embedding) VALUES ($1, $2, $3, $4)
                           ON CONFLICT (canonical_name, node_type) DO UPDATE
                               SET canonical_name = EXCLUDED.canonical_name,
                                   embedding = coalesce(nodes.embedding, EXCLUDED.embedding)
                           RETURNING id, (xmax = 0) AS inserted""", name, ntype, doc_id, vec_of[name])
                    nodes_created += row["inserted"]
                    ids.append(row["id"])
                    touched.add(row["id"])
                edge = await conn.fetchrow(
                    """INSERT INTO edges (source_id, target_id, predicate, evidence_text, chunk_id, document_id)
                       VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING RETURNING id""",
                    ids[0], ids[1], tr.predicate, tr.evidence, chunk_id, doc_id)
                edges_created += edge is not None
            await conn.execute(  # nodes nothing points to any more (replaced or re-ingested documents)
                "DELETE FROM nodes n WHERE NOT EXISTS (SELECT 1 FROM edges e WHERE e.source_id = n.id OR e.target_id = n.id)")
            if touched:  # counted from the edges, so re-ingesting a document can not inflate it
                await conn.execute(
                    """UPDATE nodes n SET mention_count = GREATEST(1, (SELECT count(*) FROM edges e
                                                                      WHERE e.source_id = n.id OR e.target_id = n.id))
                       WHERE n.id = ANY($1::uuid[])""", list(touched))
    await _stage(doc_id, job_id, "extracting", triples_found=len(found), nodes_created=nodes_created, edges_created=edges_created)
    return {"triples": len(found), "nodes_created": nodes_created, "edges_created": edges_created}


async def _extract_graph(doc_id: UUID, job_id: UUID, chunks: list[tuple[UUID, str]]) -> str | None:
    """First-time graph for a document. The graph is an enrichment: if the LLM is down the document is still
    searchable, so a failure here returns a warning instead of failing the ingestion."""
    found, warning = await _extract_triples(doc_id, job_id, chunks)
    if found:
        await _store_triples(doc_id, job_id, found, replace=False)
    return warning


async def reextract_document(doc_id: UUID) -> dict:
    """Rebuild one document graph from its stored chunks (no re-parsing) after the extraction logic improves.

    SAFE BY CONSTRUCTION: the old graph is left untouched unless new triples were actually produced, and then it is
    swapped in one transaction. (An earlier version deleted first and extracted second; when PyTorch could not load it
    wiped all six graphs.) Returns what happened; never raises for an ordinary failure."""
    async with db.pool().acquire() as conn:
        rows = await conn.fetch("SELECT id, text FROM chunks WHERE document_id = $1 ORDER BY chunk_index", doc_id)
        old_edges = await conn.fetchval("SELECT count(*) FROM edges WHERE document_id = $1", doc_id)
        job_id = await conn.fetchval(
            "INSERT INTO ingestion_jobs (document_id, stage, chunks_total) VALUES ($1, 'extracting', $2) RETURNING id",
            doc_id, len(rows))
    chunks = [(r["id"], r["text"]) for r in rows]
    try:
        found, warning = await _extract_triples(doc_id, job_id, chunks)
        if not found:
            result = {"ok": False, "kept_old_graph": True, "old_edges": old_edges,
                      "reason": warning or "the model produced no triples"}
        else:
            stats = await _store_triples(doc_id, job_id, found, replace=True)
            result = {"ok": True, "old_edges": old_edges, "warning": warning, **stats}
            async with db.pool().acquire() as conn:
                if warning:
                    await conn.execute(
                        "UPDATE documents SET metadata = jsonb_set(metadata, '{graph_warning}', to_jsonb($2::text)) WHERE id = $1",
                        doc_id, warning)
                else:
                    await conn.execute("UPDATE documents SET metadata = metadata - 'graph_warning' WHERE id = $1", doc_id)
    except Exception as e:  # noqa: BLE001 - embedder or database trouble: the old graph is still in place
        log.exception("re-extraction failed for %s; old graph kept", doc_id)
        result = {"ok": False, "kept_old_graph": True, "old_edges": old_edges, "reason": f"{type(e).__name__}: {e}"[:300]}
    async with db.pool().acquire() as conn:
        await conn.execute("UPDATE ingestion_jobs SET stage = 'done', finished_at = now() WHERE id = $1", job_id)
    return result

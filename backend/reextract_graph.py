"""Rebuild the knowledge graph of every document (or one) from its stored chunks, then merge entities and rebuild the
community summaries. Use after improving the extraction logic.

    python reextract_graph.py                    # all documents
    python reextract_graph.py ROBERTA.pdf        # one document by filename

Safe by construction: a pre-flight check proves the embedder and the language model work BEFORE anything is touched,
and each document's old graph is replaced in one transaction only if new triples were produced. A document that fails
keeps its old graph.
"""
import asyncio
import sys
import time

from app import db
from app.embeddings import get_embedder
from app.ingestion.pipeline import reextract_document
from app.ingestion.resolve import resolve_duplicates
from app.llm import get_router
from app.retrieval.communities import build_communities


async def preflight() -> None:
    try:
        vec = await asyncio.to_thread(get_embedder().embed_passages, ["pre-flight check"])
        assert len(vec[0]) > 0
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"pre-flight FAILED (embedder): {type(e).__name__}: {str(e)[:200]}\nNothing was changed.")
    try:
        await get_router().chat([{"role": "user", "content": "Reply with the single word: ready"}], timeout=180)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"pre-flight FAILED (language model): {type(e).__name__}: {str(e)[:200]}\nNothing was changed.")
    print("pre-flight ok: embedder and language model both work", flush=True)


async def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    await db.init_pool()
    await preflight()
    async with db.pool().acquire() as conn:
        docs = await conn.fetch("SELECT id, filename FROM documents WHERE status = 'done' ORDER BY ingested_at")
    ok = 0
    for d in docs:
        if only and d["filename"] != only:
            continue
        t0 = time.time()
        r = await reextract_document(d["id"])
        ok += bool(r["ok"])
        print(f"{d['filename']:26s} {'OK  ' if r['ok'] else 'KEPT'} {r}  ({time.time() - t0:.0f}s)", flush=True)
    if ok:
        pairs = await resolve_duplicates()
        print(f"entity merging: {sum(p.merged for p in pairs)} of {len(pairs)} candidate pairs merged", flush=True)
        print("communities:", await build_communities(), flush=True)
    else:
        print("no document was rebuilt; entity merging and communities were left alone", flush=True)
    await db.close_pool()


asyncio.run(main())

"""Measure every document's own relevance threshold from synthetic questions, then show what was measured.

    python calibrate_relevance.py                 # all documents
    python calibrate_relevance.py ROBERTA.pdf     # one document by filename
"""
import asyncio
import sys
import time

from app import db
from app.retrieval import rerank
from app.retrieval.calibrate import calibrate_document


async def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    await db.init_pool()
    rerank.start_loading()
    while rerank.status() != "ready":
        if rerank.status() == "failed":
            raise SystemExit("reranker failed to load")
        time.sleep(1)
    async with db.pool().acquire() as conn:
        docs = await conn.fetch("SELECT id, filename FROM documents WHERE status = 'done' ORDER BY ingested_at")
    for d in docs:
        if only and d["filename"] != only:
            continue
        r = await calibrate_document(d["id"])
        print(f"{d['filename']:22s} threshold={r['threshold']}  (raw p5={r['raw_p5']}, min={r['min']}, median={r['median']}, "
              f"max={r['max']}, {r['scored']}/{r['questions']} questions scored)", flush=True)
        print("   e.g.", *r["sample"][:2], sep="\n     ")
    await db.close_pool()


asyncio.run(main())

"""One-off: embed nodes created before node embeddings existed.   python backfill_node_embeddings.py"""
import asyncio

from app import db
from app.ingestion.pipeline import embed_nodes


async def main() -> None:
    await db.init_pool()
    async with db.pool().acquire() as conn:
        rows = await conn.fetch("SELECT id, canonical_name FROM nodes WHERE embedding IS NULL")
    await embed_nodes({r["id"]: r["canonical_name"] for r in rows})
    print(f"embedded {len(rows)} nodes")
    await db.close_pool()


asyncio.run(main())

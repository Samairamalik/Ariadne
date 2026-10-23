"""Remove filler nodes (and the facts that touch them) from the stored graph, keeping only main topics and entities.

    python prune_graph.py            # dry run: shows what would go, changes nothing
    python prune_graph.py --strict   # also drops one-off lowercase phrases (much smaller graph)
    python prune_graph.py --apply    # deletes, then recounts mentions; rebuild communities afterwards

Rules live in app/ingestion/entity_filter.py (`is_core_node`). Facts are only removed when one of their two nodes
fails the rule, in a single transaction, so an interrupted run changes nothing.
"""
import asyncio
import sys

from app import db
from app.ingestion.entity_filter import is_core_node, is_main_entity


async def main(apply: bool, strict: bool) -> None:
    await db.init_pool()
    async with db.pool().acquire() as conn:
        nodes = await conn.fetch("SELECT id, canonical_name, node_type, mention_count FROM nodes")
        keep = (lambda n: is_core_node(n["canonical_name"], n["node_type"], n["mention_count"])) if strict             else (lambda n: is_main_entity(n["canonical_name"], n["node_type"]))
        drop = [n for n in nodes if not keep(n)]
        drop_ids = [n["id"] for n in drop]
        edges = await conn.fetchval("SELECT count(*) FROM edges")
        hit = await conn.fetchval("SELECT count(*) FROM edges WHERE source_id = ANY($1::uuid[]) OR target_id = ANY($1::uuid[])", drop_ids)
        print(f"nodes: {len(nodes)} -> {len(nodes) - len(drop)} kept ({len(drop)} filler)")
        print(f"edges touching a filler node: {hit} of {edges}")
        print("examples dropped:", "; ".join(n["canonical_name"] for n in drop[:40]))
        if not apply:
            print("\ndry run only - rerun with --apply to delete")
            return
        async with conn.transaction():
            await conn.execute("DELETE FROM edges WHERE source_id = ANY($1::uuid[]) OR target_id = ANY($1::uuid[])", drop_ids)
            await conn.execute("DELETE FROM nodes WHERE id = ANY($1::uuid[])", drop_ids)
            await conn.execute(
                "DELETE FROM nodes n WHERE NOT EXISTS (SELECT 1 FROM edges e WHERE n.id IN (e.source_id, e.target_id))")
            await conn.execute(
                "UPDATE nodes n SET mention_count = (SELECT count(*) FROM edges e WHERE n.id IN (e.source_id, e.target_id))")
        left_n = await conn.fetchval("SELECT count(*) FROM nodes")
        left_e = await conn.fetchval("SELECT count(*) FROM edges")
        print(f"\ndone: {left_n} nodes, {left_e} edges remain")
    await db.close_pool()


if __name__ == "__main__":
    asyncio.run(main("--apply" in sys.argv, "--strict" in sys.argv))

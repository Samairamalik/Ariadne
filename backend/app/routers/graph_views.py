"""Structured graph views over HTTP: entity profile, path between two entities, which papers mention an entity.

These read the graph directly (no language model), so they are fast and every fact comes with its quote and paper.
"""
from fastapi import APIRouter, HTTPException, Query

from app import db
from app.retrieval.entity_views import aggregate_documents, entity_profile, find_nodes, find_paths

router = APIRouter(prefix="/api/graph", tags=["graph views"])


async def _one(conn, name: str) -> dict:
    nodes = await find_nodes(conn, name, limit=1)
    if not nodes:
        raise HTTPException(404, f"No entity named '{name}' in the graph (try /api/graph/search?name=...).")
    return nodes[0]


@router.get("/search")
async def search_entities(name: str = Query(min_length=2), limit: int = Query(8, ge=1, le=25)):
    """Entities whose name or alias is close to `name`, best first, with how many papers mention each."""
    async with db.pool().acquire() as conn:
        nodes = await find_nodes(conn, name, limit)
    return [{"id": str(n["id"]), "name": n["canonical_name"], "type": n["node_type"], "papers": n["documents"],
             "mentions": n["mention_count"], "match": n["match"]} for n in nodes]


@router.get("/entity")
async def entity(name: str = Query(min_length=2)):
    """Everything the library says about one entity, grouped by paper, every fact with its quote."""
    async with db.pool().acquire() as conn:
        node = await _one(conn, name)
        return await entity_profile(conn, node["id"])


@router.get("/path")
async def path(a: str = Query(min_length=2), b: str = Query(min_length=2), max_hops: int = Query(3, ge=1, le=4)):
    """The shortest chains of facts between two entities. `paths` is empty when they are not connected that closely."""
    async with db.pool().acquire() as conn:
        na, nb = await _one(conn, a), await _one(conn, b)
        paths = await find_paths(conn, na["id"], nb["id"], max_hops=max_hops)
    return {"from": na["canonical_name"], "to": nb["canonical_name"], "max_hops": max_hops, "paths": paths,
            "connected": bool(paths)}


@router.get("/aggregate")
async def aggregate(entity: str = Query(min_length=2), predicate: str | None = None):
    """Which papers say something about an entity (optionally only facts whose predicate contains `predicate`)."""
    async with db.pool().acquire() as conn:
        node = await _one(conn, entity)
        docs = await aggregate_documents(conn, node["id"], predicate)
    return {"entity": node["canonical_name"], "predicate": predicate, "papers": len(docs), "documents": docs}

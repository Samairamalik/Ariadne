"""Hybrid chunk retrieval: pgvector HNSW (semantic) + BM25 over Postgres full-text (lexical), fused with RRF.

The lexical side is real BM25 (IDF, term-frequency saturation, length normalisation), computed in bm25.py on top of
the GIN-indexed tsvector; it replaced ts_rank_cd, which has none of those three.
"""
from dataclasses import dataclass, field
from uuid import UUID

import asyncpg
import numpy as np

from app.config import settings
from app.retrieval.bm25 import bm25_search
from app.retrieval.fusion import rrf


@dataclass
class Candidate:
    chunk_id: UUID
    document_id: UUID
    doc_title: str
    filename: str
    section_path: str
    page_start: int
    page_end: int
    chunk_index: int
    text: str
    cosine: float
    rrf_score: float
    ranks: dict[str, int] = field(default_factory=dict)
    rerank_score: float | None = None


async def hybrid_search(
    conn: asyncpg.Connection,
    question: str,
    query_vec: np.ndarray,
    *,
    document_ids: list[UUID] | None = None,
    vector_k: int | None = None,
    lexical_k: int | None = None,
) -> list[Candidate]:
    vector_k = vector_k or settings.retrieve_vector_k
    lexical_k = lexical_k or settings.retrieve_lexical_k
    doc_filter = "WHERE document_id = ANY($3::uuid[])" if document_ids else ""
    doc_args = [document_ids] if document_ids else []

    async with conn.transaction():
        await conn.execute(f"SET LOCAL hnsw.ef_search = {max(100, vector_k * 2)}")
        if document_ids:
            # Filtered HNSW is post-filtered and can return < k rows; iterative scan keeps scanning until k.
            try:
                async with conn.transaction():  # savepoint: older pgvector lacks this GUC
                    await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
            except asyncpg.PostgresError:
                pass
        vec_rows = await conn.fetch(
            f"SELECT id FROM chunks {doc_filter} ORDER BY embedding <=> $1 LIMIT $2",
            query_vec, vector_k, *doc_args,
        )

    lex_ids = await bm25_search(conn, question, lexical_k, document_ids)  # true BM25 (see bm25.py)

    fused = rrf(
        {"vector": [str(r["id"]) for r in vec_rows], "lexical": [str(i) for i in lex_ids]},
        k=settings.rrf_k,
    )
    if not fused:
        return []

    rows = await conn.fetch(
        """SELECT c.id, c.document_id, coalesce(d.title, d.filename) AS doc_title, d.filename, c.section_path,
                  c.page_start, c.page_end, c.chunk_index, c.text, 1 - (c.embedding <=> $1) AS cosine
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE c.id = ANY($2::uuid[])""",
        query_vec, [UUID(f.id) for f in fused],
    )
    by_id = {str(r["id"]): r for r in rows}
    out = []
    for f in fused:
        r = by_id[f.id]
        out.append(Candidate(
            r["id"], r["document_id"], r["doc_title"], r["filename"], r["section_path"] or "",
            r["page_start"], r["page_end"], r["chunk_index"], r["text"], float(r["cosine"]), f.score, f.ranks,
        ))
    return out

"""True BM25 lexical ranking on top of Postgres full-text search.

Postgres's built-in ts_rank_cd is not BM25: it rewards proximity but has no inverse document frequency, no term
frequency saturation and no length normalisation. BM25 adds all three:

    score(d, q) = sum over query terms t of  IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avglen))
    IDF(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))

  * tf saturates: 10 mentions are not 5x better than 2 (k1 sets how fast).
  * IDF: a word in 3 chunks ("RoBERTa") outweighs a word in half of them ("model").
  * length: a long chunk contains more query words by chance; b scales that away.

Postgres supplies what it is good at: the GIN index narrows to chunks containing any query term, and `unnest(tsv)` gives
each term's frequency. The arithmetic is plain Python so it can be unit-tested without a database.
"""
import math
from collections import defaultdict
from uuid import UUID

import asyncpg

from app.retrieval.fusion import to_tsquery_or

K1 = 1.2
B = 0.75


def idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def bm25_scores(rows: list[tuple], df: dict[str, int], n_docs: int, avg_len: float,
                k1: float = K1, b: float = B) -> dict:
    """rows = (chunk_id, chunk_len, term, tf). Returns {chunk_id: score} for every chunk with at least one term."""
    out: dict = defaultdict(float)
    avg_len = max(avg_len, 1.0)
    for chunk_id, chunk_len, term, tf in rows:
        if term not in df or df[term] <= 0:
            continue
        norm = tf + k1 * (1.0 - b + b * (chunk_len or 1) / avg_len)
        out[chunk_id] += idf(n_docs, df[term]) * tf * (k1 + 1.0) / norm
    return dict(out)


def rank(scores: dict, limit: int) -> list:
    """Best first; ties broken by id so results are deterministic."""
    return sorted(scores, key=lambda c: (-scores[c], str(c)))[:limit]


async def bm25_search(conn: asyncpg.Connection, question: str, limit: int,
                      document_ids: list[UUID] | None = None) -> list[UUID]:
    """Chunk ids ranked by BM25 against the question; empty if it has no searchable words."""
    ts = to_tsquery_or(question)
    if not ts:
        return []
    terms = [r["lexeme"] for r in await conn.fetch(
        "SELECT DISTINCT lexeme FROM unnest(to_tsvector('english', $1))", question)]
    if not terms:
        return []
    stats = await conn.fetchrow("SELECT count(*) AS n, coalesce(avg(doc_len), 1) AS avg_len FROM chunks")
    df_rows = await conn.fetch(
        "SELECT t AS term, (SELECT count(*) FROM chunks WHERE tsv @@ quote_literal(t)::tsquery) AS df "
        "FROM unnest($1::text[]) AS t", terms)
    df = {r["term"]: r["df"] for r in df_rows}
    doc_filter = "AND c.document_id = ANY($3::uuid[])" if document_ids else ""
    args = [terms, ts] + ([document_ids] if document_ids else [])
    rows = await conn.fetch(
        f"""SELECT c.id, c.doc_len, u.lexeme AS term, coalesce(array_length(u.positions, 1), 1) AS tf
            FROM chunks c, unnest(c.tsv) AS u
            WHERE c.tsv @@ to_tsquery('english', $2) AND u.lexeme = ANY($1::text[]) {doc_filter}""", *args)
    scores = bm25_scores([(r["id"], r["doc_len"], r["term"], r["tf"]) for r in rows], df, stats["n"], float(stats["avg_len"]))
    return rank(scores, limit)

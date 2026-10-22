"""Per-document relevance calibration.

The relevance gate used one fixed cutoff for every document, found by hand from ~20 questions. Documents differ: a
dense maths paper and a plain-prose survey give the cross-encoder different score ranges. So each document gets its
own threshold, measured when it is ingested:

  1. Pick ~20 chunks spread across the document.
  2. Ask the language model to write one question each chunk answers (the questions a real reader of this document
     would plausibly ask).
  3. Run each question through the real retrieval pipeline (hybrid search, then rerank) and keep the best chunk's
     score. These are scores of questions the document DOES answer.
  4. threshold = the 5th percentile of those scores: the level that 95% of known-answerable questions reach.
  5. Clamp it to [relevance_min_threshold, relevance_max_threshold] (see config for why).

At question time the gate compares the best chunk's score with the threshold of the document that chunk came from.

Known weakness, stated in the open: the synthetic questions are answered by one passage, so they are easier than
comparison or multi-paper questions. The 5th percentile of EASY questions can sit above what a hard but legitimate
question scores. The clamp's upper bound exists for exactly that reason.
"""
import asyncio
import json
import logging
from uuid import UUID

import numpy as np

from app import db
from app.config import settings
from app.embeddings import get_embedder
from app.llm import get_router
from app.retrieval.hybrid import hybrid_search
from app.retrieval.rerank import rerank

log = logging.getLogger("ariadne.calibrate")

BATCH = 4
MIN_WORDS = 4
BANNED = ("passage", "this text", "the text", "excerpt", "the above", "this paragraph", "this section")

SYSTEM = (
    "You write test questions for a document search system. You receive numbered passages from one document. For EACH "
    "passage write ONE question that a reader could ask and that this passage answers. Rules: the question must be "
    "specific and self-contained (name the thing it asks about, never say 'the passage', 'this text' or 'the above'); "
    "it must be answerable from that passage alone; do not copy a sentence word for word. Return JSON "
    '{"questions": ["...", ...]} with exactly one question per passage, in the same order.'
)


def pick_chunks(rows: list, n: int) -> list:
    """n chunks spread evenly through the document, skipping tiny fragments and non-prose."""
    good = [r for r in rows if r["token_count"] >= 80 and r["kind"] == "prose"]
    if len(good) <= n:
        return good
    return [good[round(i * (len(good) - 1) / (n - 1))] for i in range(n)] if n > 1 else good[:1]


def clean_questions(raw: str, expected: int) -> list[str]:
    """Parse the model's JSON; keep only well-formed, self-contained questions."""
    try:
        items = json.loads(raw).get("questions", [])
    except (TypeError, ValueError, AttributeError):
        return []
    out = []
    for q in items[:expected] if isinstance(items, list) else []:
        if not isinstance(q, str):
            continue
        q = " ".join(q.split())
        if len(q.split()) < MIN_WORDS or any(b in q.lower() for b in BANNED):
            continue
        out.append(q if q.endswith("?") else q + "?")
    return out


def threshold_from(scores: list[float], pct: float | None = None) -> float | None:
    """The pct-th percentile of the scores, clamped. None if there is nothing to measure."""
    if not scores:
        return None
    raw = float(np.percentile(scores, settings.calibration_percentile if pct is None else pct))
    return min(max(raw, settings.relevance_min_threshold), settings.relevance_max_threshold)


async def generate_questions(passages: list[str]) -> list[str]:
    router = get_router()
    questions: list[str] = []
    for i in range(0, len(passages), BATCH):
        batch = passages[i : i + BATCH]
        body = "\n\n".join(f"[{j}] {p}" for j, p in enumerate(batch, 1))
        try:
            res = await router.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}], json_mode=True)
        except Exception as e:  # noqa: BLE001 - one failed batch must not abort calibration
            log.warning("question generation failed for a batch: %s", e)
            continue
        questions += clean_questions(res.text, len(batch))
    return questions


async def best_score(question: str) -> float | None:
    """Best chunk's raw rerank score for a question, through the same path a real question takes."""
    vec = await asyncio.to_thread(get_embedder().embed_query, question)
    async with db.pool().acquire() as conn:
        cands = await hybrid_search(conn, question, vec)
    top, reranked = await asyncio.to_thread(rerank, question, cands, 1)
    return float(top[0].rerank_score) if reranked and top and top[0].rerank_score is not None else None


async def calibrate_document(doc_id: UUID, n: int | None = None) -> dict:
    """Measure and store one document's threshold. Returns what was measured (also saved in documents.metadata)."""
    n = n or settings.calibration_questions
    async with db.pool().acquire() as conn:
        rows = await conn.fetch(
            "SELECT text, token_count, kind FROM chunks WHERE document_id = $1 ORDER BY chunk_index", doc_id)
    chosen = pick_chunks(list(rows), n)
    questions = await generate_questions([r["text"] for r in chosen])
    scores = [s for s in [await best_score(q) for q in questions] if s is not None]
    threshold = threshold_from(scores)
    report = {"questions": len(questions), "scored": len(scores), "threshold": threshold,
              "raw_p5": float(np.percentile(scores, settings.calibration_percentile)) if scores else None,
              "min": min(scores) if scores else None, "median": float(np.median(scores)) if scores else None,
              "max": max(scores) if scores else None, "sample": questions[:3]}
    if threshold is not None:
        async with db.pool().acquire() as conn:
            await conn.execute(
                "UPDATE documents SET relevance_threshold = $2, calibrated_at = now(), "
                "metadata = jsonb_set(metadata, '{calibration}', $3::jsonb) WHERE id = $1",
                doc_id, threshold, json.dumps(report))
    return report


async def load_thresholds(conn) -> dict[str, float]:
    """{document id: threshold} for every calibrated document."""
    rows = await conn.fetch("SELECT id, relevance_threshold FROM documents WHERE relevance_threshold IS NOT NULL")
    return {str(r["id"]): float(r["relevance_threshold"]) for r in rows}

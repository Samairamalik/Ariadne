"""Relevance gate: does the retrieved evidence actually cover the question?

Retrieval always returns the nearest chunks, even for an unrelated question. The cross-encoder score of the best
chunk separates the two cases (measured: relevant questions +2.8..+7.4 and down to -6.0 for two-topic comparisons;
unrelated -10..-11).

The cutoff is PER DOCUMENT: the best chunk is compared with the threshold calibrated for the document it came from
(see calibrate.py). A document with no calibration yet falls back to the global cutoff. Raw logits are only
guaranteed comparable within one query, so the thresholds are empirical and should be re-measured on new corpora.
Without the reranker, fall back to the best cosine (relevant 0.70..0.79, unrelated 0.41..0.48 in the same test).
"""
from app.config import settings


def cutoff_for(candidate, thresholds: dict[str, float] | None) -> float:
    """The threshold that applies to this candidate's document, else the global cutoff."""
    return (thresholds or {}).get(str(candidate.document_id), settings.relevance_rerank_cutoff)


def is_relevant(top: list, reranked: bool, thresholds: dict[str, float] | None = None) -> bool:
    if not top:
        return False
    scored = [c for c in top if c.rerank_score is not None]
    if reranked and scored:  # the best chunk by raw score (a paper preference may have reordered the list)
        best = max(scored, key=lambda c: c.rerank_score)
        return best.rerank_score >= cutoff_for(best, thresholds)
    return max(c.cosine for c in top) >= settings.relevance_cosine_cutoff

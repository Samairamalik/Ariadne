from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from app.ingestion.resolve import cosine, norm_name
from app.retrieval.relevance import is_relevant


def cand(rerank=None, cos=0.5):
    return SimpleNamespace(rerank_score=rerank, cosine=cos, document_id=uuid4())


def test_reranker_score_decides_when_available():
    assert is_relevant([cand(rerank=4.0)], reranked=True)
    assert not is_relevant([cand(rerank=-10.5, cos=0.9)], reranked=True)  # reranker overrides a misleading cosine


def test_cosine_fallback_without_reranker():
    assert is_relevant([cand(cos=0.72)], reranked=False)
    assert not is_relevant([cand(cos=0.43), cand(cos=0.41)], reranked=False)


def test_nothing_retrieved_is_not_relevant():
    assert not is_relevant([], reranked=True)


def test_norm_name_ignores_case_punctuation_spacing():
    assert norm_name("Fine-Tuning") == norm_name("fine tuning") == norm_name("  FINE_TUNING ")
    assert norm_name("BERT-base") != norm_name("BERTBASE")  # spelling variants are intentionally not merged


def test_cosine_of_context_vectors():
    a = np.array([1.0, 0.0])
    assert cosine(a, np.array([2.0, 0.0])) == 1.0
    assert abs(cosine(a, np.array([0.0, 3.0]))) < 1e-9
    assert cosine(a, np.zeros(2)) == 0.0

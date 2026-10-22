import json
from types import SimpleNamespace
from uuid import uuid4

from app.config import settings
from app.retrieval.calibrate import clean_questions, pick_chunks, threshold_from
from app.retrieval.relevance import cutoff_for, is_relevant


def rows(n, tokens=120, kind="prose"):
    return [{"text": f"chunk {i}", "token_count": tokens, "kind": kind} for i in range(n)]


def test_pick_chunks_spreads_evenly_and_skips_fragments_and_non_prose():
    picked = pick_chunks(rows(100), 5)
    assert [r["text"] for r in picked] == ["chunk 0", "chunk 25", "chunk 50", "chunk 74", "chunk 99"]
    assert pick_chunks(rows(3), 20) == rows(3)  # fewer than n: take all
    assert pick_chunks(rows(10, tokens=40), 5) == [] and pick_chunks(rows(10, kind="table"), 5) == []


def test_clean_questions_keeps_only_self_contained_questions():
    raw = json.dumps({"questions": ["What window size does Longformer use", "What does this passage say?",
                                    "Why?", 7, "How does BigBird keep full-attention power?"]})
    assert clean_questions(raw, 5) == ["What window size does Longformer use?", "How does BigBird keep full-attention power?"]
    assert clean_questions("not json", 3) == [] and clean_questions('{"questions": "x"}', 3) == []


def test_threshold_is_the_fifth_percentile_clamped(monkeypatch):
    monkeypatch.setattr(settings, "relevance_min_threshold", -9.0)
    monkeypatch.setattr(settings, "relevance_max_threshold", 0.0)
    monkeypatch.setattr(settings, "calibration_percentile", 5.0)
    scores = list(range(-4, 16))  # 20 values; 5th percentile is about -3.05
    assert abs(threshold_from(scores) - (-3.05)) < 0.01
    assert threshold_from([5.0, 6.0, 7.0]) == 0.0           # ceiling: easy questions cannot demand a positive score
    assert threshold_from([-30.0, -20.0, -25.0]) == -9.0   # floor: cannot drift below unrelated-question territory
    assert threshold_from([]) is None


def cand(doc, score):
    return SimpleNamespace(document_id=doc, rerank_score=score, cosine=0.0)


def test_gate_uses_the_threshold_of_the_document_the_best_chunk_came_from(monkeypatch):
    monkeypatch.setattr(settings, "relevance_rerank_cutoff", -8.0)
    a, b = uuid4(), uuid4()
    thresholds = {str(a): -2.0, str(b): -9.0}
    assert not is_relevant([cand(a, -3.0)], True, thresholds)   # A is strict: -3 is below A's -2
    assert is_relevant([cand(b, -3.0)], True, thresholds)       # B is lenient
    assert is_relevant([cand(uuid4(), -7.0)], True, thresholds)  # uncalibrated document: global -8 applies
    assert cutoff_for(cand(a, 0), None) == -8.0


def test_best_chunk_decides_not_list_order(monkeypatch):
    monkeypatch.setattr(settings, "relevance_rerank_cutoff", -8.0)
    a, b = uuid4(), uuid4()
    top = [cand(a, -9.5), cand(b, 3.0)]  # a paper preference put the weak chunk first
    assert is_relevant(top, True, {str(a): -1.0, str(b): 0.0})

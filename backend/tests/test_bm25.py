import math

from app.retrieval.bm25 import bm25_scores, idf, rank

N, AVG = 1000, 100.0


def score(rows, df):
    return bm25_scores(rows, df, N, AVG)


def test_rare_terms_outweigh_common_ones():
    assert idf(N, 3) > 5 * idf(N, 900)  # "RoBERTa" (3 chunks) vs "model" (900 of 1000)
    rows = [("a", 100, "roberta", 1), ("b", 100, "model", 1)]
    s = score(rows, {"roberta": 3, "model": 900})
    assert s["a"] > s["b"]


def test_term_frequency_saturates():
    df = {"batch": 50}
    one, two, ten, hundred = (score([("c", 100, "batch", tf)], df)["c"] for tf in (1, 2, 10, 100))
    assert one < two < ten < hundred
    assert (ten - two) < 5 * (two - one) * 2  # 10 mentions are nowhere near 5x the score of 2
    assert hundred < (1.2 + 1) * idf(N, 50)  # bounded above by idf * (k1 + 1)


def test_longer_chunks_are_penalised_for_the_same_count():
    df = {"batch": 50}
    short = score([("s", 50, "batch", 2)], df)["s"]
    long_ = score([("l", 400, "batch", 2)], df)["l"]
    assert short > long_


def test_matching_more_query_terms_scores_higher_and_unknown_terms_are_ignored():
    df = {"batch": 20, "size": 20}
    both = score([("a", 100, "batch", 1), ("a", 100, "size", 1), ("b", 100, "batch", 1), ("c", 100, "missing", 1)], df)
    assert both["a"] > both["b"] and "c" not in both


def test_zero_document_frequency_term_is_skipped_and_idf_is_never_negative():
    assert score([("a", 100, "x", 1)], {"x": 0}) == {}
    assert idf(10, 10) > 0  # Lucene form: a term in every chunk still has tiny positive weight, not negative


def test_rank_is_best_first_deterministic_and_limited():
    scores = {"b": 1.0, "a": 1.0, "c": 3.0, "d": 2.0}
    assert rank(scores, 3) == ["c", "d", "a"]  # tie between a and b broken by id
    assert rank({}, 5) == []


def test_formula_matches_hand_computation():
    # idf = ln(1 + (1000-10+0.5)/(10+0.5)); tf=3, len=avg -> norm = 3 + 1.2 ; score = idf * 3 * 2.2 / 4.2
    expected = math.log(1 + (1000 - 10 + 0.5) / 10.5) * 3 * 2.2 / (3 + 1.2)
    assert abs(score([("a", 100, "t", 3)], {"t": 10})["a"] - expected) < 1e-9

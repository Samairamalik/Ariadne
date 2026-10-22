from app.retrieval.fusion import lexical_terms, rrf, to_tsquery_or


def test_lexical_terms_drop_stopwords_and_dedupe():
    q = "What is the cost of attention in the Transformer, and how does attention scale?"
    assert lexical_terms(q) == ["cost", "attention", "transformer", "scale"]


def test_tsquery_is_or_and_injection_safe():
    q = "BERT's [CLS] token'); DROP TABLE chunks;-- & | !"
    ts = to_tsquery_or(q)
    assert ts == "bert | cls | token | drop | table | chunks"
    assert not any(ch in ts.replace(" | ", "") for ch in "';()&!:*<>-")


def test_tsquery_none_when_only_stopwords():
    assert to_tsquery_or("what is the of and") is None


def test_rrf_rewards_agreement_over_single_list_dominance():
    fused = rrf({"vec": ["a", "b", "c"], "lex": ["d", "b", "e"]})
    assert fused[0].id == "b"  # found by both retrievers at rank 2 beats a, d (rank 1 in only one list)
    assert fused[0].ranks == {"vec": 2, "lex": 2}


def test_rrf_uses_one_indexed_ranks():
    (only,) = rrf({"vec": ["a"]}, k=60)
    assert abs(only.score - 1 / 61) < 1e-12


def test_rrf_deterministic_tiebreak_and_weights():
    fused = rrf({"vec": ["a"], "lex": ["b"]})
    assert [f.id for f in fused] == ["a", "b"]
    weighted = rrf({"vec": ["a"], "lex": ["b"]}, weights={"lex": 2.0})
    assert [f.id for f in weighted] == ["b", "a"]

from app.retrieval.graph import candidate_phrases, fuzzy_rank

NAMES = {"1": "BigBird", "2": "Longformer", "3": "sliding window attention", "4": "model"}


def test_phrases_include_multiword_and_skip_stopword_edges():
    ph = candidate_phrases("How does Longformer attention compare with Big Bird?")
    assert "big bird" in ph and "longformer" in ph
    assert not any(p.split()[0] in {"how", "does", "with"} or p.split()[-1] in {"with", "how"} for p in ph)


def test_fuzzy_matches_spacing_and_typos():
    assert fuzzy_rank(["big bird"], NAMES, 85) == ["1"]
    assert fuzzy_rank(["longformr"], NAMES, 85) == ["2"]


def test_no_match_returns_empty():
    assert fuzzy_rank(["quantum chromodynamics"], NAMES, 90) == []


def test_exact_multiword_node_name():
    assert fuzzy_rank(["sliding window attention"], NAMES, 90) == ["3"]

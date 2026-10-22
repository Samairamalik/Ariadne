from types import SimpleNamespace
from uuid import uuid4

from app.generation import build_prompt
from app.retrieval.graph_context import Fact, rank_graph_chunks


def test_chunks_ranked_by_weighted_fact_count_and_known_chunks_excluded():
    a, b, known = uuid4(), uuid4(), uuid4()
    facts = [Fact("x", "p", "y", a, 1), Fact("x", "p", "z", b, 2), Fact("x", "p", "w", b, 2), Fact("q", "p", "r", known, 1)]
    ranked = rank_graph_chunks(facts, {known})
    assert known not in [c for c, _ in ranked]
    assert dict(ranked)[a] == 1 and dict(ranked)[b] == 2  # best hop kept


def test_closer_hops_outrank_farther_ones():
    near, far = uuid4(), uuid4()
    ranked = rank_graph_chunks([Fact("a", "p", "b", far, 2), Fact("a", "p", "c", near, 1)], set())
    assert [c for c, _ in ranked] == [near, far]


def src(cid, text):
    return SimpleNamespace(chunk_id=cid, doc_title="Doc", section_path="S", page_start=1, text=text)


def test_prompt_lists_facts_with_the_number_of_their_source_and_skips_unlisted_chunks():
    c1, c2, outside = uuid4(), uuid4(), uuid4()
    facts = [Fact("BERT", "uses", "MLM", c2, 1), Fact("ghost", "is", "gone", outside, 1)]
    prompt = build_prompt("q?", [src(c1, "one"), src(c2, "two")], facts)
    assert "GRAPH FACTS" in prompt and "(BERT) -[uses]-> (MLM)  [2]" in prompt and "ghost" not in prompt


def test_prompt_without_facts_is_unchanged_shape():
    assert "GRAPH FACTS" not in build_prompt("q?", [src(uuid4(), "one")])

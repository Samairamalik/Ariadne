from uuid import UUID

from app.generation import build_prompt
from app.retrieval.communities import detect, is_global_question


def u(n):
    return UUID(int=n)


def test_global_question_detection():
    for q in ("What are the main themes across these papers?", "Give me an overview of the documents",
              "Summarize everything", "What are the key findings overall?"):
        assert is_global_question(q), q
    for q in ("What is BERT pre-trained on?", "What window size does Longformer use?"):
        assert not is_global_question(q), q


def test_detect_splits_two_dense_groups_and_drops_small_ones():
    a = [(u(i), u(j)) for i in range(1, 5) for j in range(i + 1, 5)]       # 4-clique
    b = [(u(i), u(j)) for i in range(11, 15) for j in range(i + 1, 15)]    # another 4-clique
    tiny = [(u(21), u(22))]
    comms = detect(a + b + [(u(4), u(11))] + tiny, min_size=4)
    assert sorted(len(c) for c in comms) == [4, 4]
    assert detect([], 4) == []


def test_overview_is_in_prompt_and_marked_not_citable():
    p = build_prompt("q?", [], overview=[("Sparse attention", "Methods that restrict attention.")])
    assert p.startswith("OVERVIEW (background, do not cite)") and "Sparse attention" in p

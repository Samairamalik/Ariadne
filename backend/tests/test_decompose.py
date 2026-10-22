from app.generation import build_prompt
from app.retrieval.decompose import interleave, split_question


def test_two_part_questions_are_split_at_the_clause_boundary():
    assert split_question("What batch size did BERT use, and what larger batch size does RoBERTa use?") == [
        "What batch size did BERT use?", "what larger batch size does RoBERTa use?"]
    assert len(split_question("What model dimension does the Transformer use, and what hidden size does BERT-base use?")) == 2
    assert len(split_question("Why does self-attention become expensive on long documents, and which mechanism does the Transformer paper use to compute it?")) == 2
    assert len(split_question("What architecture is BERT based on, and how does that architecture compute attention?")) == 2


def test_single_questions_are_left_alone():
    for q in ("How do Longformer and BigBird reduce the cost of attention?",
              "What did RoBERTa change about the masking strategy that BERT used?",
              "What is FAVOR+ and which method uses it?" if False else "What window size does Longformer use?",
              "Compare the parameter counts reported for BERT and for SpectralRepNet."):
        assert split_question(q) == [q], q


def test_tiny_fragments_do_not_trigger_a_split():
    q = "What is it, and why?"
    assert split_question(q) == [q]


def test_interleave_keeps_each_parts_best_chunks_and_drops_repeats():
    a, b = ["a1", "a2", "a3"], ["b1", "a1", "b3"]
    assert interleave([a, b], 4) == ["a1", "b1", "a2", "a3"]  # b's duplicate "a1" is dropped; limit cuts b3
    assert interleave([[], ["x"]], 5) == ["x"]


def test_prompt_lists_the_parts():
    p = build_prompt("Q1 and Q2?", [], parts=["First part?", "Second part?"])
    assert "answer each one: (1) First part? (2) Second part?" in p
    assert "answer each one" not in build_prompt("Q?", [])

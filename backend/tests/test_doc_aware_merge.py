from types import SimpleNamespace
from uuid import UUID

from app.ingestion.resolve import compact_key, is_contained_variant, name_tokens
from app.retrieval.doc_aware import anchors, match_documents, names_anchor
from app.retrieval.rerank import order_with_preference


def u(n):
    return UUID(int=n)


TITLES = {
    1: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    2: "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    3: "RepVGG: Making VGG-style ConvNets Great Again",
    4: "Attention Is All You Need",
    5: "SpectralRepNet: Real-Time Image Dehazing for Edge Surveillance via Structural Reparameterization",
}
DOCS = [(u(i), t) for i, t in TITLES.items()]


def test_anchor_is_the_short_name_before_the_colon_or_a_short_whole_title():
    assert anchors(TITLES[1]) == ["BERT"] and anchors(TITLES[2]) == ["RoBERTa"]
    assert anchors(TITLES[4]) == ["Attention Is All You Need"]
    assert anchors("A very long title with no colon that goes on and on forever") == []
    assert anchors(None) == []


def test_whole_word_matching_does_not_confuse_bert_with_roberta():
    assert names_anchor("What vocabulary size does BERT use?", "BERT")
    assert names_anchor("What does bert-base use?", "BERT")
    assert not names_anchor("What did RoBERTa change?", "BERT")
    assert names_anchor("What did RoBERTa change?", "RoBERTa")


def test_match_documents_prefers_only_the_named_papers():
    assert match_documents("What vocabulary size does BERT use?", DOCS) == [u(1)]
    assert set(match_documents("How does RoBERTa differ from BERT?", DOCS)) == {u(1), u(2)}
    assert match_documents("What is the boiling point of water?", DOCS) == []
    everything = "BERT RoBERTa RepVGG Attention Is All You Need SpectralRepNet"
    assert match_documents(everything, DOCS) == []  # names every paper: no preference


def cand(doc, score):
    return SimpleNamespace(document_id=doc, rerank_score=score)


def test_preference_reorders_but_keeps_the_raw_score_for_the_relevance_gate():
    roberta, bert = cand(u(2), 3.0), cand(u(1), 0.5)
    out = order_with_preference([roberta, bert], {u(1)}, 4.0)
    assert out[0] is bert and bert.rerank_score == 0.5  # moved up by the bonus, raw logit untouched


def test_preference_does_not_resurrect_a_clearly_irrelevant_chunk():
    junk, good = cand(u(1), -10.0), cand(u(2), 5.0)
    assert order_with_preference([junk, good], {u(1)}, 4.0)[0] is good


def test_compact_key_collapses_spelling_variants():
    assert compact_key("re-parameterization") == compact_key("reparameterization")
    assert compact_key("RepVGG block") == compact_key("RepVGGBlock")
    assert compact_key("RepVGG-A0") != compact_key("RepVGG-A2")
    assert compact_key("BERTBASE") != compact_key("BERTLARGE")


def test_contained_variant_allows_only_generic_suffix_or_a_paper_name():
    allowed = frozenset({"function", "method", "repvgg"})
    assert is_contained_variant(name_tokens("GELU activation"), name_tokens("GELU activation function"), allowed)
    assert is_contained_variant(name_tokens("structural re-parameterization"),
                                name_tokens("RepVGG structural reparameterization"), allowed)
    # the wrong merges that high context similarity would have allowed:
    assert not is_contained_variant(name_tokens("learning rate"), name_tokens("learning rate optimizer"), allowed)
    assert not is_contained_variant(name_tokens("downstream tasks"), name_tokens("performance on downstream tasks"), allowed)
    assert not is_contained_variant(name_tokens("English Wikipedia"), name_tokens("BOOKCORPUS plus English WIKIPEDIA"), allowed)
    # a single-word shorter name never qualifies ("attention" vs "self attention")
    assert not is_contained_variant(name_tokens("attention"), name_tokens("attention function"), allowed)


def test_generic_one_word_nodes_are_not_walk_starting_points():
    from app.retrieval.graph import is_specific_name
    for generic in ("cost", "size", "architecture", "fine-tuning", "attention"):
        assert not is_specific_name(generic), generic
    for specific in ("BERT", "Transformer", "SQuAD2", "masked language model", "sliding window"):
        assert is_specific_name(specific), specific


def test_prompt_states_which_paper_the_question_is_about():
    from app.generation import FOCUS_ADDENDUM, build_prompt
    p = build_prompt("What vocabulary does BERT use?", [], focus=["BERT: Pre-training of Deep Bidirectional Transformers"])
    assert p.startswith("FOCUS (the question is about): BERT: Pre-training")
    assert "different document" in FOCUS_ADDENDUM
    assert "FOCUS" not in build_prompt("q?", [])

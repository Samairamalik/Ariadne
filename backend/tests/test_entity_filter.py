import pytest

from app.ingestion.entity_filter import is_core_node, is_main_entity


@pytest.mark.parametrize("name,node_type", [
    ("BERT", "model"), ("RoBERTa", "model"), ("RESIDE-6K", "dataset"), ("ResNet-50", "model"), ("SQuAD 2.0", "dataset"),
    ("self-attention", "method"), ("next sentence prediction", "method"), ("structural re-parameterization", "method"),
    ("Multi-Head Attention", "method"), ("Llion Jones", "person"), ("Attention Is All You Need", "paper"),
    ("scaled dot-product attention", "method"), ("GNMT + RL", "model"), ("fine-tuning", "method"),
])
def test_main_entities_are_kept(name, node_type):
    assert is_main_entity(name, node_type)


@pytest.mark.parametrize("name", [
    "1024", "32", "size", "formula", "Table 2", "Related Work", "a pair of natural sentences", "80% top-1 accuracy on ImageNet",
    "inference speed by 19.7%", "0.38M parameters and 4.48 GFLOPs", "to prevent positions from attending to subsequent positions",
    "Effect of Number of Training Steps", "three ingredients", "recent work", "S = 1,000,000 updates", "BLEU score of 39.2",
    "compute and memory growth", "inference latency results", "very long inputs but can lose accuracy on tasks that need precise recall",
])
def test_fragments_and_filler_are_dropped(name):
    assert not is_main_entity(name, "concept")


def test_core_node_keeps_short_proper_names_with_one_mention_but_not_lowercase_phrases():
    assert is_core_node("Longformer", "model", 1)
    assert not is_core_node("channel-wise magnitude gates", "concept", 1)
    assert is_core_node("channel-wise magnitude gates", "concept", 2)

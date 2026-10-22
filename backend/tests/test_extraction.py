import json
from types import SimpleNamespace

from app.ingestion.extraction import Triple, attribute_to_chunks, build_messages, extract_batch, parse_triples


def t(**kw):
    base = {"chunk": 1, "subject": "Ariadne", "subject_type": "model", "predicate": "uses",
            "object": "pgvector", "object_type": "method", "evidence": "Ariadne uses pgvector."}
    return {**base, **kw}


def test_valid_triple_is_parsed_with_zero_based_chunk():
    out = parse_triples(json.dumps({"triples": [t(chunk=2)]}), 3)
    assert out == [Triple(1, "Ariadne", "model", "uses", "pgvector", "method", "Ariadne uses pgvector.")]


def test_unknown_type_falls_back_to_other():
    out = parse_triples(json.dumps({"triples": [t(subject_type="Company", object_type=None)]}), 1)
    assert out[0].subject_type == "other" and out[0].object_type == "other"


def test_malformed_items_are_dropped():
    bad = [t(chunk=0), t(chunk=5), t(chunk="x"), t(chunk="9"), t(chunk=True), t(subject=""), t(predicate=None),
           t(object="ariadne"), "nope", 7]
    assert parse_triples(json.dumps({"triples": bad + [t()]}), 3) == parse_triples(json.dumps({"triples": [t()]}), 3)


def test_garbage_output_returns_empty():
    for raw in ("not json", "", None, "{}", '{"triples": "x"}', "42"):
        assert parse_triples(raw, 2) == []


def test_bare_list_is_accepted_and_long_text_truncated():
    out = parse_triples(json.dumps([t(subject="x" * 500)]), 1)
    assert len(out[0].subject) == 120


def test_messages_number_the_passages():
    msgs = build_messages(["first", "second"])
    assert msgs[1]["content"] == "[1] first\n\n[2] second"


async def test_extract_batch_uses_json_mode():
    seen = {}

    class R:
        async def chat(self, messages, *, json_mode, temperature):
            seen["json_mode"] = json_mode
            return SimpleNamespace(text=json.dumps({"triples": [t()]}))

    assert len(await extract_batch(R(), ["Ariadne uses pgvector."])) == 1 and seen["json_mode"] is True


def test_chunk_number_given_as_digit_string_is_accepted():
    out = parse_triples(json.dumps({"triples": [t(chunk="2")]}), 3)
    assert [x.chunk for x in out] == [1]


TEXTS = ["Ariadne uses pgvector.", "Light scatters in haze.", "Unrelated text."]


def tr(chunk, ev):
    return Triple(chunk, "a", "other", "p", "b", "other", ev)


def test_correct_chunk_is_kept():
    assert attribute_to_chunks([tr(1, "Light scatters in haze")], TEXTS) == [tr(1, "Light scatters in haze")]


def test_wrong_chunk_number_is_corrected_from_the_quote():
    assert attribute_to_chunks([tr(0, "light scatters in haze.")], TEXTS) == [tr(1, "light scatters in haze.")]


def test_quote_not_found_in_any_chunk_drops_the_triple():
    assert attribute_to_chunks([tr(0, "invented sentence about nothing"), tr(0, "")], TEXTS) == []


def test_quote_in_model_chunk_wins_over_an_earlier_match():
    texts = ["same words here", "same words here"]
    assert attribute_to_chunks([tr(1, "same words here")], texts)[0].chunk == 1


# ---- paper-aware extraction + verification pass
from app.ingestion.extraction import (SELF_REFERENCES, build_messages as _bm, document_subject, entity_supported,
                                      resolve_self_references, verify_triples)

BERT_TITLE = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"


def tri(s, o, ev="x", p="uses"):
    return Triple(0, s, "other", p, o, "other", ev)


def test_document_subject_is_the_short_name_in_the_title():
    assert document_subject(BERT_TITLE) == "BERT"
    assert document_subject(None) is None


def test_prompt_names_the_document_and_each_chunks_section():
    msgs = _bm(["text one", "text two"], BERT_TITLE, ["3 BERT", None])
    user = msgs[1]["content"]
    assert user.startswith(f"DOCUMENT: {BERT_TITLE}") and "what 'we' and 'our model' mean): BERT" in user
    assert "[1] (section: 3 BERT) text one" in user and "[2] text two" in user
    assert "never the pronoun" in msgs[0]["content"]


def test_without_a_title_the_prompt_is_the_old_plain_numbering():
    assert _bm(["a", "b"])[1]["content"] == "[1] a\n\n[2] b"


def test_self_references_become_the_papers_name():
    out = resolve_self_references([tri("we", "masked language models"), tri("Our model", "WordPiece"),
                                   tri("The Model", "Adam"), tri("Adam", "the proposed approach")], "BERT")
    assert [(t.subject, t.object) for t in out] == [("BERT", "masked language models"), ("BERT", "WordPiece"),
                                                     ("BERT", "Adam"), ("Adam", "BERT")]
    assert "we" in SELF_REFERENCES


def test_self_reference_to_itself_is_dropped_and_no_subject_means_no_change():
    assert resolve_self_references([tri("we", "our model")], "BERT") == []
    t = [tri("we", "x")]
    assert resolve_self_references(t, None) == t


def test_verification_drops_the_fact_attached_to_the_wrong_entity():
    quote = "we test on RTTS, which contains bounding-boxes for 4,322 real hazy images over 5 object categories"
    bad = tri("RESIDE-6K", "4,322 real hazy images", quote, "contains")  # the real error found in the graph
    good = tri("RTTS", "4,322 real hazy images", quote, "contains")
    assert verify_triples([bad, good]) == [good]


def test_verification_accepts_spelling_variants_and_the_papers_own_name():
    assert entity_supported("structural reparameterization", "a structural re-parameterization technique")
    assert entity_supported("BERT", "We introduce a new language representation model", subject="BERT")
    assert not entity_supported("BERT", "We introduce a new language representation model")
    assert not entity_supported("the model", "completely unrelated words here")


def test_extract_batch_runs_the_whole_pipeline():
    import asyncio, json as _j
    from types import SimpleNamespace as NS
    reply = {"triples": [
        {"chunk": 2, "subject": "we", "subject_type": "model", "predicate": "pre-train", "object": "deep bidirectional representations",
         "object_type": "concept", "evidence": "we pre-train deep bidirectional representations"},      # wrong chunk number, self-reference
        {"chunk": 1, "subject": "RESIDE-6K", "subject_type": "dataset", "predicate": "contains", "object": "4,322 images",
         "object_type": "dataset", "evidence": "RTTS contains 4,322 images"},                           # wrong entity: dropped
    ]}

    class R:
        async def chat(self, messages, *, json_mode, temperature):
            return NS(text=_j.dumps(reply))

    texts = ["We pre-train deep bidirectional representations from unlabeled text.", "RTTS contains 4,322 images."]
    out = asyncio.run(extract_batch(R(), texts, BERT_TITLE))
    assert [(t.subject, t.object, t.chunk) for t in out] == [("BERT", "deep bidirectional representations", 0)]

from app.ingestion.sentences import split_sentences


def test_et_al_and_fig_do_not_split():
    s = "As shown by Vaswani et al. in Fig. 3, attention helps. It scales poorly."
    assert split_sentences(s) == [
        "As shown by Vaswani et al. in Fig. 3, attention helps.",
        "It scales poorly.",
    ]


def test_decimals_and_initials():
    s = "The model gets 28.4 BLEU. J. Smith reported 0.5 less."
    assert split_sentences(s) == ["The model gets 28.4 BLEU.", "J. Smith reported 0.5 less."]


def test_eg_ie():
    s = "Use regularisation, e.g. dropout. This helps, i.e. it generalises better."
    assert len(split_sentences(s)) == 2


def test_question_and_exclamation():
    assert split_sentences("Does it scale? Yes! It does.") == ["Does it scale?", "Yes!", "It does."]


def test_no_trailing_punctuation_and_empty():
    assert split_sentences("A fragment without end") == ["A fragment without end"]
    assert split_sentences("   ") == []

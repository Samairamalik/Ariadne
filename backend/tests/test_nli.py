import numpy as np

from app import nli
from app.config import settings

LABELS = {"contradiction": 0, "entailment": 1, "neutral": 2}


class FakeNLI:
    """entailment if every content word of the claim is in the premise; contradiction if only a number differs; else neutral."""

    def __init__(self):
        self.pairs = []

    def predict(self, pairs, show_progress_bar=False):
        self.pairs += list(pairs)
        out = []
        for premise, claim in pairs:
            cw, pw = nli.content_words(claim), nli.content_words(premise)
            nums_c = {w for w in cw if w.isdigit()}
            nums_p = {w for w in pw if w.isdigit()}
            if cw <= pw:
                out.append([0.01, 0.98, 0.01])
            elif nums_c and nums_p and not nums_c & nums_p and (cw - nums_c) <= pw:
                out.append([0.97, 0.01, 0.02])
            else:
                out.append([0.10, 0.20, 0.70])
        return np.array(out)


SOURCES = ["We use WordPiece embeddings with a 30000 token vocabulary. The first token is always CLS.",
           "Training took 3.5 days on 8 P100 GPUs. We used the Adam optimizer."]


def run(answer):
    return nli.verify_answer(answer, SOURCES, FakeNLI(), LABELS)


def test_supported_claim_with_the_right_source():
    (c,) = run("We use WordPiece embeddings with a 30000 token vocabulary [1].")
    assert c.verdict == "supported" and c.cited == [1] and c.entailment > 0.9


def test_claim_citing_the_wrong_source_is_unverified_not_condemned():
    (c,) = run("We use WordPiece embeddings with a 30000 token vocabulary [2].")
    assert c.verdict == "unverified" and c.entailment < 0.5


def test_wrong_number_against_the_right_source_is_contradicted_only_when_very_sure():
    (c,) = run("We use WordPiece embeddings with a 50000 token vocabulary [1].")
    assert c.verdict == "contradicted" and c.contradiction >= settings.nli_contradiction_cutoff


def test_moderate_contradiction_stays_unverified(monkeypatch):
    class Shaky:
        def predict(self, pairs, show_progress_bar=False):
            return np.array([[0.7, 0.1, 0.2]] * len(pairs))  # contradiction 0.7: below the 0.9 bar

    (c,) = nli.verify_answer("We use WordPiece embeddings with a 30000 token vocabulary [1].", SOURCES, Shaky(), LABELS)
    assert c.verdict == "unverified"


def test_best_cited_source_counts_when_several_are_cited():
    (c,) = run("Training took 3.5 days on 8 P100 GPUs [1][2].")
    assert c.cited == [1, 2] and c.verdict == "supported"


def test_uncited_sentences_short_fragments_and_bad_numbers_are_skipped():
    assert run("This sentence has no citation at all. Too short [1]. Also cites nothing real here [9].") == []


def test_citations_are_stripped_and_the_stray_space_they_leave_is_removed():
    assert nli.strip_citations("BERT masks 15% of tokens [1][2] .") == "BERT masks 15% of tokens."
    fake = FakeNLI()
    nli.verify_answer("The Adam optimizer was used for training [2].", SOURCES, fake, LABELS)
    assert fake.pairs and all("[2]" not in claim for _, claim in fake.pairs)


def test_one_pair_per_cited_source():
    fake = FakeNLI()
    nli.verify_answer("Training took 3.5 days on 8 P100 GPUs [1][2].", SOURCES, fake, LABELS)
    assert len(fake.pairs) == 2


def test_premise_is_the_most_relevant_sentences_in_original_order(monkeypatch):
    monkeypatch.setattr(settings, "nli_premise_sentences", 2)
    text = "Alpha one. Beta two is unrelated. Gamma three matters. Delta four is filler. Alpha again matters."
    prem = nli.joined_premise("Gamma three and alpha again matter", text)
    assert prem == "Gamma three matters. Alpha again matters."
    assert nli.joined_premise("anything", "One sentence only.") == "One sentence only."


def test_to_probs_accepts_logits_and_probabilities():
    assert np.allclose(nli.to_probs([[0.2, 0.7, 0.1]]).sum(axis=1), 1.0)
    p = nli.to_probs([[5.0, -2.0, 0.0]])
    assert abs(p.sum() - 1.0) < 1e-9 and p[0, 0] > 0.9


def test_conflicting_sources_are_flagged_not_hidden():
    class Mixed:
        def predict(self, pairs, show_progress_bar=False):
            return np.array([[0.95, 0.03, 0.02] if i == 0 else [0.02, 0.96, 0.02] for i, _ in enumerate(pairs)])

    (c,) = nli.verify_answer("We use a 30000 token vocabulary here [1][2].", SOURCES, Mixed(), LABELS)
    assert c.verdict == "supported" and c.conflict


def test_summary_counts_each_verdict_and_no_model_means_no_checks(monkeypatch):
    checks = run("We use WordPiece embeddings with a 30000 token vocabulary [1]. We use WordPiece embeddings with a 50000 token vocabulary [1].")
    s = nli.summarize(checks)
    assert s["available"] and s["claims"] == 2 and s["supported"] == 1
    assert s["supported"] + s["unverified"] + s["contradicted"] == 2
    monkeypatch.setattr(nli, "load_model", lambda: (None, None))
    assert nli.verify_answer("A cited claim with enough words here [1].", SOURCES) == []

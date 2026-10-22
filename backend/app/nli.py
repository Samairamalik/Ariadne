"""Semantic citation check with a Natural Language Inference (NLI) model.

The older check only proved citations are well-formed (every [n] points at a real source). It could not tell whether
the cited passage actually says what the sentence claims. NLI can, within limits measured below.

A cross-encoder trained on one task reads a PREMISE (cited text) and a HYPOTHESIS (the answer sentence) and outputs the
probability of three labels: entailment (the premise supports the claim), contradiction, neutral (says nothing about it).

Per cited claim:
  1. strip the [n] markers: the hypothesis is the bare sentence;
  2. for each cited source, build ONE premise: the few sentences of the chunk that share the most words with the claim
     (kept in their original order). Whole chunks and single sentences both measured worse (see below);
  3. entailment = the best across the cited sources.
Verdicts: supported (entailment >= cutoff) | contradicted (contradiction >= its stricter cutoff and no entailment)
          | unverified (anything else).

WHAT THE MEASUREMENTS SAY (38 cited claims from answers that passed every other check, plus a control that pairs each
claim with a chunk from a different paper):
    config                      true claims supported   wrong citation accepted   true claims called contradicted
    small model, short windows            37%                    0 of 38                      39%
    base model, joined 3 sentences        42%                    0 of 38                       8%     <- used
So it is a PRECISE CONFIRMER (it never accepted a wrong citation) but a modest-recall one: more than half of correct,
cited sentences come back 'unverified', typically because the claim paraphrases several sentences, adds context the
premise lacks, or relies on a pronoun ("it", "we") the model cannot resolve. 'unverified' therefore means "not
confirmed", never "wrong". It also checks the cited TEXT, not which paper's version that text describes.

Runs locally on the CPU. If the model cannot load, verification is reported as unavailable; answers are never blocked.
"""
import logging
import re
import threading
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.ingestion.sentences import split_sentences

log = logging.getLogger("ariadne.nli")
_CITE = re.compile(r"\[(\d+)\]")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the of in on at to for from by with and or but is are was were be been it its this that these those as "
                  "which who what how also than then so not no can could should would may might will do does did has have had".split())
MIN_CLAIM_WORDS = 4

_state: dict = {"model": None, "labels": None, "status": "idle", "error": None}
_lock = threading.Lock()


def load_model():
    """Load once, lazily and thread-safely. Returns (model, label_index) or (None, None) with the reason recorded."""
    with _lock:
        if _state["status"] == "ready":
            return _state["model"], _state["labels"]
        if _state["status"] == "failed":
            return None, None
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(settings.nli_model, device="cpu")
            id2label = {int(k): v.lower() for k, v in model.model.config.id2label.items()}
            labels = {name: i for i, name in id2label.items()}
            if not {"entailment", "contradiction", "neutral"} <= set(labels):
                raise ValueError(f"unexpected NLI labels {id2label}")
            _state.update(model=model, labels=labels, status="ready", error=None)
        except Exception as e:  # noqa: BLE001 - verification is optional, never fatal
            _state.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
            log.warning("NLI model unavailable: %s", _state["error"])
            return None, None
        return _state["model"], _state["labels"]


def status() -> dict:
    return {"status": _state["status"], "error": _state["error"], "model": settings.nli_model}


def strip_citations(sentence: str) -> str:
    return " ".join(_CITE.sub("", sentence).split()).replace(" .", ".").replace(" ,", ",")


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def joined_premise(claim: str, source_text: str, k: int | None = None) -> str:
    """The k sentences of the source sharing the most content words with the claim, in their original order."""
    k = k or settings.nli_premise_sentences
    sents = [s for s in split_sentences(source_text) if s.strip()] or [source_text]
    claim_words = content_words(claim)
    top = sorted(range(len(sents)), key=lambda i: (-len(claim_words & content_words(sents[i])), i))[:k]
    return " ".join(sents[i] for i in sorted(top))


def to_probs(raw) -> np.ndarray:
    """Model output as probabilities: accepts logits or probabilities, one row per pair."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if np.all((arr >= 0) & (arr <= 1)) and np.allclose(arr.sum(axis=1), 1.0, atol=1e-3):
        return arr
    e = np.exp(arr - arr.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class ClaimCheck:
    sentence: str
    cited: list[int]
    entailment: float
    contradiction: float
    verdict: str  # supported | unverified | contradicted
    conflict: bool = False  # one cited source entails the claim while another contradicts it: mixed evidence


def verdict_for(entail: float, contra: float) -> str:
    if entail >= settings.nli_entail_cutoff:
        return "supported"
    if contra >= settings.nli_contradiction_cutoff:
        return "contradicted"
    return "unverified"


def verify_answer(answer_text: str, source_texts: list[str], model=None, labels=None) -> list[ClaimCheck]:
    """Check every cited sentence of an answer against the sources it cites. source_texts[i] is source [i+1].

    Sentences without a citation are skipped here (the format check already flags them), as are fragments too short to
    be claims. Returns [] if no model is available."""
    if model is None:
        model, labels = load_model()
    if model is None:
        return []
    checks: list[ClaimCheck] = []
    for sentence in split_sentences(answer_text):
        cited = sorted({int(n) for n in _CITE.findall(sentence) if 1 <= int(n) <= len(source_texts)})
        claim = strip_citations(sentence)
        if not cited or len(claim.split()) < MIN_CLAIM_WORDS:
            continue
        pairs = [(joined_premise(claim, source_texts[n - 1]), claim) for n in cited]
        probs = to_probs(model.predict(pairs, show_progress_bar=False))
        entail = float(probs[:, labels["entailment"]].max())
        contra = float(probs[:, labels["contradiction"]].max())
        both = entail >= settings.nli_entail_cutoff and contra >= settings.nli_contradiction_cutoff
        checks.append(ClaimCheck(sentence.strip(), cited, round(entail, 3), round(contra, 3),
                                 verdict_for(entail, contra), both))
    return checks


def summarize(checks: list[ClaimCheck]) -> dict:
    return {
        "available": True, "claims": len(checks),
        "supported": sum(c.verdict == "supported" for c in checks),
        "unverified": sum(c.verdict == "unverified" for c in checks),
        "contradicted": sum(c.verdict == "contradicted" for c in checks),
        "conflicting": sum(c.conflict for c in checks),
        "checks": [{"sentence": c.sentence, "cited": c.cited, "entailment": c.entailment,
                    "contradiction": c.contradiction, "verdict": c.verdict, "conflict": c.conflict} for c in checks],
    }

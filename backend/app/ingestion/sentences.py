"""Sentence splitting tuned for scientific prose.

Naive `split on ". "` shreds "et al.", "Fig. 3", "e.g.", "J. Smith" and "0.5 BLEU". Chunks are packed on
sentence boundaries, so a bad splitter directly produces bad chunk edges and bad overlap.
"""
import re

# Lower-cased tokens (without the trailing period) after which a period does NOT end a sentence.
_ABBREVIATIONS = {
    "al", "e.g", "i.e", "etc", "fig", "figs", "eq", "eqs", "sec", "secs", "tab", "tbl", "ref", "refs",
    "vs", "cf", "viz", "approx", "resp", "dr", "mr", "mrs", "ms", "prof", "no", "nos", "vol", "pp", "p",
    "ch", "st", "jr", "sr", "inc", "ltd", "co", "dept", "univ", "est", "avg", "max", "min", "thm", "lem",
    "prop", "def", "cor", "alg", "ex", "eg", "ie",
}

# A boundary candidate: sentence punctuation (+ optional closing quote/bracket), whitespace, then something
# that plausibly starts a sentence.
_CANDIDATE = re.compile(r"""([.!?]["')\]]*)(\s+)(?=["'(\[]?[A-Z0-9])""")
_SINGLE_INITIAL = re.compile(r"(?:^|\s)[A-Z]$")  # "J." in "J. Smith"


def _is_boundary(text: str, punct_start: int) -> bool:
    if text[punct_start] != ".":
        return True  # ! and ? are real boundaries
    before = text[:punct_start]
    m = re.search(r"([A-Za-z.]+)$", before)
    token = m.group(1) if m else ""
    low = token.lower().rstrip(".")
    if low in _ABBREVIATIONS:
        return False
    # "J. Smith", "A. B. Author": single capital letter before the period
    if _SINGLE_INITIAL.search(before[-3:]) and len(token) == 1:
        return False
    # "e.g." / "i.e." / "U.S." style: dotted token already contains periods
    if "." in token and len(token.replace(".", "")) <= 3:
        return False
    return True


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text.strip())
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for m in _CANDIDATE.finditer(text):
        if not _is_boundary(text, m.start(1)):
            continue
        end = m.end(1)
        sentences.append(text[start:end].strip())
        start = m.end(2)
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]

"""Pure retrieval logic (no DB, no models) so it can be unit-tested."""
import re
from dataclasses import dataclass, field

_STOPWORDS = frozenset("""
a an the of in on at to for from by with without within into onto over under between among about as is are was were be been
being am do does did done doing have has had having this that these those it its they them their we our you your he she
his her i me my and or but if then than so such not no nor can could should would may might will shall what which who whom
whose when where why how there here also very more most less least much many any all each both either neither other
another some same own just only too s t via per vs
""".split())

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def lexical_terms(question: str, max_terms: int = 24) -> list[str]:
    """Content words from a natural-language question, order-preserving and de-duplicated."""
    seen, out = set(), []
    for tok in _TOKEN.findall(question.lower()):
        if tok in _STOPWORDS or (len(tok) < 2 and not tok.isdigit()) or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out[:max_terms]


def to_tsquery_or(question: str) -> str | None:
    """OR-query for Postgres `to_tsquery('english', ...)`. `websearch_to_tsquery` ANDs every term, which matches
    nothing for a full-sentence question; OR + ts_rank_cd ranking rewards passages covering more terms."""
    terms = lexical_terms(question)
    return " | ".join(terms) if terms else None


@dataclass
class Fused:
    id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # which retriever put it where (1-indexed) — for "why this?"


def rrf(rankings: dict[str, list[str]], k: int = 60, weights: dict[str, float] | None = None) -> list[Fused]:
    """Reciprocal Rank Fusion over named ranked lists of ids.

    The design doc's snippet used `1 / (rank + k)` with 0-indexed rank, i.e. k+0 for the best item. The standard
    formulation (Cormack et al. 2009) is 1 / (k + rank) with rank starting at 1.
    """
    weights = weights or {}
    fused: dict[str, Fused] = {}
    for name, ids in rankings.items():
        w = weights.get(name, 1.0)
        for rank, _id in enumerate(ids, start=1):
            f = fused.setdefault(_id, Fused(_id, 0.0))
            f.score += w / (k + rank)
            f.ranks[name] = rank
    return sorted(fused.values(), key=lambda f: (-f.score, f.id))

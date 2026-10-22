"""Split a multi-part question into its parts so each part gets its own retrieval.

"What batch size did BERT use, and what larger batch size does RoBERTa use?" asks two things that live in two
different papers. One search ranks chunks against the whole sentence, so the top-k tends to fill with one paper's
chunks and the other half of the answer never reaches the model. Searching each part separately, then interleaving
the results, guarantees every part room in the context.

Rule-based on purpose (no LLM call, deterministic, nothing to fail): a question is split only where a clause boundary
is followed by a new question word. "How do Longformer and BigBird reduce the cost of attention?" is one question and
stays one.
"""
import re

MAX_PARTS = 3
MIN_PART_WORDS = 3
_QWORD = r"(?:what|which|how|why|who|whom|when|where|does|do|did|is|are|was|were|can|could)"
# ", and what ..." / ", what ..." / "; how ..." / " and which ..." (a comma-less "and" needs a wh-word, not a verb)
_BOUNDARY = re.compile(
    rf",\s+(?:and\s+)?(?={_QWORD}\b)|;\s*(?:and\s+)?(?={_QWORD}\b)|\s+and\s+(?=(?:what|which|how|why|who|when|where)\b)",
    re.IGNORECASE,
)


def split_question(question: str) -> list[str]:
    q = " ".join(question.split())
    parts = [p.strip(" ,;") for p in _BOUNDARY.split(q)]
    parts = [p for p in parts if p]
    if len(parts) < 2 or len(parts) > MAX_PARTS or any(len(p.split()) < MIN_PART_WORDS for p in parts):
        return [question]
    return [p if p.endswith("?") else p + "?" for p in parts]


def interleave(results: list[list], limit: int, key=lambda x: x) -> list:
    """Round-robin across the parts' result lists, dropping repeats, so every part keeps its best chunks."""
    out, seen = [], set()
    for rank in range(max((len(r) for r in results), default=0)):
        for r in results:
            if rank < len(r) and key(r[rank]) not in seen:
                seen.add(key(r[rank]))
                out.append(r[rank])
    return out[:limit]

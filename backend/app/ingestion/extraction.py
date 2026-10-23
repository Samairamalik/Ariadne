"""Triple extraction: chunks -> (subject, predicate, object) facts, via the LLM router.

Chunks are sent in small batches (fewer requests, which is what free-tier limits count). The model is asked for
JSON; everything it returns is validated here, because LLM output is untrusted. Each triple is tied to the chunk it
came from by a 1-based number within the batch, which is how the graph keeps provenance (edges.chunk_id).
"""
import json
import logging
import re
from dataclasses import dataclass, replace

from app.ingestion.entity_filter import is_main_entity
from app.llm.providers import LLMError, LLMUnavailable
from app.retrieval.doc_aware import anchors

log = logging.getLogger("ariadne.extract")

BATCH_SIZE = 4
# Must match the CHECK constraint on nodes.node_type in db/schema.sql.
NODE_TYPES = ("concept", "person", "method", "dataset", "model", "paper", "other")
MAX_NAME = 120
MAX_EVIDENCE = 400

SYSTEM = """You extract facts from numbered document passages as knowledge-graph triples.

Return JSON: {"triples": [{"chunk": <passage number as an integer>, "subject": str, "subject_type": str,
"predicate": str, "object": str, "object_type": str, "evidence": str}]}

Rules:
1. subject and object are ENTITIES: short noun phrases of at most 5 words (a name, method, model, dataset, metric
   or concept). Never a sentence, a clause, a number with its explanation, or a pronoun.
2. predicate is ONE short verb or verb phrase of at most 3 words in present tense (uses, improves, is part of,
   outperforms, is trained on). Never put extra detail in the predicate.
3. Direction matters: the subject does or has the predicate to the object. "X is introduced by Y" becomes
   (Y, introduces, X).
4. Only facts the passage states. Skip vague or negated statements, and passages with no facts.
5. evidence is a short exact quote copied from the passage that supports the fact.
6. types are one of: """ + ", ".join(NODE_TYPES) + """.
7. The user message names the DOCUMENT these passages come from, with its short name. In a research paper "we",
   "our model", "our method", "the proposed approach" and "this paper" all mean THAT name: always write the name,
   never the pronoun or the generic phrase.

Examples:
Passage: "Longformer uses a sliding window of 512 tokens, which cuts attention cost from quadratic to linear."
{"chunk": 1, "subject": "Longformer", "subject_type": "model", "predicate": "uses", "object": "sliding window attention",
 "object_type": "method", "evidence": "Longformer uses a sliding window of 512 tokens"}
{"chunk": 1, "subject": "sliding window attention", "subject_type": "method", "predicate": "reduces", "object": "attention cost",
 "object_type": "concept", "evidence": "cuts attention cost from quadratic to linear"}

Passage: "BigBird was proven Turing complete by Zaheer et al."
{"chunk": 1, "subject": "Zaheer et al.", "subject_type": "person", "predicate": "proved", "object": "BigBird Turing completeness",
 "object_type": "concept", "evidence": "BigBird was proven Turing complete by Zaheer et al."}
(direction: the passive sentence is rewritten so the doer is the subject)

Bad (do not output): subject "State space models", predicate "process", object "a sequence with a recurrence that carries
a fixed-size hidden state" (object is a clause; name the entity instead: "recurrent hidden state")."""


@dataclass(frozen=True)
class Triple:
    chunk: int  # 0-based index into the batch
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    evidence: str


def document_subject(doc_title: str | None) -> str | None:
    """Short name of the paper ("BERT", "RoBERTa", "SpectralRepNet"): what "we" and "our model" refer to."""
    found = anchors(doc_title)
    return found[0] if found else None


def build_messages(texts: list[str], doc_title: str | None = None, sections: list[str | None] | None = None) -> list[dict]:
    """Numbered passages, each with its section heading, under a header naming the document and its subject."""
    lines = []
    for i, t in enumerate(texts, 1):
        sec = sections[i - 1] if sections and i - 1 < len(sections) and sections[i - 1] else None
        lines.append(f"[{i}]" + (f" (section: {sec})" if sec else "") + f" {t}")
    head = ""
    if doc_title:
        subject = document_subject(doc_title)
        head = f"DOCUMENT: {doc_title}\n" + (f"DOCUMENT NAME (what 'we' and 'our model' mean): {subject}\n" if subject else "") + "\n"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": head + "\n\n".join(lines)}]


# Phrases that, in a research paper, mean the paper's own subject. Mapped to its name after extraction as a safety net
# for when the model ignores rule 7 (a 3B model often does).
SELF_REFERENCES = frozenset({
    "we", "our", "ours", "our model", "our models", "this model", "the model", "the proposed model", "our method",
    "this method", "the proposed method", "our approach", "this approach", "the proposed approach", "our network",
    "the proposed network", "our system", "this paper", "the paper", "this work", "our work", "our framework",
    "the proposed framework", "this study", "our architecture", "it",
})


def resolve_self_references(triples: list["Triple"], subject: str | None) -> list["Triple"]:
    if not subject:
        return triples
    fix = lambda name: subject if name.lower().strip() in SELF_REFERENCES else name  # noqa: E731
    return [replace(t, subject=fix(t.subject), object=fix(t.object)) for t in triples
            if fix(t.subject).lower() != fix(t.object).lower()]


_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_GENERIC = frozenset({"the", "and", "for", "with", "that", "this", "from", "model", "models", "method", "methods",
                      "approach", "network", "paper", "new", "using", "based", "our", "its", "are", "was", "were"})


def _tokens(text: str) -> set[str]:
    """Words of a text; a hyphenated word also contributes its joined form ('re-parameterization' ->
    'reparameterization'), so spelling variants still match."""
    out: set[str] = set()
    for w in _WORD.findall(text.lower()):
        out.add(w)
        if "-" in w:
            out.update(w.split("-"))
            out.add(w.replace("-", ""))
    return out


def entity_supported(name: str, quote: str, subject: str | None = None) -> bool:
    """Lexical support: the paper's own name always counts; otherwise at least one content word of the entity must
    appear in the evidence quote. Catches a fact attached to the wrong entity ('RESIDE-6K contains 4,322 images'
    when the quote is about RTTS) that the quote-exists check cannot see."""
    if subject and name.lower().strip() == subject.lower():
        return True
    content = {t for t in _tokens(name) if t not in _GENERIC and len(t) >= 2}
    return bool(content & _tokens(quote))


def verify_triples(triples: list["Triple"], subject: str | None = None) -> list["Triple"]:
    """Drop triples whose subject or object is not stated in their own evidence quote."""
    return [t for t in triples if entity_supported(t.subject, t.evidence, subject)
            and entity_supported(t.object, t.evidence, subject)
            and is_main_entity(t.subject, t.subject_type) and is_main_entity(t.object, t.object_type)]


def _clean(v, limit: int) -> str:
    return " ".join(v.split())[:limit] if isinstance(v, str) else ""


def _type(v) -> str:
    v = v.strip().lower() if isinstance(v, str) else ""
    return v if v in NODE_TYPES else "other"


def parse_triples(raw: str, batch_len: int) -> list[Triple]:
    """Validate model output; malformed items are dropped, never raised."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    items = data.get("triples") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        n = it.get("chunk")
        if isinstance(n, str) and n.strip().isdigit():  # models often answer "1" instead of 1
            n = int(n)
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= batch_len:
            continue
        s, p, o = _clean(it.get("subject"), MAX_NAME), _clean(it.get("predicate"), MAX_NAME), _clean(it.get("object"), MAX_NAME)
        if not (s and p and o) or s.lower() == o.lower():
            continue
        out.append(Triple(n - 1, s, _type(it.get("subject_type")), p, o, _type(it.get("object_type")),
                          _clean(it.get("evidence"), MAX_EVIDENCE)))
    return out


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def attribute_to_chunks(triples: list[Triple], texts: list[str]) -> list[Triple]:
    """Re-point each triple at the chunk that really contains its evidence quote.

    Models often number the passage wrongly, which would attach a true fact to the wrong chunk and break the
    graph-to-text link. The quote is the ground truth: find it in the batch. If the model's own chunk contains it,
    keep that; otherwise use the first chunk that does; if none does, the triple is unsupported and is dropped.
    """
    normed = [_norm(t) for t in texts]
    out = []
    for tr in triples:
        ev = _norm(tr.evidence)
        if not ev:
            continue
        if ev in normed[tr.chunk]:
            out.append(tr)
            continue
        hit = next((i for i, t in enumerate(normed) if ev in t), None)
        if hit is not None:
            out.append(replace(tr, chunk=hit))
    return out


async def extract_batch(router, texts: list[str], doc_title: str | None = None,
                        sections: list[str | None] | None = None) -> list[Triple]:
    """One LLM call for one batch. Raises LLMUnavailable / LLMError; the caller decides how to degrade.

    Pipeline: parse (drop malformed) -> self-references become the paper's name -> re-attach each triple to the chunk
    that really contains its quote (drop if none) -> drop triples whose entities are not in their own quote."""
    result = await router.chat(build_messages(texts, doc_title, sections), json_mode=True, temperature=0.0)
    subject = document_subject(doc_title)
    triples = resolve_self_references(parse_triples(result.text, len(texts)), subject)
    return verify_triples(attribute_to_chunks(triples, texts), subject)


__all__ = ["BATCH_SIZE", "Triple", "build_messages", "parse_triples", "attribute_to_chunks", "extract_batch",
           "resolve_self_references", "verify_triples", "document_subject", "LLMError", "LLMUnavailable"]

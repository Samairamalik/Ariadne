"""Grounded answer generation through the LLM router (offline Ollama, or Groq/Gemini in cloud mode), with citation
validation.

The model is told to answer only from numbered sources and cite as [n]. Afterwards we verify mechanically:
  * every [n] points at a real source (invalid ones are stripped, never shown as if valid)
  * sentences with no citation are reported, so the UI can flag unsupported claims
This is a syntactic check; semantic faithfulness (NLI) is a separate Phase-5 item.
"""
import re
from dataclasses import dataclass

from app.ingestion.sentences import split_sentences
from app.llm import LLMUnavailable, get_router  # noqa: F401 (LLMUnavailable re-exported for callers)
from app.retrieval.hybrid import Candidate

SYSTEM = (
    "You answer questions strictly from the numbered SOURCES provided. Rules:\n"
    "1. Use only information stated in the sources. Do not use outside knowledge.\n"
    "2. After every sentence that states a fact, cite the supporting source(s) like [1] or [2][3].\n"
    "3. If the sources do not contain the answer, reply exactly: "
    "\"The provided documents do not contain enough information to answer this.\"\n"
    "4. If sources disagree, say so and cite each side. Be concise."
)
NO_ANSWER = "The provided documents do not contain enough information to answer this."
_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    cited: list[int]  # 1-indexed source numbers actually referenced
    invalid_citations: list[int]
    uncited_sentences: list[str]
    refused: bool
    provider: str | None = None  # which LLM actually wrote this (ollama / groq / gemini) — shown to the user
    model: str | None = None


GRAPH_ADDENDUM = (
    "\n5. GRAPH FACTS list relations extracted from the sources; each ends with the number of the source it came "
    "from. Use them to connect information that sits in different sources, and answer every part of the question. "
    "Stay strictly on the question: ignore any fact or source that does not help answer it, and do not add "
    "background. A fact is a pointer, not evidence: only state what the source text itself supports, and cite "
    "sources [n], never facts."
)
MAX_FACTS = 25
INTENT_ADDENDA = {
    "path": ("\n8. This question asks how two things are connected. GRAPH FACTS are the chain of facts that links them, "
             "in order. Explain the connection step by step along that chain, citing the source for each step. Do not "
             "claim a link the chain does not show."),
    "aggregate": ("\n8. This question asks WHICH documents say something. Answer with one short line per document whose "
                  "sources support it, naming the document and citing its source. List only documents the sources support."),
    "profile": ("\n8. This question asks for everything known about one subject. Summarise what the sources say about it, "
                "grouped by document, citing each statement. Do not add anything the sources do not state."),
}
FOCUS_ADDENDUM = (
    "\n7. The question names specific document(s), given as FOCUS. Their sources come first. Take facts about the "
    "named subject from those sources. A source from a different document that merely discusses the subject (for "
    "example a follow-up paper describing a modified version) must not be presented as the subject's own property; "
    "use it only for comparisons and say which document it comes from."
)


OVERVIEW_ADDENDUM = (
    "\n6. An OVERVIEW may precede the sources: machine-written background on themes in the documents. Use it to "
    "organize the answer, never cite it, and state nothing from it that the numbered sources do not support."
)


def build_prompt(question: str, sources: list[Candidate], facts=None, overview=None, focus=None, parts=None) -> str:
    blocks = []
    for i, c in enumerate(sources, start=1):
        loc = f"{c.doc_title}" + (f" — {c.section_path}" if c.section_path else "") + f" (p.{c.page_start})"
        blocks.append(f"[{i}] {loc}\n{c.text}")
    out = ""
    if focus:
        out += "FOCUS (the question is about): " + "; ".join(focus) + "\n\n"
    if overview:
        out += "OVERVIEW (background, do not cite):\n" + "\n".join(f"- {t}: {s}" for t, s in overview) + "\n\n"
    out += "SOURCES:\n\n" + "\n\n".join(blocks)
    if facts:
        num = {c.chunk_id: i for i, c in enumerate(sources, start=1)}
        lines = [f"- ({f.subject}) -[{f.predicate}]-> ({f.object})  [{num[f.chunk_id]}]"
                 for f in sorted(facts, key=lambda f: f.hop)[:MAX_FACTS] if f.chunk_id in num]
        if lines:
            out += "\n\nGRAPH FACTS:\n" + "\n".join(lines)
    if parts:
        question += "\nThis question has parts; answer each one: " + " ".join(f"({i}) {p}" for i, p in enumerate(parts, 1))
    return out + f"\n\nQUESTION: {question}\n\nANSWER (with [n] citations):"


def validate(raw: str, n_sources: int) -> Answer:
    text = raw.strip()
    refused = NO_ANSWER.lower() in text.lower()
    nums = [int(m) for m in _CITE.findall(text)]
    invalid = sorted({n for n in nums if not 1 <= n <= n_sources})
    text = _CITE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", text)
    cited = sorted({n for n in nums if 1 <= n <= n_sources})
    uncited = [] if refused else [s for s in split_sentences(text) if not _CITE.search(s)]
    return Answer(text, cited, invalid, uncited, refused)


GENERAL_SYSTEM = (
    "You answer from general knowledge only. You have NO access to the user's documents and must not imply you do. "
    "Be concise and say so if you are unsure."
)
GENERAL_PREFIX = "Not from your documents: "


async def generate_general(question: str) -> Answer:
    """Opt-in fallback for questions the documents do not cover. Always marked, never cited."""
    res = await get_router().chat([
        {"role": "system", "content": GENERAL_SYSTEM},
        {"role": "user", "content": question},
    ])
    return Answer(GENERAL_PREFIX + res.text.strip(), [], [], [], False, res.provider, res.model)


async def generate_answer(question: str, sources: list[Candidate], facts=None, overview=None, focus=None,
                          parts=None, intent=None) -> Answer:
    if not sources:
        return Answer(NO_ANSWER, [], [], [], True)
    res = await get_router().chat([
        {"role": "system", "content": SYSTEM + (GRAPH_ADDENDUM if facts else "") + (OVERVIEW_ADDENDUM if overview else "") + (FOCUS_ADDENDUM if focus else "") + INTENT_ADDENDA.get(intent, "")},
        {"role": "user", "content": build_prompt(question, sources, facts, overview, focus, parts)},
    ])
    ans = validate(res.text, len(sources))
    ans.provider, ans.model = res.provider, res.model
    return ans

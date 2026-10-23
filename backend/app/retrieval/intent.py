"""What kind of graph question is this? Rule-based on purpose: no model call, deterministic, easy to audit.

  path       "How is BERT related to the Transformer paper?"      -> shortest chain of facts between two entities
  aggregate  "Which papers in the library train with Adam?"       -> one line per paper, each with its quote
  profile    "Tell me everything the library says about RepVGG."  -> everything known about one entity, by paper
  chunk      anything else                                        -> ordinary retrieval (with or without the graph)

Only these three shapes are routed, because they are the ones plain chunk retrieval cannot answer: they need facts
gathered from many places, not the few passages most similar to the question.
"""
import re

_PATH = re.compile(
    r"\b(?:how\s+(?:is|are|does|do)\b.+\b(?:related|connected|linked|relate|connect|link)\b"
    r"|(?:relationship|connection|link|relation)s?\s+between\b)", re.I)
_PROFILE = re.compile(
    r"\b(?:tell me everything|summari[sz]e everything|everything\s+(?:that\s+)?(?:the\s+)?(?:library|documents?|papers?|corpus)\s+"
    r"(?:says?|knows?|contains?)|what do(?:es)? (?:my |the )?(?:library|corpus|documents?|papers?) say|all (?:the )?(?:facts|information) (?:about|on))\b", re.I)
_AGGREGATE = re.compile(
    r"^\s*(?:which|what)\s+(?:papers?|documents?|sources?)\b.*\b"
    r"(?:use|uses|used|using|train|trains|trained|report|reports|discuss|discusses|mention|mentions|describe|describes|"
    r"evaluate|evaluates|include|includes|introduce|introduces|employ|employs)\b", re.I)


def classify_intent(question: str) -> str:
    q = " ".join(question.split())
    if _PROFILE.search(q):
        return "profile"
    if _AGGREGATE.search(q):
        return "aggregate"
    if _PATH.search(q):
        return "path"
    return "chunk"

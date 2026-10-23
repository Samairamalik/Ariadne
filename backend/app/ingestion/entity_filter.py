"""Which names deserve to be a node in the graph.

The graph should hold the main things a paper is about: models, methods, datasets, metrics, people, named concepts.
Language models also emit fragments ("inference speed by 19.7%", "to prevent positions from attending ...", "1024"),
filler nouns ("size", "formula", "recent work") and document structure ("Table 2", "Related Work"). `is_main_entity`
is a rule-based gate on the NAME alone, so it works at extraction time (before a fact is stored) and can clean an
existing graph.
"""
import re

MAX_WORDS = 5
MAX_CHARS = 60

# a name cannot start with one of these: it is then a clause or a modifier, not an entity
_BAD_START = frozenset("""the a an to each all every very recent similar as jointly both of in on for with by from that
this these those our their its per than into over under between using based such other another same some any many more
most less few several various multiple one two three four five six seven eight nine ten""".split())
# a name with two or more of these inside is a sentence fragment ("Effect of Number of Training Steps")
_GLUE = frozenset("""of that to with for and on by in than which where while when is are was were be can lose during via
alongside across through at into from using""".split())
# a lowercase name (no capital/digit word) ending in one of these describes a result, not a topic
_FILLER_TAIL = frozenset("""results result techniques ingredients measurements costs cost gain gains growth count counts
speedup improvements improvement properties""".split())
_STRUCTURE = re.compile(r"^(table|figure|fig\.?|section|appendix|algorithm|equation|eq\.?|chapter|part)\b|"
                        r"^(related work|introduction|conclusion|conclusions|abstract|background|discussion|"
                        r"future work|references|acknowledg(e)?ments|experiments?|results)$", re.I)
# numbers with units / amounts: facts about a thing, not a thing
_QUANTITY = re.compile(r"(?<![A-Za-z-])\d[\d.,]*\s*(%|m\b|k\b|b\b|gflops|flops|db\b|ms\b|fps|x\b|×|million|billion|percent|points?\b|"
                       r"parameters|tokens|epochs|steps|images|sentences|layers|hours|days|gb\b|mb\b)|"
                       r"(by|of|to|at)\s+\d", re.I)
_CITATION = re.compile(r"\bet al\b|\(\s*\w[^)]*\d{4}\s*\)|\(\d{4}\)", re.I)
# single lowercase words that are filler, not topics
_FILLER = frozenset("""size formula formulas visualization visualizations performance result results accuracy speed cost
memory input inputs output outputs layer layers data task tasks number numbers value values example examples way ways
approach approaches work works paper papers model models method methods system systems problem problems idea ideas
operation operations step steps time part parts case cases type types set sets level levels one two three four five
thing things form forms rate rates change changes effect effects difference differences improvement improvements
property properties matrix feature features representation representations sequence sequences function functions
position positions token tokens word words sentence sentences pair pairs symbol symbols""".split())
_FILLER_PHRASES = frozenset({"recent work", "state of the art", "state-of-the-art", "previous work", "prior work",
                             "future work", "this work", "our work", "our model", "our method", "the model",
                             "base model", "memory occupation", "state-of-the-art performance"})


def _proper(word: str) -> bool:
    """Looks like a name: has a capital letter or a digit (BERT, RoBERTa, GLUE, SQuAD2, ResNet-50)."""
    return any(ch.isupper() or ch.isdigit() for ch in word)


def is_main_entity(name: str, node_type: str = "") -> bool:
    name = " ".join((name or "").split())
    if not name or len(name) > MAX_CHARS:
        return False
    words = name.split()
    if len(words) > MAX_WORDS:
        return False
    low = name.lower()
    if low in _FILLER_PHRASES or _STRUCTURE.search(name) or _CITATION.search(name):
        return False
    if not any(ch.isalpha() for ch in name) or any(ch in name for ch in "=∗*{}"):   # "1024", "S = 1,000,000 updates"
        return False
    if _QUANTITY.search(name):                                     # "0.38M parameters", "inference speed by 19.7%"
        return False
    if words[0].lower() in _BAD_START and node_type != "paper":    # clauses, modifiers
        return False
    proper = any(_proper(w) for w in words)
    glue = sum(1 for w in words[1:-1] if w.lower() in _GLUE)
    if node_type != "paper" and (glue >= 2 or (glue >= 1 and not proper)):
        return False
    if not proper and words[-1].lower() in _FILLER_TAIL:
        return False
    if words[-1].lower() in _GLUE or words[-1].lower() in {"is", "are", "can"}:   # dangling clause
        return False
    if len(words) == 1 and not _proper(words[0]) and "-" not in words[0] and low in _FILLER:
        return False
    if len(words) == 1 and not _proper(words[0]) and len(words[0]) <= 2:
        return False
    if all(w.lower() in _FILLER or w.lower() in _BAD_START or w.lower() in _GLUE for w in words):
        return False
    return True


KEEP_ONE_MENTION_TYPES = frozenset({"model", "dataset", "person", "paper", "method"})


def is_core_node(name: str, node_type: str, mention_count: int) -> bool:
    """The stricter test used to clean a stored graph: a plausible entity name AND either mentioned in more than one
    fact, or a short proper name (a model, dataset, person, paper or method named with capitals/digits). A one-off
    lowercase phrase is almost always a description, not a topic."""
    if not is_main_entity(name, node_type):
        return False
    if mention_count >= 2:
        return True
    words = name.split()
    return node_type in KEEP_ONE_MENTION_TYPES and len(words) <= 3 and any(_proper(w) for w in words)

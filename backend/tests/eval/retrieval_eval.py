"""Retrieval evaluation against a running API. Gold-answer based: a question is answered at rank r if the r-th
returned chunk contains ALL gold substrings. Matching on text (not chunk ids) keeps the eval valid when the
chunker changes, which is exactly when you need to re-run it.

    python tests/eval/retrieval_eval.py http://localhost:8000 [--no-print-misses] [--no-rerank]

Questions are written against the two papers used while tuning the parser (BERT; Lyu et al. IEEE T-ITS 2023).
Add more as the corpus grows; keep the multi-hop set separate once the graph path exists.
"""
import json
import re
import sys

import httpx

BASE = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8000")

# (question, [gold substrings that must co-occur in one chunk], tag)
QUESTIONS = [
    # ── BERT: facts in prose
    ("What percentage of WordPiece tokens does BERT mask in each sequence?", ["mask 15% of all WordPiece tokens"], "prose"),
    ("How many parameters does BERT-large have?", ["Total Parameters=340M"], "prose"),  # alt answer chunk: see ALT_GOLD
    ("What is the size of the WordPiece vocabulary BERT uses?", ["30,000 token vocabulary"], "prose"),
    ("How long did pre-training BERT take on Cloud TPUs?", ["Each pretraining took 4 days"], "prose"),
    ("What learning rate and optimizer were used for pre-training?", ["Adam with learning rate of 1e-4"], "prose"),
    ("What does the final hidden state of the [CLS] token represent?", ["aggregate sequence representation for classification"], "prose"),
    ("Why is the [MASK] token sometimes replaced by a random word or kept unchanged?", ["[MASK] token does not appear during fine-tuning"], "paraphrase"),
    ("How were sentences A and B chosen for next sentence prediction?", ["50% of the time B is the actual next sentence"], "prose"),
    ("Which corpora was BERT pre-trained on?", ["BooksCorpus (800M words)", "English Wikipedia (2,500M words)"], "prose"),
    ("Why can't a deep bidirectional model simply be trained like a left-to-right language model?", ["see itself"], "paraphrase"),
    # ── BERT: footnotes and tables (the structures the parser had to rebuild)
    ("What accuracy does the final BERT model reach on the next sentence prediction task?", ["97%-98% accuracy on NSP"], "footnote"),
    ("What was BERT-large's MNLI accuracy on the GLUE test set?", ["BERTLARGE | 86.7/85.9"], "table"),
    ("What does removing next sentence prediction do to MNLI accuracy in the ablation?", ["No NSP | 83.9"], "table"),
    ("What are the SQuAD 1.1 test F1 results for BERT-large single model with TriviaQA?", ["BERTLARGE (Sgl.+TriviaQA)", "91.8"], "table"),
    ("What are BERT-base's SWAG dev accuracy and BERT-large's test accuracy?", ["BERTLARGE | 86.6 86.3"], "table"),
    ("Which masking strategy gives the best MNLI accuracy in the masking ablation?", ["80% | 20% | 0% | 84.4"], "table"),
    # ── IEEE dehazing paper
    ("How many frames is the large update cycle for atmospheric light?", ["6000 frames"], "prose"),
    ("Which object detector is used to evaluate the benefit of dehazing?", ["Yolov5s"], "prose"),
    ("By how much does the method improve detection accuracy?", ["improved by 4.11%"], "prose"),
    ("What does the inverted triangle describe in driving scenes?", ["compositional characteristics of the sky region"], "paraphrase"),
    ("What patch size is used for the dark channel?", ["the fixed value is 15"], "prose"),
    ("Which datasets are used for the experiments?", ["Foggy_CityScapes dataset", "RTTS"], "prose"),
    ("Why is the image dark when atmospheric light is estimated too high?", ["the entire image becomes dark when A too high"], "paraphrase"),
    ("How does the scattering model relate the foggy image to the clean image?", ["I (x) = J(x)t(x) + A(1 −t(x))"], "equation"),
]


# Other chunks that legitimately answer the same question (so a correct-but-different passage isn't scored a miss).
ALT_GOLD = {
    "How many parameters does BERT-large have?": [["BERTLARGE contains 340M parameters"]],
}


def norm(s: str) -> str:
    # hyphens are ignored: "pre-training" vs "pretraining" depends on typesetter line breaks, not on retrieval
    return re.sub(r"\s+", " ", s.replace("-", "")).lower()


def main() -> int:
    show_misses = "--no-print-misses" not in sys.argv
    ranks, rows = [], []
    reranked = None
    with httpx.Client(base_url=BASE, timeout=120) as c:
        for q, gold, tag in QUESTIONS:
            res = c.post("/api/search", json={"question": q, "top_k": 10, "rerank": "--no-rerank" not in sys.argv}).json()
            reranked = res["timing"]["reranked"]
            options = [gold] + ALT_GOLD.get(q, [])
            rank = next((i for i, r in enumerate(res["results"], 1)
                         if any(all(norm(g) in norm(r["text"]) for g in opt) for opt in options)), None)
            ranks.append(rank)
            rows.append((tag, q, rank, res["results"][0]["section_path"] if res["results"] else ""))

    n = len(ranks)
    hit = lambda k: sum(1 for r in ranks if r is not None and r <= k)  # noqa: E731
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"reranker applied: {reranked}")
    print(f"hit@1 {hit(1)}/{n}   hit@3 {hit(3)}/{n}   hit@5 {hit(5)}/{n}   hit@10 {hit(10)}/{n}   MRR {mrr:.3f}\n")
    by_tag: dict[str, list] = {}
    for tag, _, rank, _ in rows:
        by_tag.setdefault(tag, []).append(rank)
    for tag, rs in by_tag.items():
        print(f"  {tag:10} hit@1 {sum(1 for r in rs if r == 1)}/{len(rs)}  hit@5 {sum(1 for r in rs if r and r <= 5)}/{len(rs)}")
    if show_misses:
        print("\nnot at rank 1:")
        for tag, q, rank, top in rows:
            if rank != 1:
                print(f"  [{tag}] rank={rank}  {q}\n        top-1 section: {top!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

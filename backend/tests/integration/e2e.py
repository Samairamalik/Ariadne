"""End-to-end check against a running API + Postgres (not collected by pytest; run manually):

    python tests/integration/e2e.py http://localhost:8000

Builds a realistic PDF, uploads it, waits for ingestion over SSE, prints chunking stats, and verifies that
each question's expected section is retrieved at rank 1 (and in the top 3).
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pymupdf

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

TITLE = "Efficient Attention Mechanisms: A Short Survey"
SECTIONS = [
    (1, "1 Introduction", [
        "Self-attention lets every token in a sequence attend to every other token, which gives Transformers their "
        "modelling power but also makes compute and memory grow quadratically with sequence length n. For a "
        "document of 64,000 tokens the attention matrix alone holds more than four billion entries per head. This "
        "survey reviews three families of remedies: sparse attention, linear attention, and state space models.",
    ]),
    (1, "2 Sparse Attention", [
        "Sparse attention restricts each token to a small subset of positions so that the number of computed "
        "scores grows linearly rather than quadratically. The patterns differ in how the subset is chosen.",
    ]),
    (2, "2.1 Sliding Window", [
        "Longformer uses a sliding window pattern in which each token attends only to its neighbours within a "
        "window of 512 tokens. A handful of designated global tokens, such as the classification token, attend to "
        "the whole sequence and are attended to by every position. Stacking layers lets information propagate "
        "beyond the window, so the effective receptive field grows with depth.",
    ]),
    (2, "2.2 Block-Sparse and Random Patterns", [
        "BigBird combines three ingredients: a sliding window, a set of global tokens, and randomly chosen "
        "connections between positions. The authors proved that this sparse mechanism is a universal "
        "approximator of sequence functions and that it is Turing complete, so it keeps the expressive power of "
        "full attention while using only linear memory.",
    ]),
    (1, "3 Linear Attention", [
        "Linear attention replaces the softmax with a kernel feature map so the product of queries and keys can be "
        "reassociated and computed in time linear in n. Performer approximates the softmax kernel with positive "
        "orthogonal random features, a method it calls FAVOR+, which gives an unbiased estimate of the attention "
        "matrix without ever materialising it. The approximation error depends on the number of random features.",
    ]),
    (1, "4 State Space Models", [
        "State space models process a sequence with a recurrence that carries a fixed-size hidden state, so cost "
        "scales linearly with length. Mamba makes the state space parameters functions of the input, a design it "
        "calls selective state spaces, which lets the model decide what to remember or forget at each step. "
        "The authors report roughly five times higher inference throughput than a comparable Transformer.",
    ]),
    (1, "5 Conclusion", [
        "No single mechanism dominates. Sparse patterns are simple and hardware friendly, linear attention is "
        "attractive for very long inputs but can lose accuracy on tasks that need precise recall, and selective "
        "state space models currently offer the best throughput at long context lengths.",
    ]),
]

QUESTIONS = [  # (question, expected section substring) — several deliberately paraphrased, sharing no keywords
    ("What window size does Longformer use?", "2.1 Sliding Window"),
    ("Which method was shown to be Turing complete while using linear memory?", "2.2"),
    ("How does Performer approximate the softmax kernel?", "3 Linear Attention"),
    ("What lets Mamba choose what information to keep or discard at each step?", "4 State Space"),
    ("Why is full self-attention too expensive for very long documents?", "1 Introduction"),
    ("FAVOR+", "3 Linear Attention"),  # pure rare-keyword query: lexical path must carry it
]


def build_pdf(path: Path) -> None:
    d = pymupdf.open()
    state = {"page": None, "y": 0}

    def new_page():
        state["page"], state["y"] = d.new_page(), 70
        state["page"].insert_text((57, 30), "Survey Draft", fontsize=8)
        state["page"].insert_text((290, 820), str(len(d)), fontsize=8)

    new_page()
    state["page"].insert_text((57, state["y"]), TITLE, fontsize=20, fontname="hebo")
    state["y"] += 40
    for level, heading, paras in SECTIONS:
        if state["y"] > 700:
            new_page()
        state["page"].insert_text((57, state["y"] + 12), heading, fontsize=13 if level == 1 else 11, fontname="hebo")
        state["y"] += 26
        for para in paras:
            h = 14 * (len(para) // 85 + 2)
            if state["y"] + h > 780:
                new_page()
            state["page"].insert_textbox(pymupdf.Rect(57, state["y"], 540, state["y"] + h), para, fontsize=10, fontname="helv")
            state["y"] += h + 6
    new_page()
    state["page"].insert_text((57, 90), "References", fontsize=13, fontname="hebo")
    state["page"].insert_textbox(pymupdf.Rect(57, 100, 540, 200),
                                 "[1] Beltagy et al. Longformer: the long-document transformer. 2020. "
                                 "[2] Zaheer et al. Big Bird: transformers for longer sequences. 2020. "
                                 "[3] Gu and Dao. Mamba: linear-time sequence modeling with selective state spaces. 2023.",
                                 fontsize=10, fontname="helv")
    d.save(path)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "survey.pdf"
        build_pdf(pdf)
        with httpx.Client(base_url=BASE, timeout=600) as c:
            up = c.post("/api/documents/upload", files={"file": ("survey.pdf", pdf.read_bytes(), "application/pdf")}).json()
            print("upload:", up)
            with c.stream("GET", f"/api/jobs/{up['job_id']}/status") as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        print("  progress:", line[5:].strip())
            doc = next(d for d in c.get("/api/documents").json() if d["id"] == up["document_id"])
            print("document:", {k: doc[k] for k in ("title", "status", "page_count", "chunk_count", "error")})
            if doc["status"] != "done":
                return 1
            info = c.get(f"/api/documents/{up['document_id']}/chunks").json()
            print("chunk stats:", info["stats"]["chunking"])
            for ch in info["chunks"]:
                print(f"  #{ch['chunk_index']:>2} p{ch['page_start']} {ch['token_count']:>3}t  {ch['section_path']!r}")

            top1 = top3 = 0
            for q, want in QUESTIONS:
                t = time.perf_counter()
                res = c.post("/api/search", json={"question": q, "top_k": 3}).json()
                paths = [s["section_path"] for s in res["results"]]
                ok1, ok3 = want in paths[0], any(want in p for p in paths)
                top1 += ok1; top3 += ok3
                print(f"{'OK ' if ok1 else ('~3 ' if ok3 else 'MISS')} {q!r} -> {paths[0]!r} "
                      f"found_by={res['results'][0]['found_by']} ({int((time.perf_counter()-t)*1000)} ms)")
            print(f"\nrank-1: {top1}/{len(QUESTIONS)}   top-3: {top3}/{len(QUESTIONS)}")
            return 0 if top3 == len(QUESTIONS) else 2


if __name__ == "__main__":
    sys.exit(main())

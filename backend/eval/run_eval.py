"""Step 11: measure the system, with and without the graph.

    python eval/run_eval.py [http://localhost:8100]

For every question in eval/questions.json it asks /api/query twice (use_graph false, then true) and reports:
  gate        - unrelated questions are refused, covered ones are not
  evidence    - every `evidence` quote appears in the sources the answer was given (retrieval recall)
  keywords    - every keyword group (any-of) appears in the answer text (answer correctness, crude)
  citations   - no invalid [n] and at least one valid citation on answered questions
Questions marked "split": "heldout" were written BEFORE the document-aware / spelling-merge / graph-filter changes
and were never used to tune them; compare the two splits to see how much of an improvement is real.
Keyword matching is a cheap proxy: a wrong answer can still contain the keyword, so read the answers for failures.
"""
import json
import sys
import time
from pathlib import Path

import httpx

import os
VERIFY = os.environ.get("EVAL_VERIFY", "1") == "1"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
SPLITS = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
if SPLITS:
    QUESTIONS = [q for q in QUESTIONS if q.get("split", "dev") in SPLITS]


def judge(q: dict, r: dict) -> dict:
    refused = r["source"] == "none"
    if q["type"] == "unrelated":
        return {"gate": refused, "evidence": None, "keywords": None, "citations": None}
    text = " ".join(s["text"] for s in r["sources"]).lower()
    ans = (r["answer"] or "").lower()
    return {
        "gate": not refused,
        "evidence": all(e.lower() in text for e in q["evidence"]) if q["evidence"] else None,
        "keywords": all(any(k in ans for k in group) for group in q["keywords"]) if not refused else False,
        "citations": (not r["invalid_citations"] and bool(r["cited"])) if not refused else False,
    }


def report(results: list[dict]) -> None:
    for split in ("dev", "heldout", "heldout2", "graph"):
        rs_split = [x for x in results if x["split"] == split]
        if not rs_split:
            continue
        print(f"\n{'=' * 20} {split.upper()} SET ({len(rs_split) // 2} questions) {'=' * 20}")
        print("PASS RATE by question type (pass/total)   plain -> graph")
        for metric in ("gate", "evidence", "keywords", "citations"):
            print(f"\n{metric}")
            for typ in ("lookup", "multi", "cross", "graph", "global", "unrelated"):
                row = {}
                for g in (False, True):
                    vals = [x["judged"][metric] for x in rs_split if x["type"] == typ and x["graph"] == g
                            and x["judged"][metric] is not None]
                    row[g] = f"{sum(vals)}/{len(vals)}" if vals else "-"
                if row[False] != "-":
                    print(f"  {typ:10s} {row[False]:>6s} -> {row[True]:>6s}")
        print("\nFULLY CORRECT (every applicable check passes)")
        for g in (False, True):
            rs = [x for x in rs_split if x["graph"] == g]
            ok = sum(all(v is not False for v in x["judged"].values()) for x in rs)
            print(f"  graph={g!s:5s} {ok}/{len(rs)}")
        print("\nNLI CLAIM CHECK (cited sentences: supported / unverified / contradicted)")
        for g in (False, True):
            tot = {k: sum(x.get("nli", {}).get(k, 0) for x in rs_split if x["graph"] == g)
                   for k in ("claims", "supported", "unverified", "contradicted")}
            share = tot["supported"] / tot["claims"] if tot["claims"] else 0
            print(f"  graph={g!s:5s} {tot['supported']}/{tot['unverified']}/{tot['contradicted']} of {tot['claims']} claims ({share:.0%} supported)")
        print("\nCONTEXT SIZE / LATENCY (mean, excluding unrelated)")
        for g in (False, True):
            rs = [x for x in rs_split if x["graph"] == g and x["type"] != "unrelated"]
            print(f"  graph={g!s:5s} sources={sum(x['sources'] for x in rs)/len(rs):.1f} "
                  f"docs={sum(x['docs'] for x in rs)/len(rs):.1f} seconds={sum(x['seconds'] for x in rs)/len(rs):.1f}")

    print("\nFAILURES")
    for x in results:
        bad = [m for m, v in x["judged"].items() if v is False]
        if bad:
            print(f"  [{x['split']}] {x['id']:20s} graph={x['graph']!s:5s} failed={bad}")


def ask(c: httpx.Client, question: str, use_graph: bool, tries: int = 2) -> dict:
    """One /api/query call; a slow or failed request is retried once, then recorded as an error, never fatal."""
    last = None
    for _ in range(tries):
        try:
            return c.post("/api/query", json={"question": question, "use_graph": use_graph, "verify_claims": VERIFY}).json()
        except httpx.HTTPError as e:
            last = e
    return {"error": f"{type(last).__name__}: {last}"}


def main() -> None:
    results = []
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    partial = out / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with httpx.Client(base_url=BASE, timeout=240) as c:  # a healthy answer takes ~5 s; 240 s means something hung
        for q in QUESTIONS:
            for use_graph in (False, True):
                t0 = time.perf_counter()
                r = ask(c, q["question"], use_graph)
                if "error" in r:
                    print(f"  ERROR {q['id']} graph={use_graph}: {r['error']}", flush=True)
                    results.append({"id": q["id"], "type": q["type"], "split": q.get("split", "dev"), "graph": use_graph,
                                    "judged": {"gate": False, "evidence": False, "keywords": False, "citations": False},
                                    "sources": 0, "docs": 0, "seconds": round(time.perf_counter() - t0, 1),
                                    "answer": None, "error": r["error"]})
                else:
                    results.append({"id": q["id"], "type": q["type"], "split": q.get("split", "dev"), "graph": use_graph,
                                    "judged": judge(q, r), "sources": len(r["sources"]),
                                    "docs": len({s["document_id"] for s in r["sources"]}),
                                    "seconds": round(time.perf_counter() - t0, 1), "answer": r["answer"],
                                    "nli": {k: r.get("verification", {}).get(k, 0)
                                            for k in ("claims", "supported", "unverified", "contradicted", "conflicting")}})
                    print(f"  done {q['id']} graph={use_graph}", flush=True)
                partial.write_text(json.dumps(results, indent=1), encoding="utf-8")  # a crash keeps what was measured
    report(results)
    errors = [x for x in results if "error" in x]
    if errors:
        print(f"\n{len(errors)} request(s) errored and count as failures: {[x['id'] for x in errors]}")
    print("\nsaved", partial)


main()

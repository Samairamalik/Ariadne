# Ariadne

**A GraphRAG question-answering system for your own documents. It answers with citations, shows the evidence behind
every answer, checks its own claims, and lets you explore your documents as a living 3D knowledge graph. It runs fully
offline on a laptop, or online through Groq and Gemini.**

> Named for the thread that lets you find your way through a labyrinth. A pile of PDFs is a labyrinth; Ariadne is the
> thread.

This file is the single reference for the whole project: what it does, how it works, why graph retrieval beats plain
retrieval, every feature, what was measured (with the real numbers, including the unflattering ones), what does not
work yet, and where it goes next.

**How to read the status labels used throughout**

| Label | Meaning |
|---|---|
| **Verified** | I ran it and saw the result. |
| **Measured** | There is a number from an evaluation or experiment, reported below. |
| **Written, not verified** | The code exists and parses, but it has not been exercised end to end. |
| **Planned** | Not built yet. |

---

## Contents

1. [What Ariadne is](#1-what-ariadne-is)
2. [Why this is not "normal RAG"](#2-why-this-is-not-normal-rag)
3. [Why GraphRAG is better, and where it is not](#3-why-graphrag-is-better-and-where-it-is-not)
4. [Architecture at a glance](#4-architecture-at-a-glance)
5. [Ingestion: from file to searchable chunks](#5-ingestion-from-file-to-searchable-chunks)
6. [Retrieval: finding the right passages](#6-retrieval-finding-the-right-passages)
7. [The knowledge graph](#7-the-knowledge-graph)
8. [Graph-augmented answering](#8-graph-augmented-answering)
9. [Structured questions: profile, path, aggregate](#9-structured-questions-profile-path-aggregate)
10. [Ask with AI: grounded generation and citations](#10-ask-with-ai-grounded-generation-and-citations)
11. [NLI claim verification](#11-nli-claim-verification)
12. [Offline and online modes](#12-offline-and-online-modes)
13. [The frontend: a living knowledge graph](#13-the-frontend-a-living-knowledge-graph)
14. [HTTP API reference](#14-http-api-reference)
15. [Configuration reference](#15-configuration-reference)
16. [Evaluation: what was measured](#16-evaluation-what-was-measured)
17. [Reliability and engineering decisions](#17-reliability-and-engineering-decisions)
18. [Known limitations and failures](#18-known-limitations-and-failures)
19. [Future work](#19-future-work)
20. [Running it](#20-running-it)
21. [Repository layout](#21-repository-layout)
22. [Glossary](#22-glossary)

---

## 1. What Ariadne is

You upload documents (research papers, reports, notes). Ariadne:

1. **Reads** them, splits them into sentence-aware chunks, and embeds the chunks.
2. **Builds a knowledge graph** from them: entities (models, methods, datasets, people, concepts) connected by facts,
   every fact tied to the exact quote and chunk it came from.
3. **Answers questions** by combining ordinary search with a walk across the graph, so it can connect information that
   sits in different paragraphs, or different papers.
4. **Cites everything.** Each statement in an answer points at a numbered source; invalid citations are stripped.
5. **Checks its own claims** with a separate language-inference model and marks each cited sentence as supported,
   unverified, or contradicted.
6. **Refuses honestly.** When your documents do not contain the answer it says so, and only offers an outside-knowledge
   answer if you ask for one, clearly marked as not from your documents.
7. **Shows the graph** as an interactive 3D visualization you can orbit, filter, search and trace paths through.

It is built to run **entirely on one laptop with no internet** (local models through Ollama), and can switch to cloud
models (Groq, Gemini) for speed and quality with one click.

### The stack

| Layer | Technology |
|---|---|
| API | FastAPI (Python 3.11), async |
| Database | PostgreSQL 16 + pgvector (HNSW), in Docker, host port 5433 |
| Embeddings | `BAAI/bge-small-en-v1.5` (384 dimensions), asymmetric query/passage encoding |
| Reranker | cross-encoder `ms-marco-MiniLM-L-6-v2` |
| Claim checker | cross-encoder NLI `nli-deberta-v3-base`, in its own worker process |
| Local LLM | Ollama, default `granite4:micro-h` (3.2B parameters) |
| Cloud LLMs | Groq (`openai/gpt-oss-20b`), Gemini (`gemini-3.1-flash-lite-preview`) |
| Graph algorithms | NetworkX (Louvain communities, personalized PageRank) |
| Frontend | React, Vite, TypeScript, `react-force-graph-3d`, three.js |

---

## 2. Why this is not "normal RAG"

Normal (naive) RAG is: split documents into chunks, embed them, find the chunks nearest to the question, paste them
into a prompt. It is simple and often good enough. It also has predictable weaknesses, and Ariadne is designed around
each of them.

| Weakness of plain RAG | What Ariadne does about it |
|---|---|
| **Only finds passages that look like the question.** A fact in a paragraph with different wording is missed. | Hybrid search: meaning-based (dense vectors) **and** word-based (true BM25), merged with reciprocal rank fusion, then reranked by a cross-encoder. |
| **Cannot connect facts across passages or papers.** "How does A relate to B?" needs two separate passages. | A knowledge graph. Questions are linked to entities, the graph is walked up to 2 hops, and the chunks those facts came from are added as evidence. |
| **Answers confidently when the documents say nothing.** | A relevance gate. If no retrieved passage clears a calibrated cross-encoder threshold, the answer is "The provided documents do not contain enough information", never a guess. |
| **Mixes up documents.** "What is the batch size in RoBERTa?" returns BERT's paragraph. | Document-aware retrieval: a question that names a paper prefers that paper's chunks. |
| **Multi-part questions get half an answer.** | Rule-based question decomposition: each part is retrieved separately, then merged. |
| **You cannot see why it said what it said.** | Citations, an evidence trail (fact, quote, paper, section, source number), and the retrieved graph facts are all returned. |
| **Citations can be fake.** | Mechanical validation: every `[n]` must point at a real source, uncited sentences are reported. |
| **A citation can be real but irrelevant.** | NLI claim verification: each cited sentence is checked against the text it cites. |
| **"Which papers say X?" and "everything about X" need aggregation, not nearest-neighbour search.** | Structured question types answered from the graph across all papers. |
| **Vector search alone is a black box.** | Every result records which retrievers found it and at what rank (`found_by`). |
| **It is a text box.** | A 3D living knowledge graph of your library, with the answer's entities highlighted in it. |
| **Locked to one provider.** | Offline (Ollama) and online (Groq, Gemini) modes, an ordered fallback chain, and a visible indicator of whether text leaves the machine. |

---

## 3. Why GraphRAG is better, and where it is not

### What the graph adds

A knowledge graph stores **facts**, not just text. A fact is `(subject, predicate, object)`, for example
`(RoBERTa, reimplements, BERT)`, plus its **provenance**: the exact quote, the chunk, the section and the paper.

1. **Connection across documents.** An entity that appears in two papers is one node. A question about it reaches both
   papers even if the question's wording matches only one. The `RoBERTa → BERT` link above is a real edge in the
   library: it is how a question about "the model RoBERTa is based on" finds BERT's paper.
2. **Multi-hop reasoning.** "What dataset was used to pre-train the model that RoBERTa improves on?" needs
   RoBERTa → BERT → BooksCorpus. Vector search retrieves one hop at best. The graph walk follows the chain.
3. **Aggregation.** "Which of my papers mention BERT?" is a graph question (one entity, many papers), not a similarity
   question.
4. **Explanations.** Because each fact is a typed link with a quote, the system can show the *chain of reasoning*, not
   just a list of similar paragraphs.
5. **A map of the library.** Communities (clusters of related entities) give an overview of what themes your documents
   contain, which supports "what is this collection about?" questions and a navigable visualization.
6. **Provenance for trust.** Every fact points back to a quote. Nothing in the graph is "just the model's opinion".

### The honest part: where it does not help, or hurts

- **Single-passage lookups gain nothing.** If the answer sits in one paragraph, plain hybrid retrieval already finds it.
  Most questions in the evaluation are like this, which is why the measured gain is modest.
- **The graph is only as good as extraction.** A small 3B local model reads each chunk in isolation. It produces junk
  entities, merges the wrong things, and drops facts. Ariadne spends most of its engineering on filtering this (section
  7) rather than pretending it is not a problem.
- **Linked is not relevant.** Being connected to the same entity is a lead, not evidence. Unfiltered graph chunks made
  answers wander off-topic, so every graph-found chunk must also pass the reranker.
- **Cost.** Building the graph needs one LLM call per small batch of chunks. Offline on a laptop this takes minutes to
  hours per paper; online it is limited by free-tier rate limits.
- **Measured result:** on 49 hand-written questions, plain retrieval answered **34 fully correctly (69%)** and graph
  mode **38 (78%)**. That is a consistent direction, but **within the noise** of a 49-question set. See section 16.

The claim Ariadne makes is therefore narrow and defensible: the graph adds capabilities plain RAG lacks (cross-document
links, aggregation, path explanations, a visual map, an evidence trail), and does not make ordinary lookups worse.

---

## 4. Architecture at a glance

```
                          UPLOAD (.pdf  .md  .txt)
                                   │
                    ┌──────────────▼───────────────┐
                    │ parse → sentence-aware chunks │  <400 tokens, contextual headers
                    │ → bge-small embeddings        │
                    └──────────────┬───────────────┘
                                   ▼
        ┌─────────────────── PostgreSQL + pgvector ───────────────────┐
        │ chunks (vector + tsvector)   nodes   edges(+quote, chunk)   │
        │ documents   communities   merges log   calibration          │
        └──────────────▲───────────────────────────────┬──────────────┘
                       │                               │ background, after chunks are saved
                       │                    ┌──────────▼──────────────┐
                       │                    │ LLM triple extraction   │ Ollama OR Groq/Gemini
                       │                    │ 4-stage filter          │
                       │                    │ + name-quality filter   │
                       │                    │ entity merging          │
                       │                    │ Louvain communities     │
                       │                    └─────────────────────────┘
 QUESTION                                                
    │                                                    
    ▼                                                    
 split into parts ─► per part: dense + BM25 ─► RRF ─► document-aware ─► cross-encoder rerank
    │                                                                         │
    │                                                                  relevance gate
    │                                                                         │
    ├─ graph mode: link question → entities → walk ≤2 hops → extra chunks (reranker-filtered)
    ├─ structured intent (profile / path / aggregate): graph decides the evidence
    ▼
 grounded LLM answer with [n] citations ─► citation validation ─► NLI claim check (worker process)
    │
    ▼
 answer + sources + graph facts + evidence trail + verdicts ─► React UI (answer + 3D graph)
```

---

## 5. Ingestion: from file to searchable chunks

**Supported types today:** `.pdf`, `.md`, `.txt` (upload limit 100 MB, duplicates detected by SHA-256 content hash, so
re-uploading the same file returns the existing document instead of re-processing it). `.docx`, `.pptx`, `.html` are
**planned** (section 19).

### Stages (shown live in the UI through a server-sent-events status stream)

1. **Parsing.** PDFs go through a layout-aware parser that keeps page numbers and heading structure.
2. **Chunking.** Sentence-aware: chunks target about 300 tokens and never exceed 400, and never cut a sentence in half.
   Each chunk keeps its section path (for example `3 Experimental Setup > 3.1 Implementation`) and its page range.
3. **Contextual headers.** The paper title and section path are included with the chunk when it is embedded, so a
   chunk that says "the model" is embedded together with *which* paper and section it belongs to.
4. **Embedding.** `bge-small-en-v1.5`, asymmetric: passages and queries are encoded differently, as the model expects.
   Stored in pgvector with an HNSW index (cosine distance). A Postgres full-text (`tsvector`) column is stored
   alongside for the keyword side.
5. **Calibration** (`calibrate_on_ingest`). Generates sample questions from the new document and measures the reranker
   score distribution to set that document's relevance threshold (section 6.5).
6. **Graph build** (background). Triple extraction, filtering, entity resolution, community rebuild (section 7). The
   document is already searchable while this runs; the job status reports `extracting`, `summarizing`, then `done`.
7. **Recovery.** If the server restarts mid-ingestion, `recover_interrupted()` closes the stuck jobs at startup so the
   UI never shows an eternal spinner.

**Verified:** uploading a new document end to end (chunk, embed, calibrate, extract facts through Groq, summarize,
done), then asking a question about it and receiving a correct, cited answer in about 4 seconds.

---

## 6. Retrieval: finding the right passages

### 6.1 Dense retrieval
Question embedded with the query prompt, nearest chunks by cosine distance through the HNSW index.

### 6.2 True BM25 (not just Postgres full-text rank)
Postgres `ts_rank` is not BM25: it has no inverse document frequency and no length normalization. Ariadne uses
Postgres full-text search only to produce **candidates**, then scores them with a real BM25 implementation in Python:
IDF, term-frequency saturation (`k1 = 1.2`), and document length normalization (`b = 0.75`). This matters for rare,
exact terms (model names, dataset names, numbers) that embeddings blur.

### 6.3 Reciprocal rank fusion
Dense and BM25 rankings are merged with RRF (`k = 60`). A chunk ranked well by both rises to the top; a chunk found by
only one is still kept. Every result remembers which retrievers found it and at what rank.

### 6.4 Cross-encoder reranking
A cross-encoder reads the question and each candidate **together** and scores relevance far more accurately than
comparing two separately computed vectors. It rescored the merged candidates and produces the final order.

### 6.5 The relevance gate (refusing honestly)
If even the best chunk scores below a threshold, the system treats the question as not covered by the documents.
Thresholds are **per document**, calibrated at ingestion from that document's own sample questions, clamped to
`[-9.0, -7.0]`.

*Measured finding:* on the current library every calibrated threshold collapsed to the ceiling (`-7.0`). The per-document
idea is implemented and sound, but with this data it gave no benefit over a single global threshold. That is reported
here rather than hidden.

Out-of-corpus questions then either return the refusal sentence, or, if the user ticks **"Allow General Knowledge
Fallback if out-of-corpus"**, an answer from the model's own knowledge that is **always prefixed "Not from your
documents:" and never carries citations**.

### 6.6 Document-aware retrieval
If the question names a paper ("in the RoBERTa paper…"), chunks from that paper get a bonus (`doc_prefer_bonus = 2.0`)
and the answer prompt is told the question's FOCUS, with a warning not to attribute a follow-up paper's modified setup
to the original. This was added after the system confused BERT's and RoBERTa's training details.

### 6.7 Question decomposition
Multi-part questions ("What is X, and how does it differ from Y?") are split by rules into parts. Each part is
retrieved separately and the results are merged, so every part gets evidence. The prompt then tells the model to answer
each part. *An isolated A/B of splitting on versus off has not been run;* the last full evaluation had it active.

### 6.8 Community overview
For broad questions ("what is this collection about?") the retriever can attach machine-written community summaries as
background (never cited, labelled OVERVIEW in the prompt).

### 6.9 Retrieval-only endpoint
`POST /api/search` returns ranked chunks with their scores and `found_by` rank per retriever, no LLM involved, which
makes retrieval quality inspectable on its own.

---

## 7. The knowledge graph

### 7.1 Data model

| Table | What it holds |
|---|---|
| `nodes` | Entities: canonical name, type (`concept`, `person`, `method`, `dataset`, `model`, `paper`, `other`), aliases, name embedding, mention count |
| `edges` | Facts: source node, target node, predicate, **evidence quote**, chunk id, document id |
| `communities` | Louvain clusters with an LLM-written title and summary |
| merge log | Every entity merge, so merges are inspectable |

An edge always carries `evidence_text` (an exact quote) and `chunk_id`. That provenance is what powers citations, the
evidence trail, path explanations, and the entity profile.

### 7.2 Triple extraction
Chunks are sent to the LLM in small batches (4 per call). The prompt asks for strict JSON triples, requires entities to
be short noun phrases of at most 5 words, predicates of at most 3 words, correct direction (active voice), an exact
quote as evidence, and gives examples including a "bad" example. The user message names the **paper**, because in a
research paper "we", "our model" and "the proposed approach" all mean the paper's own method.

### 7.3 The filter stack (why the graph is not full of garbage)
LLM output is untrusted. Every triple passes:

1. **Parse and validate.** Malformed JSON, missing fields, unknown types, over-long fields are dropped.
2. **Self-reference resolution.** "we", "our model", "this paper", "the proposed method", "it" become the paper's name.
   Triples that end up linking an entity to itself are dropped. *Measured:* self-reference nodes went from **3 to 0**.
3. **Attachment by quote.** Each triple is re-attached to the chunk that **actually contains** its evidence quote
   (the model's own chunk number is not trusted). No matching chunk, no triple.
4. **Lexical entity verification.** At least one content word of each entity must appear in the triple's own quote.
   This catches a fact attached to the wrong entity.
5. **Entity name quality** (`entity_filter.py`, **new**). A rule-based gate on the *name alone*:
   - rejects sentence fragments ("to prevent positions from attending to subsequent positions"),
   - rejects numbers and quantities ("1024", "0.38M parameters and 4.48 GFLOPs", "inference speed by 19.7%"),
   - rejects filler nouns ("size", "formula", "recent work", "three ingredients"),
   - rejects document structure ("Table 2", "Related Work"), citations ("Zhu et al., 2015"), equations,
   - rejects names starting with determiners or prepositions, and names with more than five words,
   - keeps real names and technical terms: `BERT`, `RESIDE-6K`, `ResNet-50`, `self-attention`,
     `next sentence prediction`, `Attention Is All You Need` (a paper title may contain glue words).

   It is covered by 34 unit tests. There is also a stricter stored-graph cleanup (`prune_graph.py`, dry run by default,
   `--apply` to delete) that can additionally drop one-off lowercase phrases. On the current graph the standard mode
   would remove 241 of 561 nodes (and the 231 of 408 facts that touch them); strict mode would leave only 120 nodes
   and is **not recommended**, since it removes about 89% of the facts. **Written, not applied:** run it yourself, and
   take a backup first.

### 7.4 Entity resolution (merging duplicates)
Different spellings of the same thing are merged into one node:

- **Same-name tier:** names that are near-identical (case, punctuation, hyphenation) and whose **contexts** agree
  (embedding similarity of the surrounding facts, threshold 0.75).
- **Contained-name tier:** a name contained in another ("RepVGG block" in "RepVGG") merges only for a generic suffix or
  the paper's own name (subset threshold 0.70), so "BERT" is not merged into "BERT-base fine-tuned on SQuAD".
- Every merge is logged.

*Measured:* nodes spanning more than one paper went from **22 to 5** after the rebuild (most of the 22 were wrong merges,
for example generic words like "model" shared across papers).

### 7.5 Communities
Louvain community detection groups related entities. An LLM writes a title and summary per community. These feed the
Communities view and the OVERVIEW context. After the last rebuild: **19 communities**. *Known weakness:* summaries were
originally built when the library held only three documents.

### 7.6 Rebuilding safely
`reextract_graph.py` rebuilds one document or all of them. It is **crash-safe by design**, because an early version
wiped the graph when extraction failed halfway:

- extraction runs **first**; the old graph is replaced in **one transaction**, only if new triples were produced,
- a pre-flight check confirms the embedder and the LLM both work before touching anything,
- a failed document keeps its old graph,
- entity merging and community rebuild run only if at least one document rebuilt.

### 7.7 Measured effect of the filter stack (rebuild of 6 papers)

| | Before | After |
|---|---|---|
| Edges (facts) | 629 | 408 |
| Nodes | 894 | 561 |
| Nodes spanning 2+ papers | 22 | 5 |
| Self-reference nodes | 3 | 0 |
| Communities | 16 | 19 |

The graph is smaller and cleaner. Whether answers improved after the rebuild has **not** been measured yet (the full
evaluation on the rebuilt graph did not complete because of memory crashes; section 18).

---

## 8. Graph-augmented answering

When `use_graph` is on:

1. **Question → entities.** Candidate phrases from the question are matched against node names and aliases (fuzzy
   matching), against name embeddings, and against the entities of the already-retrieved chunks. Only *specific* names
   are used as starting points: a multi-word name, or a single word with a capital or digit (`BERT`, `SQuAD2`). Plain
   lowercase words like "size" would match half the library and only add noise.
2. **Walk.** Breadth-first over edges in either direction, up to 2 hops, at most 30 edges per hop. **Hub nodes** (more
   than 25 connections, such as a paper's own model name) are expanded only at the first hop so one hub cannot flood the
   context.
3. **Multi-hop seeds.** The answer's entities are often inside the best chunk rather than in the question ("what are the
   three families?" → the introduction chunk names them), so the walk also starts from the entities of the top
   retrieved chunks.
4. **Rank the chunks the walk found.** Default: count of facts pointing at each chunk, weighted `1/hop`. Optional
   (`graph_scoring = "ppr"`): **personalized PageRank**, a random walk that restarts at the question's entities. It
   handles several routes to one chunk and damps hubs by itself. *Written, not yet A/B tested.*
5. **Filter by relevance.** Over-fetch, then keep only chunks the cross-encoder scores above `-8.0`, at most 4. This is
   what stopped the graph from adding distracting passages.
6. **Append, never replace.** Graph chunks go **after** the normal top-k.
7. **GRAPH FACTS** (with their source numbers) are given to the model as pointers, with the instruction that a fact is a
   lead, not evidence: only state what the source text supports, and cite sources, never facts.
8. **Evidence trail.** The response includes the chain of facts actually behind the answer, nearest hop first, each with
   its quote, paper, section and the source number it appears as, so every step can be checked.

---

## 9. Structured questions: profile, path, aggregate

Some questions are not "find similar text". Ariadne detects three shapes and lets the graph choose the evidence, then
the LLM still writes the answer with citations.

| Intent | Example | What happens |
|---|---|---|
| **Profile** | "Tell me everything about RoBERTa" | Gathers the entity's facts from **every** paper, up to 3 per paper, round-robin so all papers are represented, with their chunks |
| **Path** | "How is BERT connected to RoBERTa?" | Finds the shortest chains of facts between the two entities (up to 3 hops) and explains them step by step |
| **Aggregate** | "Which papers mention BERT?" | Lists the papers that contain the entity, including papers whose text names it but where no fact was extracted |

If the graph cannot support the intent (entity not found, or no chain connects them) it falls back to ordinary
retrieval, so an unusual question never gets a worse answer than before.

**Verified live (REST):** entity search, entity profile (BERT: 59 facts across 2 papers, with quotes), "which papers
mention BERT", and a path query (`BERT → RoBERTa` returns the 1-hop `RoBERTa reimplements BERT` with its quote, and
longer chains). `BERT → Transformer` correctly reported *not connected* rather than inventing a link.

**Written, not verified end to end:** intent detection and the structured branch inside `/api/query` (the three example
questions above). The first live attempt exposed a crash (a `max()` over a single value) which was fixed; the re-run
was interrupted. These need a clean test run and unit tests before being called working.

---

## 10. Ask with AI: grounded generation and citations

### 10.1 The prompt contract
The model is told to answer **only** from numbered sources, cite after every factual sentence as `[1]` or `[2][3]`,
reply with an exact refusal sentence if the sources lack the answer, and say so when sources disagree. Addenda are added
only when relevant: graph facts, overview, focus, and one for each structured intent.

### 10.2 Mechanical citation validation
After generation:

- every `[n]` must point at a real source; invalid ones are **stripped, never shown as if valid** and reported,
- sentences with no citation are listed as `uncited_sentences`, so the UI can flag unsupported claims,
- a refusal is detected and reported (`refused`).

### 10.3 What comes back with every answer
The answer text; which provider and model wrote it; whether it came from `documents` or `general_knowledge`; the
numbered sources with document title, section, pages and text; cited and invalid citation numbers; uncited sentences;
NLI verdicts; the graph facts and seeds used; the evidence trail; the query's parts; which retrievers found what; timing
(retrieval ms, total ms, candidates, reranked or not).

### 10.4 Request options
`use_graph`, `verify_claims`, `general_knowledge`, `rerank`, `top_k`, `document_ids` (restrict to chosen documents).

---

## 11. NLI claim verification

### 11.1 What it is
Citation validation only proves that `[3]` exists. It does not prove that source 3 *supports the sentence*. NLI (natural
language inference) asks a trained model: given this premise (the cited text), is this claim **entailed**, **neutral**,
or **contradicted**? Ariadne runs `nli-deberta-v3-base` on every cited sentence against the text it cites.

### 11.2 Design (driven by measurement, not guesswork)
The first version looked fine on two examples. Measured on 38 claims it confirmed only **37%** of true claims and
wrongly called **39%** of them "contradicted". It was redesigned after comparing several premise strategies and two
models, including a **control** that cites a chunk from a *different* paper.

The shipped version:

- **Premise = the three most relevant sentences** of the cited chunk (not the whole chunk, which dilutes the signal).
- **Entailment cutoff 0.5** to call a claim *supported*.
- **Contradiction cutoff 0.9**: contradiction is only called when the model is very sure, because it misfires on true
  claims.
- Verdicts: `supported`, `unverified`, `contradicted`, plus a `conflict` flag when sources disagree.

### 11.3 Measured behaviour

| Measurement (38 claims, with a wrong-paper control) | Result |
|---|---|
| True claims confirmed | **42%** |
| Wrong-paper citations accepted | **0 of 38** |
| True claims wrongly called contradicted | **8%** |
| Cited sentences confirmed across the question sets | about a quarter |
| Cited sentences returned "unverified" | about three quarters |

### 11.4 How to read it (important)
**It is a precise confirmer, not a fault-finder.** A green "supported" is strong evidence: the control never produced a
false confirmation. "Unverified" means *not confirmed*, **never** "wrong": a correct sentence that blends two facts, or
paraphrases heavily, often cannot be confirmed from three sentences. **Do not treat it as a hallucination detector.**

### 11.5 Isolation
It runs in its **own worker process**. Running it inside the API crashed the server (native crash, access violation) in
three earlier evaluation runs. If the worker dies or cannot load, the answer is still returned with `verification.available
= false` instead of failing the request. It adds roughly 3 GB of memory; `ARIADNE_NLI_WARMUP=false` skips starting it.

---

## 12. Offline and online modes

Ariadne decides per request which language model writes the answer and builds the graph. This is a first-class feature,
shown in the UI, not a hidden setting.

### 12.1 Offline mode (Ollama)
- **Fully local.** No text leaves the machine. Works without internet.
- Default model `granite4:micro-h`; any installed Ollama model can be selected in the UI.
- Built for a laptop GPU (tested on an RTX 5050 8 GB). If the GPU run fails, it retries on CPU with a smaller context.

### 12.2 Online mode (cloud)
- Ordered provider list, default **Groq then Gemini**, editable.
- **Groq:** `openai/gpt-oss-20b`, with fallbacks to other Groq models; low reasoning effort to save tokens; if Groq's strict
  JSON mode rejects a response it retries without it; per-model timeouts move on to the next model.
- **Gemini:** `gemini-3.1-flash-lite-preview`, then `gemini-flash-lite-latest`, then `gemini-3-flash-preview`;
  per-model timeouts move on to the next model. API keys go in a header, never in a URL, so they cannot leak into logs.
- **`allow_local_fallback`:** if every cloud provider fails, fall back to local Ollama.
- The status endpoint reports `sends_text_off_machine: true/false`, and the UI shows which provider and model actually
  wrote each answer.

### 12.3 What the router does
`GET /api/llm` reports the mode, the order tried, installed local models and provider configuration. `PUT /api/llm`
changes mode, local model, or cloud order at runtime and persists it. `POST /api/llm/test` sends a test message through
the current chain and reports which provider answered and how long it took. The settings modal in the UI wraps these.

**Verified (this build):** cloud answers through Groq (about 4 to 5 s); answers forced through Gemini alone (about 5 s);
triple extraction through both providers (2 triples from a test sentence each, in 1.8 s and 3.7 s); a 12-request
local-Ollama stress run in which requests 6 to 11 completed on the GPU at about 3.3 s each.

### 12.4 Why Ollama sometimes fails (diagnosed)
It is **not the GPU**. The model loads and runs on the GPU (`100% GPU`) when memory is available. The failures seen were:

- `failed to allocate buffer` / `std::bad_alloc` on the **host side**: Windows ran out of virtual memory (commit limit),
  because the system drive was almost full and the secondary pagefile was not active,
- an Ollama prompt-cache save failing with `bad_alloc` for the same reason.

When memory was free, the same requests succeeded. After a failure Ollama retries on CPU, which is very slow (one paper
took hours). **Cloud mode avoids all of this**, which is why it is the practical default on this machine.

---

## 13. The frontend: a living knowledge graph

> **Status: written, not verified by me.** I can read and type-check the frontend code but could not view it in a
> browser in this environment, so everything below is described from the code, not from screenshots. It also contains
> changes made outside this document's session.

The UI is a React + Vite + TypeScript single-page app. The backend is expected at `http://localhost:8100`
(override with `VITE_API_TARGET`).

### 13.1 The 3D graph (`Graph3D.tsx`, built on `react-force-graph-3d` and three.js)
A force-directed graph of your whole library, rendered in 3D, where nodes are entities and edges are facts.

- **Node types are colour-coded:** concept, model, method, dataset, paper, person, other.
- **Relationship categories** group predicates: *builds on*, *contradicts*, *supports*, *is example of*, *required by*,
  *tangentially related*, each with its own style, so a contradiction looks different from a supporting link.
- **Four spatial formations:** `neural` (organic network), `galaxy` (a 4-arm 3D logarithmic spiral with a hub core),
  `sphere`, and `pillars`. Switching reorganizes the same graph.
- **Living motion** (`livingMotion`): the graph breathes and drifts instead of sitting dead still.
- **Auto-orbit:** the camera slowly circles; any interaction stops it.
- **Label modes:** `smart` (labels for important nodes), `all`, `minimal`.
- **Controls:** force spacing (repulsion), maximum visible nodes (keeps large libraries responsive).
- **Filters:** by node type, by relationship category, by community cluster, by source document, and a text search that
  focuses matching nodes.
- **Selection and focus:** click a node to select, hover for details, right-click for the node's neighbourhood
  (**ego network**), focus the camera on any node.
- **Answer highlighting:** after you ask a question, the **entities and edges used in the answer are highlighted inside
  the graph** (`highlightedNodeNames`, `highlightedEdgePairs`), so you literally see the path the answer took through
  your documents.

### 13.2 The views
| View | What it is |
|---|---|
| **Ask Ariadne** (`QueryView`) | The question box and answer. Toggles for **GraphRAG Walk**, **Cross-Encoder Rerank**, **NLI Claim Check**, and **Allow General Knowledge Fallback if out-of-corpus**. Shows the LLM answer with citations, expandable sources, the graph facts, the evidence trail, the provider that answered, and per-sentence NLI verdicts. |
| **Vault** (`VaultView`) | Upload documents, watch ingestion progress live (chunking, embedding, extracting, summarizing), list documents, inspect chunks. |
| **Communities** (`CommunitiesView`) | The Louvain clusters with their written summaries, each with an **Ask Ariadne** entry point. Says plainly when none have been built yet. |
| **Synapse Path Finder & Paradigm Tensions** (`ContradictionsView`) | Pick two entities (with one-click hub-entity shortcuts) and trace the path between them; also shows explicit `CONTRADICTS` links and natural-language tensions or performance comparisons between papers. |

### 13.3 Supporting components
- **Concept drawer:** click a node to see its connected entities, its **source paper**, and a **Trace Path** action.
- **Floating controls:** filters (node types, relationship categories, clusters), force spacing, formations, max nodes.
- **Neural telemetry HUD:** a live overlay with graph and system status.
- **LLM settings modal:** switch **Offline (Ollama)** or **Cloud**, choose the local model, reorder cloud providers, and
  **test** the connection and see which provider responded and how fast.
- **Top navigation:** switches between the views.

### 13.4 Frontend items still to do
The API already returns everything needed to show the use-graph toggle result, the "not in your documents" state, the
general-knowledge marker, graph facts, the evidence trail, NLI verdicts and the entity profile; making sure each of
these is surfaced cleanly in the UI, plus upload types, delete, and an API-key field, is still **to do**.

---

## 14. HTTP API reference

Base URL `http://localhost:8100` (the dev proxy target). Interactive docs at `/docs` (FastAPI).

| Method and path | Purpose |
|---|---|
| `POST /api/documents/upload` | Upload a file; returns `document_id`, `job_id`, and whether it was a duplicate |
| `GET /api/jobs/{job_id}/status` | Server-sent events stream of ingestion progress |
| `GET /api/documents` | List documents |
| `GET /api/documents/{id}/chunks` | Inspect chunk quality |
| `DELETE /api/documents/{id}` | Delete a document *(graph cleanup after delete is planned)* |
| `POST /api/search` | Retrieval only, with scores and `found_by` |
| `POST /api/query` | Full question answering (see 10.3 and 10.4) |
| `POST /api/graph/link` | Which graph nodes a question links to |
| `POST /api/graph/communities/build` | Rebuild communities |
| `GET /api/graph/communities` | List communities |
| `GET /api/graph` | Whole graph for the visualization |
| `GET /api/graph/nodes/{node_id}` | One node with neighbours and source |
| `GET /api/graph/stats` | Node, edge and community counts |
| `GET /api/graph/search?name=` | Entities close to a name, with how many papers mention each |
| `GET /api/graph/entity?name=` | Entity profile: facts grouped by paper, each with quote |
| `GET /api/graph/path?a=&b=&max_hops=` | Shortest chains of facts between two entities |
| `GET /api/graph/aggregate?entity=&predicate=` | Which papers say something about an entity |
| `GET /api/llm` | LLM mode and provider status |
| `PUT /api/llm` | Change mode, local model, cloud order |
| `POST /api/llm/test` | Test the current LLM chain |
| `GET /api/health` | Health, including reranker status |

---

## 15. Configuration reference

Set in `backend/.env` (prefix `ARIADNE_`), defaults in `backend/app/config.py`.

| Setting | Default | Meaning |
|---|---|---|
| `ARIADNE_GROQ_API_KEY` / `ARIADNE_GEMINI_API_KEY` | empty | Cloud keys (never commit `.env`; the repo's `.gitignore` excludes it) |
| `ARIADNE_GROQ_MODEL` / `ARIADNE_GEMINI_MODEL` | `openai/gpt-oss-20b` / `gemini-3.1-flash-lite-preview` | Preferred cloud models |
| `ARIADNE_NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` (this machine points at a local copy of the `-base` model) | NLI model |
| `ARIADNE_NLI_WARMUP` | `true` | `false` = do not start the NLI worker (saves ~3 GB) |
| `llm_model` | `granite4:micro-h` | Default local Ollama model |
| `graph_max_hops` / `graph_hop_edge_cap` / `graph_hub_degree` | 2 / 30 / 25 | Graph walk limits |
| `graph_extra_chunks` / `graph_extra_min_rerank` | 4 / -8.0 | Graph chunks kept and their rerank floor |
| `graph_scoring` | `hops` | `hops` or `ppr` |
| `doc_prefer_bonus` | 2.0 | Boost for a named paper's chunks |
| `relevance_min_threshold` / `relevance_max_threshold` | -9.0 / -7.0 | Clamp for calibrated thresholds |
| `relevance_rerank_cutoff` | -8.0 | Global fallback threshold |
| `resolve_*` | context 0.75, subset 0.70 | Entity-merge similarity |
| `nli_entail_cutoff` / `nli_contradiction_cutoff` / `nli_premise_sentences` | 0.5 / 0.9 / 3 | NLI behaviour |
| `calibrate_on_ingest` / `calibration_questions` / `calibration_percentile` | on / 20 / 5.0 | Per-document threshold calibration |

Runtime LLM choices (mode, local model, cloud order) persist to `backend/data/llm_settings.json`.

---

## 16. Evaluation: what was measured

### 16.1 Setup
49 hand-written questions over a library of 6 research papers (BERT, RoBERTa, Attention Is All You Need, RepVGG, a
dehazing paper, and a survey), split into **dev (27)**, **heldout (10)** and **heldout2 (12)** so that tuning on one
split can be checked on others, plus a separate **graph (8)** set of "graph-shaped" questions kept as a future target
(multi-hop, aggregation, path). Each question is scored on keyword, evidence and citation checks and a combined
**"fully correct"** metric. Answers were generated by a 3.2B local model.

### 16.2 Latest full result (before the graph rebuild)

| Split | Plain RAG fully correct | Graph mode fully correct |
|---|---|---|
| dev | 21 of 27 | 23 of 27 |
| heldout | 6 of 10 | 7 of 10 |
| heldout2 | 7 of 12 | 8 of 12 |
| **Total** | **34 of 49 (69%)** | **38 of 49 (78%)** |

The direction is consistent across all three splits, but with 49 questions a 4-question difference is **within noise**.
It supports "graph mode does no harm and probably helps", not "graph mode is proven better".

### 16.3 Other measurements
- Graph rebuild effect on structure: section 7.7.
- NLI behaviour: section 11.3.
- Relevance calibration: collapsed to the ceiling for all six documents (section 6.5).
- Unit tests: 134 passing at the last full run, plus 34 new tests for the entity-name filter. **The whole suite has not been
  rerun since the latest changes.**

### 16.4 Not measured yet
- A/B of **question splitting** on versus off.
- A/B of **personalized PageRank** versus hop counting.
- Answer quality on the **rebuilt, filtered** graph (the evaluation run did not complete; see below).
- Answer quality on the 8 graph-shaped questions through the structured intents.
- Latency under a stable GPU (earlier latency numbers were taken while Ollama was falling back to CPU and are invalid).

---

## 17. Reliability and engineering decisions

| Problem encountered | Decision |
|---|---|
| Native crash (access violation) when the NLI model ran inside the API | NLI isolated in a separate worker process; failure degrades to "unavailable", not a failed request |
| First graph rebuild deleted edges *before* extraction, then extraction failed and the graph was empty | Extract first, swap in one transaction, pre-flight checks, never delete on failure |
| Interrupted ingestions left jobs "running" forever | Startup recovery closes them |
| Duplicate uploads re-processed | SHA-256 content hash, existing document returned |
| LLM output is untrusted | Parse, validate, attach by quote, lexically verify, name-filter before anything is stored |
| Cloud providers rate-limit and fail | Ordered fallback chain, per-model fallbacks, per-model timeouts, optional local fallback, always report which provider answered |
| Cloud API keys leaking | Keys sent in headers, not URLs; `.env` git-ignored |
| A stored model setting silently pointed the local fallback at a tiny 0.5B model, which returned junk | Setting corrected to `granite4:micro-h`; flagged as something to surface in the UI |
| Hard-to-trust numbers | Every metric reported with its sample size; invalid runs (CPU fallback, crashed API) discarded rather than averaged in |
| Memory pressure on a laptop (model loads failing with "paging file too small") | Cloud mode default; `ARIADNE_NLI_WARMUP`; documented diagnosis |

---

## 18. Known limitations and failures

Honest list, roughly by severity.

1. **Extraction quality is the main limit of graph RAG here.** A 3B local model produces fragments and wrong merges; the
   filters (section 7.3) fix much of it, and drop many true facts too. On a new, different-domain report the name filter
   and the quote check discarded most of an LLM's 27 proposed facts in a test batch; tuning is unfinished.
2. **Memory.** On the development laptop the Windows commit limit is exhausted often (low free space on the system drive,
   a configured secondary pagefile that Windows does not activate). This crashes Ollama and the API during model loads.
   It caused the unfinished post-rebuild evaluation. It is an environment problem, not a design problem.
3. **No isolated measurement yet for splitting or PageRank, and no post-rebuild answer-quality number.**
4. **Structured intents inside `/api/query` are unverified end to end** (the REST endpoints are verified).
5. **NLI confirms only about 42% of true claims.** It is useful as a precise confirmer, not as a complete verifier.
6. **Relevance calibration collapses to a constant** on the current library.
7. **Only `.pdf`, `.md`, `.txt` are supported.**
8. **Deleting a document does not yet clean up** orphan nodes, stale community summaries, or merge records.
9. **No authentication** on the API (it is built for local use): no API key, no rate limiting, permissive CORS.
10. **Abbreviations and aliases are not merged** ("BERT" and "Bidirectional Encoder Representations from Transformers"
    can be two nodes).
11. **Community summaries can be stale** after documents are added or removed unless rebuilt.
12. **The frontend has not been visually verified** by me (section 13).
13. **The graph for the most recent upload (`TDR_Team AUV MIT-B_RS2026.pdf`) has no facts yet:** it was extracted while the
    cloud providers were failing and the local fallback was a 0.5B model. A cloud rebuild was started; its result is not
    confirmed here.
14. **Cloud mode sends document text to third parties** (Groq, Gemini). Offline mode does not.

---

## 19. Future work

Ordered by value for the effort.

### 19.1 Quality of the graph
- **Alias and abbreviation merging.** Mine "Long Form (ABBR)" definitions from the text, add initials matching, and merge
  only when the surrounding context also agrees (so two unrelated "BERT"s are not merged). New tier in `resolve.py`.
- **Tune the name filter and quote check on more domains**, and measure precision and recall of kept facts against a
  hand-labelled set of about 30 facts per paper.
- **Better extraction model.** Use a stronger cloud model for the one-time extraction and keep the small one for
  answering; or fine-tune a small extractor.
- **Typed, normalized predicates** (a small controlled vocabulary) so "uses", "employs" and "utilizes" are one relation.
- **Relation confidence scores** and a "show low-confidence facts" toggle.

### 19.2 Retrieval and answering
- **Chunk-to-chunk similarity edges** (`chunk_links`): link each chunk to its top neighbours (cosine about 0.8 or more)
  and use them as extra, relevance-filtered candidates.
- **Personalized PageRank A/B** (code written) and make it the default if it wins.
- **Map-reduce global search** over community summaries for "what are the main themes across everything?".
- **Learned query routing** replacing the regex-based intent detection, and richer aggregation queries (counts, comparisons).
- **A/B for question splitting**, and a smarter splitter for nested questions.
- **Query rewriting** for follow-up questions in a conversation, and **conversation memory**.
- **Hand-labelled gold evaluation set** and a larger one (hundreds of questions) so differences stop being "within noise";
  add retrieval-only metrics (recall@k, MRR) next to answer metrics.

### 19.3 Verification
- **Claim splitting** before NLI (one claim per check).
- **A larger NLI model** once memory allows, and multi-sentence premises for synthesized claims.
- **Self-correcting answers:** regenerate or remove sentences that come back contradicted.
- **Highlight the supporting sentence** in the source for each supported claim in the UI.

### 19.4 Product and platform
- **More file types:** `.docx`, `.pptx`, `.html`, spreadsheets, scanned PDFs (OCR), web pages by URL, and an
  **incremental re-ingest** when a file changes.
- **Document deletion cleanup:** remove orphan nodes and merge records, mark communities stale, recount mentions.
- **Security:** API-key authentication (`ARIADNE_API_KEY`), rate limiting, upload hardening (type sniffing, size, malware
  scan hooks), configurable CORS, per-user libraries and permissions.
- **Multi-user and collections:** projects or workspaces, sharing, per-collection graphs.
- **Export:** answers with citations to Markdown/PDF, graph to GraphML or Neo4j.
- **Observability:** per-stage timings, a retrieval debugger UI (why this chunk ranked here), cost and token tracking for
  cloud mode.
- **Packaging:** one-command start (Docker Compose for API, DB and frontend), a production build, a model-download helper
  that fetches only the needed weights, CI running the unit tests.
- **Resilient local inference:** detect low memory before loading, queue background graph work so it never competes with
  an interactive question, and expose the "why did it fall back" reason in the UI.

### 19.5 Frontend
- Show every response field cleanly: graph toggle result, "not in your documents", general-knowledge badge, graph facts,
  evidence trail, NLI verdicts, entity profile panel, provider badge.
- Upload with type hints, delete with confirmation, API-key field.
- Click a citation to open the source passage with the supporting sentence highlighted; click an evidence-trail step to
  fly the 3D camera to that fact's edge.
- Time slider or per-document layers in the graph; saved views; screenshot and share.
- Accessibility pass (keyboard navigation for the graph, reduced-motion setting for "living motion").
- Mobile layout.

---

## 20. Running it

```powershell
# 1. database (Postgres + pgvector on host port 5433)
docker compose up -d db

# 2. backend
cd backend
pip install -r requirements.txt
copy .env.example .env          # then add GROQ / GEMINI keys if you want cloud mode
python -m uvicorn app.main:app --port 8100

# 3. frontend
cd ..\frontend
npm install
npm run dev                     # proxies /api to http://localhost:8100

# tests (no services needed)
cd ..\backend
python -m pytest tests -q
```

**Optional and maintenance commands (backend folder)**

| Command | Purpose |
|---|---|
| `python reextract_graph.py [file.pdf]` | Rebuild the graph for all documents, or one |
| `python prune_graph.py` / `--apply` / `--strict` | Preview or apply removal of filler nodes (take a backup first) |
| `python eval/run_eval.py http://localhost:8100 [splits]` | Run the plain-vs-graph evaluation (`EVAL_VERIFY=0` skips NLI) |
| `python calibrate_relevance.py` | Recompute relevance thresholds |
| `python resolve_entities.py` | Re-run entity merging |
| `python switch_llm.py` | Switch the LLM mode from the command line |

**Backing up the graph before destructive operations**
```
docker exec ariadne-db pg_dump -U ariadne -t nodes -t edges -t communities --data-only --inserts ariadne > graph-backup.sql
```

**Local models (offline mode):** install [Ollama](https://ollama.com), then `ollama pull granite4:micro-h`. If memory is
tight, keep the NLI worker off (`ARIADNE_NLI_WARMUP=false`) and use cloud mode.

---

## 21. Repository layout

```
Ariadne/
├─ README.md                      this file
├─ docker-compose.yml             Postgres + pgvector
├─ scripts/                       helper scripts (e.g. Windows pagefile move)
├─ test_docs/                     sample papers used for evaluation
├─ backend/
│  ├─ app/
│  │  ├─ main.py                  FastAPI app, startup (recover jobs, warm models)
│  │  ├─ config.py                all settings
│  │  ├─ generation.py            grounded prompt, citation validation, general fallback
│  │  ├─ nli.py, nli_worker.py    claim verification in its own process
│  │  ├─ llm/                     router, Ollama / Groq / Gemini providers
│  │  ├─ ingestion/               pdf_parser, chunker, pipeline, extraction,
│  │  │                           entity_filter, resolve
│  │  ├─ retrieval/               hybrid, bm25, fusion, rerank, relevance, calibrate,
│  │  │                           doc_aware, decompose, graph, graph_context,
│  │  │                           communities, intent, structured, entity_views
│  │  └─ routers/                 documents, query, graph, graph_views, llm
│  ├─ db/                         schema and migrations
│  ├─ eval/                       questions.json, run_eval.py, results/
│  ├─ tests/                      unit tests (plus integration/e2e)
│  ├─ reextract_graph.py, prune_graph.py, calibrate_relevance.py, resolve_entities.py
│  └─ requirements.txt
└─ frontend/
   └─ src/
      ├─ components/              Graph3D, ConceptDrawer, FloatingControls,
      │                           NeuralTelemetryHUD, LLMConfigModal, TopNav
      ├─ views/                   QueryView, VaultView, CommunitiesView, ContradictionsView
      ├─ api/                     typed API client
      └─ types/                   shared TypeScript types
```

---

## 22. Glossary

| Term | Plain meaning |
|---|---|
| **RAG** | Retrieval-augmented generation: look up relevant text first, then have a language model answer using it |
| **GraphRAG** | RAG that also uses a knowledge graph built from the documents |
| **Chunk** | A small piece of a document (here under 400 tokens) that is searched and cited |
| **Embedding / dense vector** | Numbers that represent a text's meaning, so similar meanings are close together |
| **pgvector / HNSW** | A Postgres extension and index for fast nearest-neighbour search over vectors |
| **BM25** | A classic keyword scoring formula that rewards rare words and discounts long documents |
| **RRF** | Reciprocal rank fusion: merging several ranked lists by rank, not score |
| **Cross-encoder / reranker** | A model that reads the question and a passage together and scores relevance accurately |
| **Relevance gate** | The rule that refuses to answer when no passage is relevant enough |
| **Triple** | A fact `(subject, predicate, object)` |
| **Node / edge** | An entity / a fact connecting two entities |
| **Provenance** | The exact quote and chunk a fact came from |
| **Entity resolution** | Merging different spellings of the same thing into one node |
| **Community (Louvain)** | A cluster of tightly connected entities found automatically |
| **Hop** | One step along a fact from one entity to the next |
| **Hub** | An entity with very many connections, expanded with care |
| **Personalized PageRank** | A random-walk score that stays near the question's entities |
| **NLI** | Natural language inference: does text A entail, contradict, or say nothing about claim B |
| **Entailment / contradiction / neutral** | The three NLI outcomes (here mapped to supported / contradicted / unverified) |
| **Evidence trail** | The ordered facts and quotes behind an answer |
| **Ego network** | A node and everything directly connected to it |
| **Ollama** | A program that runs language models locally |
| **Commit limit** | The Windows cap on total memory promises (RAM plus pagefile); exceeding it makes allocations fail |
| **Fully correct** | The evaluation metric: right keywords, evidence found, and citations valid |

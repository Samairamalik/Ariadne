from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARIADNE_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://ariadne:ariadne@localhost:5433/ariadne"
    upload_dir: str = "data/uploads"

    # Demo data: when true, GET /api/graph seeds a hand-written sample graph (NOT extracted from your documents)
    # into an empty graph. Off by default: real graph extraction now exists, and fabricated edges with
    # "evidence" text would contradict Ariadne's core promise that every edge quotes your own sources.
    demo_graph: bool = False

    # Embedding (bge-small-en-v1.5: 384-d, hard 512-token input limit)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    embed_max_tokens: int = 512
    # CPU by default: both models are tiny, and CUDA on this machine has hard-crashed the whole process during model
    # load (a native crash no try/except can catch). Opt in with ARIADNE_EMBED_DEVICE=cuda / ARIADNE_RERANK_DEVICE=cuda.
    embed_device: str = "cpu"  # cpu | cuda | auto
    rerank_device: str = "cpu"

    # Chunking. Max leaves headroom under the 512 embedder limit for the section-path header.
    chunk_target_tokens: int = 300
    chunk_max_tokens: int = 400
    chunk_min_tokens: int = 100
    chunk_overlap_sentences: int = 1

    # LLM. Two modes:
    #   offline: Ollama only, nothing leaves the machine. Pick the model at runtime (PUT /api/llm).
    #   cloud:   try providers in `cloud_order` (groq -> gemini); optionally fall back to local Ollama last.
    # Cloud mode SENDS DOCUMENT TEXT to those services. Keys come from env/.env (ARIADNE_GROQ_API_KEY, ...),
    # are never returned by the API, and never written to disk by the app.
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "granite4:micro-h"  # default Ollama model; installed models are listed by GET /api/llm
    llm_mode: str = "offline"  # offline | cloud
    cloud_order: str = "groq,gemini"
    cloud_allow_local_fallback: bool = True
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"  # model names change; override via env if rejected
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite-preview"
    llm_settings_path: str = "data/llm_settings.json"  # persisted mode/model/order (no keys)

    # Question -> graph node linking (starting values; tune against real questions)
    link_fuzzy_cutoff: int = 90       # rapidfuzz ratio, 0-100
    link_embed_cutoff: float = 0.75   # cosine between question vector and node-name vector
    link_embed_k: int = 10
    link_max_nodes: int = 8

    # Document-aware retrieval: score bonus (cross-encoder logits) for chunks from documents the question names
    doc_prefer_bonus: float = 2.0   # 4.0 pushed a needed second paper out of context (x-transformer regressed)
    doc_prefer_extra_candidates: int = 12  # best chunks fetched from the named documents and added to the pool

    # Graph-augmented retrieval (step 9)
    graph_max_hops: int = 2
    graph_hop_edge_cap: int = 30     # edges read per hop
    graph_hub_degree: int = 25       # nodes with more edges than this are not expanded past hop 1
    graph_extra_chunks: int = 4      # chunks added beyond the normal top-k
    graph_scoring: str = "hops"      # how graph-found chunks are ordered: "hops" (count facts by hop) or "ppr" (personalized PageRank)
    graph_extra_min_rerank: float = -8.0  # same cutoff as the relevance gate: a graph-found chunk must at least be 'about' the question

    # Community summaries (step 7)
    community_min_size: int = 4
    community_top_k: int = 3
    community_cutoff: float = 0.45   # starting value; not calibrated
    community_rebuild_on_ingest: bool = True  # summaries describe the whole graph, so any new document makes them stale

    # Semantic citation check with an NLI model (see app/nli.py). The hub id downloads ~4.5 GB of variants; this machine
    # keeps just the weights it needs on D: and points here with ARIADNE_NLI_MODEL in .env.
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_warmup: bool = False               # False = do not start the NLI worker at startup (saves ~3 GB; claims then stay unverified)
    nli_entail_cutoff: float = 0.5         # entailment probability needed to call a claim supported
    nli_contradiction_cutoff: float = 0.9  # contradiction is called only when very sure (it misfires on true claims)
    nli_premise_sentences: int = 3         # the most claim-relevant sentences of a cited chunk form the premise

    # Per-document relevance calibration (see app/retrieval/calibrate.py)
    calibrate_on_ingest: bool = True
    calibration_questions: int = 20      # synthetic questions generated per document
    calibration_percentile: float = 5.0  # threshold = this percentile of the answerable-question scores
    # The calibrated threshold is clamped to a band measured on this corpus:
    #   floor   -9.0: just above the best score ever seen for an UNRELATED question (-9.98);
    #   ceiling -7.0: below the hardest COVERED question seen (-6.0, a two-topic comparison) with 1 point of margin.
    # Calibrating on single-passage questions gave a raw 5th percentile of +0.06..+4.0 for five of six documents (the
    # questions are easy), so without this ceiling the gate would refuse legitimate hard questions.
    relevance_min_threshold: float = -9.0
    relevance_max_threshold: float = -7.0

    # Relevance gate: below this the question is treated as not covered by the documents (global fallback)
    # Cross-encoder logit of the best chunk. Measured: unrelated <= -9.98, covered >= -6.0 (comparison questions
    # score low because one chunk rarely covers both topics). -8 sits mid-gap; re-check on new corpora.
    relevance_rerank_cutoff: float = -8.0
    relevance_cosine_cutoff: float = 0.6   # used only when the reranker is unavailable

    # Entity merging (same normalized name, different node rows).
    resolve_on_ingest: bool = True
    resolve_context_cutoff: float = 0.75  # measured: the one generic pair ("the model") scored 0.63, true duplicates >= 0.80
    resolve_subset_cutoff: float = 0.70   # "contained" variants: the word rule does the policing; context only checks same domain

    # Retrieval
    retrieve_vector_k: int = 30
    retrieve_lexical_k: int = 30
    rrf_k: int = 60
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 25
    rerank_top_k: int = 6


settings = Settings()

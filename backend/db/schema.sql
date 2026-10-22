-- Ariadne schema. Derived from ariadne-design.md §2 with the deviations noted inline (marked DEVIATION).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── DOCUMENTS ───────────────────────────────────────────────────────
CREATE TABLE documents (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename     TEXT NOT NULL,
    -- DEVIATION: content hash makes re-upload of identical bytes idempotent (design left this undefined)
    content_hash TEXT NOT NULL UNIQUE,
    title        TEXT,
    author       TEXT,
    doc_type     TEXT CHECK (doc_type IN ('pdf', 'markdown', 'text')),
    page_count   INT,
    token_count  INT,
    ingested_at  TIMESTAMPTZ DEFAULT now(),
    status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending','chunking','embedding','extracting','resolving','done','failed')),
    error        TEXT,
    metadata     JSONB DEFAULT '{}',
    relevance_threshold DOUBLE PRECISION,  -- calibrated from synthetic questions (app/retrieval/calibrate.py)
    calibrated_at TIMESTAMPTZ
);

-- ── CHUNKS ──────────────────────────────────────────────────────────
CREATE TABLE chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'prose' CHECK (kind IN ('prose','table','code','footnote','caption')),
    section_path  TEXT,                 -- "3 Model Architecture > 3.2 Attention"
    page_start    INT,
    page_end      INT,
    text          TEXT NOT NULL,        -- clean text shown to the user / cited
    token_count   INT NOT NULL,
    doc_len       INT,                  -- length in lexemes (repeats counted): BM25 length normalisation
    embedding     vector(384),          -- embedded as "section_path\n\ntext" (contextual header)
    -- DEVIATION: stored generated tsvector instead of an expression index; includes section path so
    -- heading words are searchable. NOTE: this is Postgres FTS (ts_rank_cd), not true BM25.
    tsv           tsvector GENERATED ALWAYS AS (
                      to_tsvector('english', coalesce(section_path, '') || ' ' || text)
                  ) STORED,
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_doc_idx ON chunks (document_id, chunk_index);

-- ── GRAPH NODES ─────────────────────────────────────────────────────
CREATE TABLE nodes (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_name TEXT NOT NULL,
    aliases        TEXT[] DEFAULT '{}',
    node_type      TEXT NOT NULL DEFAULT 'concept'
                   CHECK (node_type IN ('concept','person','method','dataset','model','paper','other')),
    description    TEXT,
    embedding      vector(384),
    first_seen_doc UUID REFERENCES documents(id) ON DELETE SET NULL,
    created_at     TIMESTAMPTZ DEFAULT now(),
    mention_count  INT DEFAULT 1,
    -- DEVIATION: uniqueness is (name, type), not name alone. The design's UNIQUE(canonical_name) made its own
    -- "type-bucketing fixes polysemy" mitigation impossible (two "attention" nodes could never coexist).
    UNIQUE (canonical_name, node_type)
);
CREATE INDEX nodes_embedding_idx ON nodes USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- ── GRAPH EDGES ─────────────────────────────────────────────────────
CREATE TABLE edges (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id     UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id     UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    predicate     TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    chunk_id      UUID REFERENCES chunks(id) ON DELETE CASCADE,
    document_id   UUID REFERENCES documents(id) ON DELETE CASCADE,
    confidence    TEXT DEFAULT 'high' CHECK (confidence IN ('high','medium','low')),
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (source_id, target_id, predicate, chunk_id)
);
CREATE INDEX edges_source_idx ON edges (source_id);
CREATE INDEX edges_target_idx ON edges (target_id);
CREATE INDEX edges_predicate_idx ON edges (predicate);
CREATE INDEX edges_doc_idx ON edges (document_id);

CREATE TABLE merges (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kept_id      UUID REFERENCES nodes(id) ON DELETE CASCADE,
    merged_id    UUID,
    merged_name  TEXT,
    similarity   FLOAT,
    method       TEXT CHECK (method IN ('auto','llm','manual')),
    decided_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE ingestion_jobs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   UUID REFERENCES documents(id) ON DELETE CASCADE,
    stage         TEXT,
    chunks_total  INT DEFAULT 0,
    chunks_done   INT DEFAULT 0,
    triples_found INT DEFAULT 0,
    nodes_created INT DEFAULT 0,
    edges_created INT DEFAULT 0,
    started_at    TIMESTAMPTZ DEFAULT now(),
    finished_at   TIMESTAMPTZ
);

CREATE TABLE queries (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text    TEXT,
    path_taken    TEXT CHECK (path_taken IN ('graph','chunk','hybrid')),
    nodes_visited UUID[],
    chunk_ids     UUID[],
    hops          INT,
    answer        TEXT,
    latency_ms    INT,
    asked_at      TIMESTAMPTZ DEFAULT now()
);

-- ── GRAPH COMMUNITIES (step 7; also in migrations/002_communities.sql for existing databases) ──
CREATE TABLE communities (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title      TEXT NOT NULL,
    summary    TEXT NOT NULL,
    node_ids   UUID[] NOT NULL,
    size       INT NOT NULL,
    embedding  vector(384),
    created_at TIMESTAMPTZ DEFAULT now()
);

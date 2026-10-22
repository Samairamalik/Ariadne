-- Step 7: graph communities (groups of densely connected nodes) with an LLM-written summary each.
-- Apply to an existing database:  docker exec -i ariadne-db psql -U ariadne < backend/db/migrations/002_communities.sql
CREATE TABLE IF NOT EXISTS communities (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title      TEXT NOT NULL,
    summary    TEXT NOT NULL,
    node_ids   UUID[] NOT NULL,
    size       INT NOT NULL,
    embedding  vector(384),
    created_at TIMESTAMPTZ DEFAULT now()
);

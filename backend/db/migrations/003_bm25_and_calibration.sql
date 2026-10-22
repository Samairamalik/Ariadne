-- 1) BM25 needs each chunk's length (in lexemes, counting repeats) for length normalisation.
-- 2) Per-document relevance thresholds, calibrated from synthetic questions (see app/retrieval/calibrate.py).
-- Apply to an existing database:  docker exec -i ariadne-db psql -U ariadne < backend/db/migrations/003_bm25_and_calibration.sql
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_len INT;
UPDATE chunks
   SET doc_len = GREATEST(1, (SELECT coalesce(sum(coalesce(array_length(u.positions, 1), 1)), 1) FROM unnest(chunks.tsv) u))
 WHERE doc_len IS NULL;

ALTER TABLE documents ADD COLUMN IF NOT EXISTS relevance_threshold DOUBLE PRECISION;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS calibrated_at TIMESTAMPTZ;

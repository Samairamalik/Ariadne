# Ariadne backend

```powershell
docker compose up -d db                      # from repo root; Postgres+pgvector on host port 5433
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
python -m pytest tests -q                    # 33 unit tests, no services needed
python tests/integration/e2e.py              # needs the API + DB running
```

Useful endpoints: `POST /api/documents/upload`, `GET /api/jobs/{job_id}/status` (SSE),
`GET /api/documents/{id}/chunks` (inspect chunk quality), `POST /api/search` (retrieval only, with scores and
`found_by`), `POST /api/query` (answer + validated citations), `GET /api/health` (includes reranker status).

Deviations from `ariadne-design.md` are marked `DEVIATION` in `db/schema.sql` and in module docstrings.

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app import db
from app.config import settings
from app.ingestion.pipeline import run_ingestion

router = APIRouter(prefix="/api")
_tasks: set[asyncio.Task] = set()  # hold references so background tasks aren't garbage-collected
_TYPES = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown", ".txt": "text"}
MAX_BYTES = 100 * 1024 * 1024


@router.post("/documents/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _TYPES:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Supported: {', '.join(_TYPES)}")
    data = await file.read()
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(413 if data else 400, "File is empty or larger than 100 MB.")
    digest = hashlib.sha256(data).hexdigest()

    async with db.pool().acquire() as conn:
        existing = await conn.fetchrow(
            """SELECT d.id, d.status, (SELECT id FROM ingestion_jobs j WHERE j.document_id = d.id
                                       ORDER BY started_at DESC LIMIT 1) AS job_id
               FROM documents d WHERE content_hash = $1""", digest)
        if existing and existing["status"] != "failed":
            return {"document_id": existing["id"], "job_id": existing["job_id"], "duplicate": True}

        doc_id = existing["id"] if existing else uuid4()
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / f"{doc_id}{ext}"
        path.write_bytes(data)
        if existing:  # retry of a failed ingest of identical bytes
            await conn.execute("UPDATE documents SET status = 'pending', error = NULL WHERE id = $1", doc_id)
        else:
            await conn.execute(
                "INSERT INTO documents (id, filename, content_hash, doc_type) VALUES ($1,$2,$3,$4)",
                doc_id, file.filename, digest, _TYPES[ext])
        job_id = await conn.fetchval("INSERT INTO ingestion_jobs (document_id, stage) VALUES ($1,'pending') RETURNING id", doc_id)

    task = asyncio.create_task(run_ingestion(doc_id, job_id, path, _TYPES[ext]))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"document_id": doc_id, "job_id": job_id, "duplicate": False}


@router.get("/documents")
async def list_documents():
    async with db.pool().acquire() as conn:
        rows = await conn.fetch(
            """SELECT d.id, d.filename, d.title, d.author, d.doc_type, d.page_count, d.token_count, d.status, d.error,
                      d.ingested_at, (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count
               FROM documents d ORDER BY d.ingested_at DESC""")
    return [dict(r) for r in rows]


@router.get("/documents/{doc_id}/chunks")
async def document_chunks(doc_id: UUID, limit: int = 200, offset: int = 0):
    """Inspect what the chunker produced — essential for judging retrieval quality."""
    async with db.pool().acquire() as conn:
        meta = await conn.fetchval("SELECT metadata FROM documents WHERE id = $1", doc_id)
        if meta is None:
            raise HTTPException(404, "Document not found")
        rows = await conn.fetch(
            """SELECT id, chunk_index, kind, section_path, page_start, page_end, token_count, text FROM chunks
               WHERE document_id = $1 ORDER BY chunk_index LIMIT $2 OFFSET $3""", doc_id, limit, offset)
    return {"stats": json.loads(meta) if isinstance(meta, str) else meta, "chunks": [dict(r) for r in rows]}


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: UUID):
    async with db.pool().acquire() as conn:
        row = await conn.fetchrow("DELETE FROM documents WHERE id = $1 RETURNING doc_type", doc_id)
    if not row:
        raise HTTPException(404, "Document not found")
    for p in Path(settings.upload_dir).glob(f"{doc_id}.*"):
        p.unlink(missing_ok=True)


@router.get("/jobs/{job_id}/status")
async def job_status(job_id: UUID):
    """Server-sent events: one JSON message per progress change, closing when the job finishes or fails."""
    async def stream():
        last = None
        while True:
            async with db.pool().acquire() as conn:
                row = await conn.fetchrow(
                    """SELECT j.stage, j.chunks_total, j.chunks_done, j.finished_at, d.status, d.error
                       FROM ingestion_jobs j JOIN documents d ON d.id = j.document_id WHERE j.id = $1""", job_id)
            if not row:
                yield f"event: error\ndata: {json.dumps({'error': 'job not found'})}\n\n"
                return
            payload = {k: v for k, v in dict(row).items() if k != "finished_at"}
            if payload != last:
                last = payload
                yield f"data: {json.dumps(payload)}\n\n"
            if row["status"] in ("done", "failed"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")

import logging
import os
import threading
from contextlib import asynccontextmanager

from app.config import settings

if settings.embed_device == "cpu" and settings.rerank_device == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db, nli_worker
from app.embeddings import get_embedder
from app.ingestion.pipeline import recover_interrupted
from app.retrieval import rerank
from app.routers import documents, graph, graph_views, llm, query

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.init_pool()
    if n := await recover_interrupted():
        logging.getLogger("ariadne").warning("closed %d graph job(s) interrupted by a restart", n)
    rerank.start_loading()
    threading.Thread(target=get_embedder, name="embedder-load", daemon=True).start()
    if settings.nli_warmup:
        nli_worker.warm_up()  # starts the NLI worker process and loads its model before the first question
    yield
    nli_worker.reset()
    await db.close_pool()


app = FastAPI(title="Ariadne", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(graph.router)
app.include_router(graph_views.router)
app.include_router(llm.router)


@app.get("/api/health")
async def health():
    async with db.pool().acquire() as conn:
        return {"ok": await conn.fetchval("SELECT 1") == 1, "reranker": rerank.status()}

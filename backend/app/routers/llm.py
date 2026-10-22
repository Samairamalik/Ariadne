import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.llm import LLMUnavailable, get_router
from app.llm.router import VALID_MODES

router = APIRouter(prefix="/api/llm", tags=["llm"])


class LLMConfigIn(BaseModel):
    mode: str | None = None  # offline | cloud
    ollama_model: str | None = None  # must be installed in Ollama
    cloud_order: list[str] | None = None  # e.g. ["gemini", "groq"]


@router.get("")
async def llm_status():
    """Current mode, which providers would be tried, installed Ollama models, which cloud keys are set (not the keys)."""
    return await get_router().status()


@router.put("")
async def llm_configure(cfg: LLMConfigIn):
    r = get_router()
    if cfg.mode is not None:
        if cfg.mode not in VALID_MODES:
            raise HTTPException(422, f"mode must be one of {VALID_MODES}")
        r.mode = cfg.mode
    if cfg.ollama_model is not None:
        st = await r.status()
        if st["ollama"]["installed"] and cfg.ollama_model not in st["ollama"]["installed"]:
            raise HTTPException(422, f"'{cfg.ollama_model}' is not installed in Ollama. Installed: {st['ollama']['installed']}")
        r.ollama_model = cfg.ollama_model
    if cfg.cloud_order is not None:
        bad = [p for p in cfg.cloud_order if p not in ("groq", "gemini")]
        if bad:
            raise HTTPException(422, f"unknown cloud provider(s): {bad}")
        r.cloud_order = cfg.cloud_order
    r.save()
    return await r.status()


@router.post("/test")
async def llm_test():
    """One tiny round trip through the router, so you can see which provider actually answers."""
    t0 = time.perf_counter()
    try:
        res = await get_router().chat([{"role": "user", "content": "Reply with the single word: ready"}], timeout=180)
    except LLMUnavailable as e:
        return {"ok": False, "error": str(e)[:600]}
    return {"ok": True, "provider": res.provider, "model": res.model, "reply": res.text.strip()[:80],
            "latency_ms": int((time.perf_counter() - t0) * 1000)}

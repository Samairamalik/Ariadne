"""Cross-encoder reranking with non-blocking model loading.

The model is loaded in a background thread at startup. Requests never wait on a download: until the model is
ready (or if it can never load, e.g. offline first run) they get the RRF-fused order and `reranked=False`, which
the API reports so the caller knows the precision stage was skipped.
"""
import logging
import threading

from app.config import settings
from app.retrieval.hybrid import Candidate

log = logging.getLogger("ariadne.rerank")
_state: dict = {"status": "idle", "model": None, "error": None}  # idle | loading | ready | unavailable
_lock = threading.Lock()


def _load() -> None:
    try:
        from sentence_transformers import CrossEncoder

        device = None if settings.rerank_device == "auto" else settings.rerank_device
        try:
            _state["model"] = CrossEncoder(settings.rerank_model, max_length=512, device=device)
        except RuntimeError as e:  # CUDA unavailable/OOM at load: the model is tiny, CPU is fine
            log.warning("reranker load on %s failed (%s); using CPU", device, str(e).splitlines()[0])
            _state["model"] = CrossEncoder(settings.rerank_model, max_length=512, device="cpu")
        _state["status"] = "ready"
        log.info("reranker ready: %s", settings.rerank_model)
    except Exception as e:  # noqa: BLE001
        _state["status"], _state["error"] = "unavailable", f"{type(e).__name__}: {e}"
        log.warning("reranker unavailable, falling back to RRF order: %s", _state["error"])


def start_loading() -> None:
    with _lock:
        if _state["status"] != "idle":
            return
        _state["status"] = "loading"
    threading.Thread(target=_load, name="reranker-load", daemon=True).start()


def status() -> str:
    return _state["status"]


def order_with_preference(pool: list[Candidate], prefer: set | None, bonus: float) -> list[Candidate]:
    """Sort by rerank score, adding `bonus` to candidates from preferred documents for ORDERING only: the stored
    rerank_score stays the raw cross-encoder logit, which the relevance gate reads."""
    prefer = prefer or set()
    return sorted(pool, key=lambda c: c.rerank_score + (bonus if c.document_id in prefer else 0.0), reverse=True)


def rerank(question: str, candidates: list[Candidate], top_k: int | None = None,
           prefer: set | None = None, bonus: float = 0.0) -> tuple[list[Candidate], bool]:
    """Returns (top_k candidates, whether the cross-encoder was actually applied).

    The cross-encoder sees (question, passage) jointly, which the bi-encoder cannot — the main precision gain in
    the pipeline. Scores are raw logits: comparable within one query only. `prefer`/`bonus` favour the documents the
    question names (see doc_aware.py)."""
    top_k = top_k or settings.rerank_top_k
    pool = candidates[: settings.rerank_candidates]
    if not pool:
        return [], status() == "ready"
    if status() != "ready":
        return pool[:top_k], False
    scores = _state["model"].predict(
        [(question, f"{c.section_path}. {c.text}" if c.section_path else c.text) for c in pool],
        show_progress_bar=False)
    for c, s in zip(pool, scores):
        c.rerank_score = float(s)
    return order_with_preference(pool, prefer, bonus)[:top_k], True

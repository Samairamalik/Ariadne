"""Run the NLI check in its own process.

The NLI model is a ~740 MB PyTorch model. Run inside the API process, a native crash or a memory spike in it takes the
whole server down (three evaluation runs died that way). In a separate worker process the API stays up: if the worker
dies or stalls, that one answer simply comes back with verification marked unavailable, and the next question gets a
fresh worker. The model is loaded once per worker, not per request.
"""
import asyncio
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

log = logging.getLogger("ariadne.nli")
TIMEOUT_S = 90
_pool: ProcessPoolExecutor | None = None


def _warm() -> bool:
    """Runs in the worker: load the model now so the first real check is fast."""
    from app import nli

    model, _ = nli.load_model()
    return model is not None


def verify_in_worker(answer_text: str, source_texts: list[str]) -> dict:
    """Runs in the worker: the whole check, returned as plain data."""
    from app import nli

    model, labels = nli.load_model()
    if model is None:
        return {"enabled": True, "available": False, "reason": nli.status()["error"]}
    return {"enabled": True, **nli.summarize(nli.verify_answer(answer_text, source_texts, model, labels))}


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
    return _pool


def reset() -> None:
    """Drop a broken worker; the next call starts a fresh one."""
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def warm_up() -> None:
    """Start the worker and begin loading the model, without waiting for it."""
    try:
        _get_pool().submit(_warm)
    except Exception as e:  # noqa: BLE001
        log.warning("could not start the NLI worker: %s", e)


async def verify(answer_text: str, source_texts: list[str]) -> dict:
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_get_pool(), verify_in_worker, answer_text, source_texts), TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 - a crashed, stalled or unstartable worker must never fail the answer
        log.warning("NLI worker failed (%s: %s); answer returned unverified", type(e).__name__, e)
        reset()
        return {"enabled": True, "available": False, "reason": f"verification worker failed: {type(e).__name__}"}

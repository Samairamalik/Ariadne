import asyncio
from concurrent.futures import ProcessPoolExecutor

from app import nli, nli_worker


def test_worker_function_reports_unavailable_when_the_model_cannot_load(monkeypatch):
    monkeypatch.setattr(nli, "load_model", lambda: (None, None))
    monkeypatch.setitem(nli._state, "error", "boom")
    out = nli_worker.verify_in_worker("A cited claim with enough words [1].", ["source"])
    assert out == {"enabled": True, "available": False, "reason": "boom"}


def test_a_broken_worker_never_fails_the_answer(monkeypatch):
    class Broken:
        def submit(self, *a, **k):
            raise RuntimeError("pool is dead")

    monkeypatch.setattr(nli_worker, "_pool", None)
    monkeypatch.setattr(nli_worker, "_get_pool", lambda: Broken())

    async def fake_executor(*a, **k):
        raise RuntimeError("worker crashed")

    out = asyncio.run(nli_worker.verify("A cited claim [1].", ["source"]))
    assert out["enabled"] and out["available"] is False and "worker failed" in out["reason"]


def test_reset_discards_the_pool_so_the_next_call_starts_fresh():
    nli_worker._pool = ProcessPoolExecutor(max_workers=1)
    nli_worker.reset()
    assert nli_worker._pool is None

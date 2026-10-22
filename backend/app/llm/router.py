"""Mode-aware LLM router with fallback.

  offline -> [ollama]                              nothing leaves the machine
  cloud   -> [groq, gemini, ...] (+ ollama last)    first provider that is configured AND succeeds wins

Runtime choices (mode, Ollama model, cloud order) are persisted to a small JSON file; API keys never are.
"""
import json
import logging
import time
from pathlib import Path

from app.config import settings
from app.llm.providers import (GeminiProvider, GroqProvider, LLMError, LLMResult, LLMUnavailable, Messages,
                               OllamaProvider, Provider, timed)

log = logging.getLogger("ariadne.llm")
VALID_MODES = ("offline", "cloud")


class LLMRouter:
    def __init__(self, extra_providers: dict[str, Provider] | None = None):
        self.mode = settings.llm_mode
        self.ollama_model = settings.llm_model
        self.cloud_order = [p.strip() for p in settings.cloud_order.split(",") if p.strip()]
        self.providers: dict[str, Provider] = {
            "ollama": OllamaProvider(lambda: self.ollama_model),
            "groq": GroqProvider(),
            "gemini": GeminiProvider(),
            **(extra_providers or {}),
        }
        self._load()

    # ── persistence (no secrets) ────────────────────────────────────
    def _path(self) -> Path:
        return Path(settings.llm_settings_path)

    def _load(self) -> None:
        try:
            d = json.loads(self._path().read_text())
            if d.get("mode") in VALID_MODES:
                self.mode = d["mode"]
            self.ollama_model = d.get("ollama_model") or self.ollama_model
            order = [p for p in d.get("cloud_order", []) if p in self.providers and p != "ollama"]
            self.cloud_order = order or self.cloud_order
        except (OSError, ValueError):
            pass

    def save(self) -> None:
        p = self._path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"mode": self.mode, "ollama_model": self.ollama_model, "cloud_order": self.cloud_order}))

    # ── routing ─────────────────────────────────────────────────────
    def plan(self) -> list[Provider]:
        """Providers that would be tried, in order, for the current mode."""
        if self.mode == "offline":
            return [self.providers["ollama"]]
        chain = [self.providers[n] for n in self.cloud_order if n in self.providers and self.providers[n].configured()]
        if settings.cloud_allow_local_fallback:
            chain.append(self.providers["ollama"])
        return chain

    async def chat(self, messages: Messages, *, json_mode: bool = False, temperature: float = 0.0,
                   timeout: float = 600) -> LLMResult:
        chain = self.plan()
        if not chain:
            raise LLMUnavailable("No LLM available: cloud mode has no configured provider "
                                 "(set ARIADNE_GROQ_API_KEY or ARIADNE_GEMINI_API_KEY) and local fallback is off.")
        errors = []
        for p in chain:
            t0 = time.perf_counter()
            try:
                text = await p.chat(messages, json_mode=json_mode, temperature=temperature, timeout=timeout)
                return timed(p, text, t0)
            except LLMError as e:  # auth, rate limit, outage, bad model name...: all reasons to try the next one
                log.warning("LLM provider failed, trying next: %s", e)
                errors.append(str(e))
        raise LLMUnavailable(" | ".join(errors))

    # ── status for the API/UI (never includes keys) ─────────────────
    async def status(self) -> dict:
        try:
            models, ollama_error = await OllamaProvider.list_models(), None
        except Exception as e:  # noqa: BLE001
            models, ollama_error = [], f"{type(e).__name__}: Ollama not reachable at {settings.ollama_url}"
        return {
            "mode": self.mode,
            "will_try": [p.name for p in self.plan()],
            "sends_text_off_machine": any(p.name != "ollama" for p in self.plan()),
            "ollama": {"selected": self.ollama_model, "installed": models, "error": ollama_error},
            "cloud": {
                "order": self.cloud_order,
                "allow_local_fallback": settings.cloud_allow_local_fallback,
                "providers": {n: {"configured": self.providers[n].configured(), "model": self.providers[n].model()}
                              for n in ("groq", "gemini")},
            },
        }


_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router

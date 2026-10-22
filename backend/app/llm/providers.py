"""LLM providers behind one tiny interface. Plain httpx, no vendor SDKs, so there is nothing extra to install and
each request format is visible in one place.

Every provider supports `json_mode` (needed for triple extraction): Ollama `format: json`, Groq
`response_format: json_object`, Gemini `responseMimeType: application/json`.
"""
import logging
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import settings

log = logging.getLogger("ariadne.llm")

Messages = list[dict]  # [{"role": "system"|"user"|"assistant", "content": str}]


class LLMError(Exception):
    def __init__(self, provider: str, message: str, status: int | None = None):
        super().__init__(f"{provider}: {message}")
        self.provider, self.status = provider, status


class LLMUnavailable(Exception):
    """Raised by the router when every provider it was allowed to try has failed."""


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: int


class Provider(Protocol):
    name: str

    def configured(self) -> bool: ...
    def model(self) -> str: ...
    async def chat(self, messages: Messages, *, json_mode: bool, temperature: float, timeout: float) -> str: ...


def _err_text(r: httpx.Response) -> str:
    try:
        j = r.json()
        if isinstance(j, dict):
            e = j.get("error", j)
            return str(e.get("message", e) if isinstance(e, dict) else e)[:300]
    except Exception:  # noqa: BLE001
        pass
    return r.text[:300]


class OllamaProvider:
    name = "ollama"

    def __init__(self, get_model):
        self._get_model = get_model

    def configured(self) -> bool:
        return True

    def model(self) -> str:
        return self._get_model()

    async def chat(self, messages, *, json_mode, temperature, timeout) -> str:
        payload = {"model": self.model(), "stream": False, "messages": messages,
                   "options": {"temperature": temperature, "num_ctx": 8192}}
        if json_mode:
            payload["format"] = "json"
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(f"{settings.ollama_url}/api/chat", json=payload)
                if r.status_code >= 500:
                    # e.g. "CUDA device busy/unavailable": a small model still runs on CPU
                    log.warning("Ollama failed on default device (%s); retrying on CPU", _err_text(r)[:120])
                    payload["options"]["num_gpu"] = 0
                    r = await c.post(f"{settings.ollama_url}/api/chat", json=payload)
        except httpx.HTTPError as e:
            raise LLMError(self.name, f"cannot reach Ollama at {settings.ollama_url} ({type(e).__name__})") from e
        if r.status_code >= 400:
            raise LLMError(self.name, _err_text(r), r.status_code)
        return r.json()["message"]["content"]

    @staticmethod
    async def list_models() -> list[str]:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{settings.ollama_url}/api/tags")
            r.raise_for_status()
            return sorted(m["name"] for m in r.json().get("models", []))


class GroqProvider:
    name = "groq"

    def configured(self) -> bool:
        return bool(settings.groq_api_key)

    def model(self) -> str:
        return settings.groq_model

    async def chat(self, messages, *, json_mode, temperature, timeout) -> str:
        body = {"model": self.model(), "messages": messages, "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post("https://api.groq.com/openai/v1/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {settings.groq_api_key}"})
        except httpx.HTTPError as e:
            raise LLMError(self.name, f"network error ({type(e).__name__})") from e
        if r.status_code >= 400:
            raise LLMError(self.name, _err_text(r), r.status_code)
        return r.json()["choices"][0]["message"]["content"]


class GeminiProvider:
    name = "gemini"

    def configured(self) -> bool:
        return bool(settings.gemini_api_key)

    def model(self) -> str:
        return settings.gemini_model

    async def chat(self, messages, *, json_mode, temperature, timeout) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        gen = {"temperature": temperature}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {"contents": contents, "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model()}:generateContent"
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                # key in a header, not the URL, so it can't leak into logs
                r = await c.post(url, json=body, headers={"x-goog-api-key": settings.gemini_api_key})
        except httpx.HTTPError as e:
            raise LLMError(self.name, f"network error ({type(e).__name__})") from e
        if r.status_code >= 400:
            raise LLMError(self.name, _err_text(r), r.status_code)
        try:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LLMError(self.name, "empty or blocked response") from e


def timed(provider: Provider, text: str, t0: float) -> LLMResult:
    return LLMResult(text, provider.name, provider.model(), int((time.perf_counter() - t0) * 1000))

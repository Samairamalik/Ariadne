import pytest

from app.config import settings
from app.llm import LLMRouter, LLMUnavailable
from app.llm.providers import LLMError


class Fake:
    def __init__(self, name, *, configured=True, fail=None, reply="ok"):
        self.name, self._configured, self.fail, self.reply, self.calls = name, configured, fail, reply, 0

    def configured(self):
        return self._configured

    def model(self):
        return f"{self.name}-model"

    async def chat(self, messages, *, json_mode, temperature, timeout):
        self.calls += 1
        if self.fail:
            raise LLMError(self.name, self.fail, 429)
        return self.reply


@pytest.fixture
def make(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "llm_settings_path", str(tmp_path / "llm.json"))
    monkeypatch.setattr(settings, "cloud_allow_local_fallback", True)

    def _make(mode, **fakes):
        r = LLMRouter(extra_providers=fakes)
        r.providers.update(fakes)
        r.mode, r.cloud_order = mode, ["groq", "gemini"]
        return r
    return _make


async def test_offline_mode_only_ever_uses_ollama(make):
    ollama, groq = Fake("ollama", reply="local"), Fake("groq")
    r = make("offline", ollama=ollama, groq=groq)
    res = await r.chat([{"role": "user", "content": "hi"}])
    assert (res.provider, res.text) == ("ollama", "local") and groq.calls == 0
    assert [p.name for p in r.plan()] == ["ollama"]


async def test_cloud_falls_back_groq_then_gemini_then_local(make):
    groq, gemini, ollama = Fake("groq", fail="rate limited"), Fake("gemini", fail="bad key"), Fake("ollama", reply="local")
    r = make("cloud", groq=groq, gemini=gemini, ollama=ollama)
    res = await r.chat([{"role": "user", "content": "hi"}])
    assert res.provider == "ollama" and (groq.calls, gemini.calls, ollama.calls) == (1, 1, 1)


async def test_cloud_uses_first_working_provider_and_stops(make):
    groq, gemini = Fake("groq", reply="from groq"), Fake("gemini")
    r = make("cloud", groq=groq, gemini=gemini, ollama=Fake("ollama"))
    assert (await r.chat([{"role": "user", "content": "x"}])).provider == "groq" and gemini.calls == 0


async def test_unconfigured_cloud_providers_are_skipped(make):
    groq = Fake("groq", configured=False)
    r = make("cloud", groq=groq, gemini=Fake("gemini", configured=False), ollama=Fake("ollama"))
    assert [p.name for p in r.plan()] == ["ollama"]
    await r.chat([{"role": "user", "content": "x"}])
    assert groq.calls == 0


async def test_everything_failing_raises_with_every_reason(make, monkeypatch):
    monkeypatch.setattr(settings, "cloud_allow_local_fallback", False)
    r = make("cloud", groq=Fake("groq", fail="rate limited"), gemini=Fake("gemini", fail="bad key"), ollama=Fake("ollama"))
    with pytest.raises(LLMUnavailable) as e:
        await r.chat([{"role": "user", "content": "x"}])
    assert "rate limited" in str(e.value) and "bad key" in str(e.value)


async def test_cloud_with_no_keys_and_no_local_fallback_is_a_clear_error(make, monkeypatch):
    monkeypatch.setattr(settings, "cloud_allow_local_fallback", False)
    r = make("cloud", groq=Fake("groq", configured=False), gemini=Fake("gemini", configured=False))
    with pytest.raises(LLMUnavailable, match="ARIADNE_GROQ_API_KEY"):
        await r.chat([{"role": "user", "content": "x"}])


def test_settings_persist_without_secrets(make, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "groq_api_key", "gsk_SECRET")
    r = make("offline")
    r.mode, r.ollama_model = "cloud", "llama3.1:8b"
    r.save()
    raw = (tmp_path / "llm.json").read_text()
    assert "SECRET" not in raw and "gsk_" not in raw
    r2 = LLMRouter()
    assert (r2.mode, r2.ollama_model) == ("cloud", "llama3.1:8b")

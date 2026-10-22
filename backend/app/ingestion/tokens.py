"""Token counting that matches the embedder.

bge-small truncates input silently at 512 WordPiece tokens. Counting with a different tokenizer (e.g. tiktoken)
would let chunks exceed that without anyone noticing, and the tail of the chunk would simply not be embedded.
"""
import re
from functools import lru_cache
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class ApproxTokenCounter:
    """Dependency-free fallback/test counter: words + punctuation, inflated ~1.3x for subword splitting."""

    _pat = re.compile(r"\w+|[^\w\s]")

    def count(self, text: str) -> int:
        return int(len(self._pat.findall(text)) * 1.3) + 1 if text.strip() else 0


class HFTokenCounter:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        return len(self._tok(text, add_special_tokens=False, truncation=False)["input_ids"])


@lru_cache(maxsize=1)
def get_token_counter() -> TokenCounter:
    from app.config import settings

    try:
        return HFTokenCounter(settings.embed_model)
    except Exception:  # offline and model not cached yet
        return ApproxTokenCounter()

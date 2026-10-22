"""bge-small-en-v1.5 wrapper.

BGE v1.5 is asymmetric: *queries* get the retrieval instruction prefix, *passages* get none. The design doc
prefixed passages too (and said to use the same prefix at query time); that mismatches how the model was trained
and degrades similarity. Vectors are L2-normalised so cosine distance (pgvector `<=>`) is well-behaved.

Device: "auto" uses CUDA when present, but a 33M-parameter model must never fail an ingestion because a GPU is
busy or memory-constrained (seen in practice: an 84 MiB allocation failing with 6.5 GiB reported free). On any CUDA
error the model is moved to CPU permanently and the call is retried.
"""
import logging
from functools import lru_cache

import numpy as np

from app.config import settings

log = logging.getLogger("ariadne.embed")
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        device = None if settings.embed_device == "auto" else settings.embed_device
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except RuntimeError as e:  # CUDA OOM / busy at load time (seen with a "6.9 GiB free" GPU failing a 46 MiB alloc)
            log.warning("loading %s on %s failed (%s); using CPU", model_name, device or "auto", str(e).splitlines()[0])
            self.model = SentenceTransformer(model_name, device="cpu")
        self.model.max_seq_length = settings.embed_max_tokens

    def _encode(self, *args, **kwargs) -> np.ndarray:
        try:
            return self.model.encode(*args, normalize_embeddings=True, show_progress_bar=False, **kwargs)
        except RuntimeError as e:  # torch.cuda.OutOfMemoryError subclasses RuntimeError; also CUDA launch failures
            if "cuda" not in str(e).lower() and "out of memory" not in str(e).lower():
                raise
            log.warning("CUDA failure during embedding (%s); falling back to CPU", str(e).splitlines()[0])
            import torch

            torch.cuda.empty_cache()
            self.model.to("cpu")
            return self.model.encode(*args, normalize_embeddings=True, show_progress_bar=False, **kwargs)

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode(QUERY_INSTRUCTION + query)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder(settings.embed_model)

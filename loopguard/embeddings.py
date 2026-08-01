"""Embeddings support for semantic detection (Detector A).

Detector A needs to ask "are these two agent decisions *similar*?" even when they are
not byte-identical. The standard way: turn each text into a vector (an "embedding") via
an embedding model, then measure the angle between vectors (cosine similarity).

We use OpenAI's embedding API for the vectors. The Embedder interface means you can swap
in a different backend (local sentence-transformers, etc.) without touching the detector.
"""

from __future__ import annotations

import math
import os
from typing import Protocol


class Embedder(Protocol):
    """The contract Detector A depends on: anything with encode() works."""

    def encode(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """Embeds text with OpenAI's embedding API.

    The client is created lazily on first use (so importing this module never requires
    a key), and embeddings are cached by text so we never pay to embed the same string
    twice within a run.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model
        self._client = None
        self._cache: dict[str, list[float]] = {}

    @property
    def client(self):
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to your environment or a .env file "
                    "to use semantic detection."
                )
            from openai import OpenAI  # imported lazily so the base demo needs no openai install

            self._client = OpenAI()
        return self._client

    def encode(self, text: str) -> list[float]:
        if text not in self._cache:
            resp = self.client.embeddings.create(model=self.model, input=text)
            self._cache[text] = resp.data[0].embedding
        return self._cache[text]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors. 1.0 = same direction, 0.0 = unrelated.
    OpenAI vectors are not unit-length, so we normalize by both magnitudes."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

"""FinBERT — finance-specific sentiment (free, local, offline after first load).

Uses HuggingFace `ProsusAI/finbert` (a BERT fine-tuned on financial text) via
`transformers` + `torch`. One-time ~450MB model download, then runs locally with
NO API cost. Fed the free Google News RSS headlines (src/ai/news.py).

Import-guarded and lazy: if `transformers`/`torch` aren't installed or the model
isn't downloaded yet, `available` is False and callers fall back to Gemini/neutral
— nothing breaks. Returns a bias in {-1, 0, +1} plus a 0..1 confidence.
"""
from __future__ import annotations

import time
from typing import Optional


class FinBertSentiment:
    _MODEL = "ProsusAI/finbert"

    def __init__(self, ttl_seconds: float = 1800):
        self.ttl = ttl_seconds
        self._pipe = None
        self._tried = False
        self._cache: dict[str, tuple[float, int, float]] = {}  # sym -> (ts, bias, conf)

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._pipe is not None or self._tried:
            return self._pipe
        self._tried = True
        try:
            from transformers import pipeline  # noqa: PLC0415

            self._pipe = pipeline("sentiment-analysis", model=self._MODEL,
                                  tokenizer=self._MODEL, truncation=True)
        except Exception:
            self._pipe = None
        return self._pipe

    def sentiment_bias(self, symbol: str, headlines: Optional[list[str]] = None) -> int:
        """-1 / 0 / +1 aggregate bias for a stock from its headlines."""
        return self.score(symbol, headlines)[0]

    def score(self, symbol: str, headlines: Optional[list[str]] = None) -> tuple[int, float]:
        cached = self._cache.get(symbol)
        if cached and (time.time() - cached[0] < self.ttl):
            return cached[1], cached[2]

        pipe = self._load()
        if pipe is None:
            return 0, 0.0

        if headlines is None:
            try:
                from .news import fetch_headlines
                headlines = fetch_headlines(symbol)
            except Exception:
                headlines = []
        if not headlines:
            self._cache[symbol] = (time.time(), 0, 0.0)
            return 0, 0.0

        try:
            results = pipe(headlines[:10])
        except Exception:
            return 0, 0.0

        # Aggregate: +1 positive, -1 negative, 0 neutral, weighted by confidence.
        net = 0.0
        conf_sum = 0.0
        for r in results:
            label = r.get("label", "").lower()
            score = float(r.get("score", 0.0))
            if label == "positive":
                net += score
            elif label == "negative":
                net -= score
            conf_sum += score
        bias = 1 if net > 0.15 else (-1 if net < -0.15 else 0)
        conf = round(conf_sum / max(len(results), 1), 3)
        self._cache[symbol] = (time.time(), bias, conf)
        return bias, conf

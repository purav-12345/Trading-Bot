"""Gemini helper — the ONLY place we call the LLM during/around trading.

Strict cost controls:
  * Model defaults to gemini-2.5-flash-lite (cheapest tier).
  * Hard daily call cap (GEMINI_MAX_CALLS_PER_DAY); once hit, every method
    degrades gracefully (returns neutral) instead of spending money.
  * 30-min in-memory cache for sentiment so the tick loop never spams the API.
  * Never called inside the per-tick order decision unless a strategy
    explicitly opts in via `use_ai_bias`, and even then it's cached + capped.
"""
from __future__ import annotations

import json
import time
from datetime import date
from typing import Any, Optional


class GeminiHelper:
    def __init__(self, config):
        self.config = config
        self.model = config.secrets.gemini_model or "gemini-2.5-flash-lite"
        self.max_calls = config.secrets.gemini_max_calls_per_day
        self._client = None
        self._calls_date = date.today()
        self._calls = 0
        self._sentiment_cache: dict[str, tuple[float, int]] = {}  # symbol -> (ts, bias)

    @property
    def available(self) -> bool:
        return self.config.secrets.has_gemini

    def _client_or_none(self):
        if not self.available:
            return None
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.config.secrets.gemini_api_key)
            except Exception:
                return None
        return self._client

    def _budget_ok(self) -> bool:
        if self._calls_date != date.today():
            self._calls_date = date.today()
            self._calls = 0
        return self._calls < self.max_calls

    def _generate(self, prompt: str, system: str = "") -> Optional[str]:
        client = self._client_or_none()
        if client is None or not self._budget_ok():
            return None
        try:
            self._calls += 1
            contents = (system + "\n\n" + prompt) if system else prompt
            resp = client.models.generate_content(model=self.model, contents=contents)
            return (resp.text or "").strip()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Used by SupertrendORBStrategy as a VETO only (never an initiator).
    # Returns +1 (positive), 0 (neutral/unknown), -1 (negative).
    # ------------------------------------------------------------------
    def sentiment_bias(self, symbol: str, headlines: Optional[list[str]] = None) -> int:
        cached = self._sentiment_cache.get(symbol)
        if cached and (time.time() - cached[0] < 1800):
            return cached[1]

        # Pull real, free headlines from Google News RSS if none supplied.
        if headlines is None:
            try:
                from .news import fetch_headlines

                headlines = fetch_headlines(symbol)
            except Exception:
                headlines = []
        # No news + no way to judge -> stay neutral (don't waste an API call).
        if not headlines:
            self._sentiment_cache[symbol] = (time.time(), 0)
            return 0

        hl = "\n".join(headlines)
        prompt = (
            f"Stock: {symbol} (NSE India).\n"
            f"Recent headlines (may be empty):\n{hl or '(none provided)'}\n\n"
            "Classify the SHORT-TERM news sentiment for an intraday long trade. "
            "Reply with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL."
        )
        out = self._generate(prompt, system="You are a terse financial news classifier.")
        bias = 0
        if out:
            u = out.upper()
            if "POSITIVE" in u:
                bias = 1
            elif "NEGATIVE" in u:
                bias = -1
        self._sentiment_cache[symbol] = (time.time(), bias)
        return bias

    # ------------------------------------------------------------------
    # End-of-day plain-English summary for the notification.
    # ------------------------------------------------------------------
    def eod_summary(self, stats: dict[str, Any]) -> str:
        fallback = (
            f"EOD: P&L Rs.{stats.get('total_pnl', 0):,.0f} | "
            f"trades {stats.get('trade_count', 0)} | "
            f"status {stats.get('status', 'n/a')}"
        )
        out = self._generate(
            "Summarise this trading day in 2-3 sentences for the trader. "
            "Be factual, mention P&L, number of trades, and whether a limit "
            f"was hit. Data:\n{json.dumps(stats, default=str)}",
            system="You are a concise trading-desk assistant. No financial advice.",
        )
        return out or fallback

    # ------------------------------------------------------------------
    # Free-form helper used by the LangGraph agent for reasoning.
    # ------------------------------------------------------------------
    def reason(self, prompt: str, system: str = "") -> Optional[str]:
        return self._generate(prompt, system)

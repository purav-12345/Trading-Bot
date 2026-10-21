"""SCENARIO 1 — autonomous momentum: Opening-Range Breakout + Supertrend
trend filter, with an optional Gemini news-sentiment veto.

This is a transparent ("white box") rules strategy. The AI only acts as a
VETO on longs when news is clearly negative; it never initiates a trade.
Long-only intraday for safety; the risk engine enforces stops/limits.
"""
from __future__ import annotations

from datetime import datetime

from . import indicators as ind
from .base import Signal, Strategy, StrategyContext


class SupertrendORBStrategy(Strategy):
    type = "supertrend_orb"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.atr_period = int(cfg.get("atr_period", 10))
        self.atr_mult = float(cfg.get("atr_multiplier", 3.0))
        self.or_minutes = int(cfg.get("opening_range_minutes", 15))
        self.use_ai_bias = bool(cfg.get("use_ai_bias", False))
        self.target_pct = float(cfg.get("target_pct", 0.01))     # +1% target
        self.trail_pct = float(cfg.get("trail_pct", 0.005))      # 0.5% trailing stop
        # opening range state
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._or_locked = False
        self._peak: float | None = None

    def _update_opening_range(self, ctx: StrategyContext, ltp: float) -> None:
        if self._or_locked or ctx.session_start is None:
            return
        # Use the context clock (real time live; candle time in backtests).
        now_t = ctx.now.time() if getattr(ctx, "now", None) else datetime.now().time()
        elapsed = (now_t.hour * 60 + now_t.minute) - (
            ctx.session_start.hour * 60 + ctx.session_start.minute
        )
        if elapsed < 0:
            return
        if elapsed <= self.or_minutes:
            self._or_high = ltp if self._or_high is None else max(self._or_high, ltp)
            self._or_low = ltp if self._or_low is None else min(self._or_low, ltp)
        else:
            self._or_locked = True

    def on_tick(self, ctx: StrategyContext) -> list[Signal]:
        ltp = ctx.ltp(self.symbol)
        ctx.record_tick(self.symbol, ltp)
        prices = ctx.prices(self.symbol)
        pos = ctx.position(self.symbol)
        self._update_opening_range(ctx, ltp)

        # ---- manage an open long: trailing stop + target ----
        if pos.quantity > 0:
            self._peak = ltp if self._peak is None else max(self._peak, ltp)
            target = pos.avg_price * (1 + self.target_pct)
            trail = self._peak * (1 - self.trail_pct)
            if ltp >= target:
                return [Signal(self.symbol, "SELL", pos.quantity, "target hit", is_exit=True)]
            if ltp <= trail:
                self._peak = None
                return [Signal(self.symbol, "SELL", pos.quantity, "trailing stop", is_exit=True)]
            return []

        # ---- look for a breakout entry ----
        if not self._or_locked or self._or_high is None:
            return []  # wait for opening range to form

        trend, _, _ = ind.supertrend_lite(prices, self.atr_period, self.atr_mult)
        breakout = ltp > self._or_high
        if breakout and trend >= 0:
            if self.use_ai_bias and ctx.ai is not None:
                bias = ctx.ai.sentiment_bias(self.symbol)  # +1 / 0 / -1
                if bias < 0:
                    return []  # AI veto: negative news, skip the long
            self._peak = ltp
            return [Signal(self.symbol, "BUY", self.quantity,
                           reason=f"ORB breakout > {self._or_high:.2f}, trend up")]
        return []

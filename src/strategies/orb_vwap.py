"""Strategy #1 — Opening Range Breakout + volume + VWAP filter (long-only).

Rules (per session):
  * Build the opening range (high/low) over the first `opening_range_minutes`.
  * After that, go LONG only when ALL are true:
      - close breaks above the opening-range high
      - close is above session VWAP (trend confirmation)
      - breakout bar volume > volume_mult x average session volume
  * Stop = opening-range low. Target = entry + target_r x (entry - stop).
  * Flat again after stop/target -> can re-enter on a fresh breakout.

Realistic expectation: ~45-55% win rate; profit (if any) comes from winners
running further than losers, NOT from a high hit-rate. Backtest with costs.
"""
from __future__ import annotations

import pandas as pd

from . import indicators as ind
from .base import Signal, Strategy, StrategyContext


class ORBVwapStrategy(Strategy):
    type = "orb_vwap"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.or_minutes = int(cfg.get("opening_range_minutes", 15))
        self.volume_mult = float(cfg.get("volume_mult", 1.5))
        self.target_r = float(cfg.get("target_r", 2.0))
        self.min_bars = int(cfg.get("min_bars", 20))
        self._reset_session(None)

    def _reset_session(self, day):
        self._day = day
        self._or_high = None
        self._or_low = None
        self._or_locked = False
        self._session_start_min = None
        self._stop = None
        self._target = None

    def on_tick(self, ctx: StrategyContext) -> list[Signal]:
        df = ctx.bars(self.symbol)
        if len(df) < self.min_bars or "date" not in df.columns:
            return []
        df = df.copy()
        df["ts"] = pd.to_datetime(df["date"])
        last = df.iloc[-1]
        day = last["ts"].date()
        if day != self._day:
            self._reset_session(day)

        session = df[df["ts"].dt.date == day]
        if session.empty:
            return []
        start_min = session["ts"].iloc[0].hour * 60 + session["ts"].iloc[0].minute
        cur_min = last["ts"].hour * 60 + last["ts"].minute
        elapsed = cur_min - start_min

        # Build / lock the opening range.
        in_range = session[session["ts"].apply(
            lambda t: (t.hour * 60 + t.minute) - start_min <= self.or_minutes)]
        if not in_range.empty:
            self._or_high = float(in_range["high"].max())
            self._or_low = float(in_range["low"].min())
        if elapsed > self.or_minutes:
            self._or_locked = True

        price = float(last["close"])
        pos = ctx.position(self.symbol)

        # Manage an open long.
        if pos.quantity > 0:
            if self._stop and price <= self._stop:
                return [Signal(self.symbol, "SELL", pos.quantity, "ORB stop", is_exit=True)]
            if self._target and price >= self._target:
                return [Signal(self.symbol, "SELL", pos.quantity, "ORB target", is_exit=True)]
            return []

        if not self._or_locked or self._or_high is None:
            return []

        vwap = ind.vwap(session).iloc[-1]
        avg_vol = session["volume"].mean()
        last_vol = float(last["volume"])
        vol_ok = (avg_vol <= 0) or (last_vol > self.volume_mult * avg_vol)

        if price > self._or_high and price > vwap and vol_ok:
            self._stop = self._or_low
            risk = max(price - self._or_low, 1e-6)
            self._target = price + self.target_r * risk
            return [Signal(self.symbol, "BUY", self.quantity,
                           reason=f"ORB breakout>{self._or_high:.2f}, >VWAP{vwap:.2f}, vol ok")]
        return []

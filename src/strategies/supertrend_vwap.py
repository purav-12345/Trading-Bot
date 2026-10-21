"""Strategy #2 — Supertrend + VWAP trend filter (long-only intraday).

Rules:
  * Enter LONG when Supertrend flips to up (prev bar down -> this bar up)
    AND close is above session VWAP.
  * Exit when Supertrend flips to down OR close falls below VWAP.
  * Optional ATR stop below entry as a hard safety.

Works on trending days, chops in sideways markets — which is exactly why you
backtest it before trusting it. Transparent, no black box.
"""
from __future__ import annotations

from . import indicators as ind
from .base import Signal, Strategy, StrategyContext


class SupertrendVwapStrategy(Strategy):
    type = "supertrend_vwap"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.atr_period = int(cfg.get("atr_period", 10))
        self.atr_mult = float(cfg.get("atr_multiplier", 3.0))
        self.stop_atr_mult = float(cfg.get("stop_atr_mult", 2.0))
        self.min_bars = int(cfg.get("min_bars", 30))
        self._stop = None

    def on_tick(self, ctx: StrategyContext) -> list[Signal]:
        df = ctx.bars(self.symbol)
        if len(df) < max(self.min_bars, self.atr_period + 2):
            return []

        trend = ind.supertrend(df, self.atr_period, self.atr_mult)
        vwap = ind.vwap(df)
        price = float(df["close"].iloc[-1])
        cur_trend = int(trend.iloc[-1])
        prev_trend = int(trend.iloc[-2])
        cur_vwap = float(vwap.iloc[-1])
        pos = ctx.position(self.symbol)

        # Manage open long.
        if pos.quantity > 0:
            if self._stop and price <= self._stop:
                return [Signal(self.symbol, "SELL", pos.quantity, "ATR stop", is_exit=True)]
            if cur_trend < 0 or price < cur_vwap:
                self._stop = None
                return [Signal(self.symbol, "SELL", pos.quantity,
                               "supertrend down / below VWAP", is_exit=True)]
            return []

        # Entry: fresh up-flip confirmed by VWAP.
        flipped_up = prev_trend < 0 <= cur_trend or (prev_trend <= 0 and cur_trend > 0)
        if flipped_up and price > cur_vwap:
            a = float(ind.atr(df, self.atr_period).iloc[-1])
            self._stop = price - self.stop_atr_mult * a
            return [Signal(self.symbol, "BUY", self.quantity,
                           reason=f"supertrend up + close>{cur_vwap:.2f} VWAP")]
        return []

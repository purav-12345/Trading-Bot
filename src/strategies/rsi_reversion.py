"""Strategy #3 — RSI mean-reversion (long-only, buy-the-dip in an uptrend).

Rules:
  * Trend filter: only buy when close is above a longer SMA (uptrend).
  * Entry: RSI drops below `oversold` -> buy the dip.
  * Exit: RSI recovers above `exit_level`, OR close falls below the SMA
    (trend broke -> get out).

This is the range-day counterpart to the trend strategies: it tends to make
money when the market is choppy/mean-reverting (~65-70% of sessions) and lose
when a strong trend runs against the reversion. Backtest before trusting.
"""
from __future__ import annotations

from . import indicators as ind
from .base import Signal, Strategy, StrategyContext


class RSIReversionStrategy(Strategy):
    type = "rsi_reversion"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.rsi_period = int(cfg.get("rsi_period", 2))
        self.oversold = float(cfg.get("oversold", 10.0))
        self.exit_level = float(cfg.get("exit_level", 60.0))
        self.trend_sma = int(cfg.get("trend_sma", 200))
        self.min_bars = int(cfg.get("min_bars", 50))

    def on_tick(self, ctx: StrategyContext) -> list[Signal]:
        df = ctx.bars(self.symbol)
        need = max(self.min_bars, self.rsi_period + 2, min(self.trend_sma, self.min_bars))
        if len(df) < need:
            return []

        close = df["close"]
        rsi = ind.rsi(close, self.rsi_period).iloc[-1]
        sma_period = min(self.trend_sma, len(df))
        sma = float(close.rolling(sma_period).mean().iloc[-1])
        price = float(close.iloc[-1])
        pos = ctx.position(self.symbol)

        if pos.quantity > 0:
            if rsi > self.exit_level or price < sma:
                return [Signal(self.symbol, "SELL", pos.quantity,
                               f"RSI {rsi:.0f}>exit or trend broke", is_exit=True)]
            return []

        if price > sma and rsi < self.oversold:
            return [Signal(self.symbol, "BUY", self.quantity,
                           reason=f"dip buy: RSI {rsi:.0f}<{self.oversold} in uptrend")]
        return []

"""Classic moving-average crossover (trend following), long-only intraday.

Enter long on fast-SMA crossing ABOVE slow-SMA; exit on the reverse cross.
A simple, transparent baseline strategy you can backtest and reason about.
"""
from __future__ import annotations

from . import indicators as ind
from .base import Signal, Strategy, StrategyContext


class MACrossoverStrategy(Strategy):
    type = "ma_crossover"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.fast = int(cfg.get("fast_period", 9))
        self.slow = int(cfg.get("slow_period", 21))
        self._prev_diff: float | None = None

    def on_tick(self, ctx: StrategyContext) -> list[Signal]:
        ltp = ctx.ltp(self.symbol)
        ctx.record_tick(self.symbol, ltp)
        prices = ctx.prices(self.symbol)
        pos = ctx.position(self.symbol)

        fast = ind.sma(prices, self.fast)
        slow = ind.sma(prices, self.slow)
        if fast is None or slow is None:
            return []

        diff = fast - slow
        crossed_up = self._prev_diff is not None and self._prev_diff <= 0 < diff
        crossed_down = self._prev_diff is not None and self._prev_diff >= 0 > diff
        self._prev_diff = diff

        if crossed_up and pos.quantity == 0:
            return [Signal(self.symbol, "BUY", self.quantity, reason="fast crossed above slow")]
        if crossed_down and pos.quantity > 0:
            return [Signal(self.symbol, "SELL", pos.quantity, reason="fast crossed below slow", is_exit=True)]
        return []

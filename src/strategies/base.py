"""Strategy interface + the per-tick context handed to strategies."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional

import pandas as pd

from ..brokers.base import Position


@dataclass
class Signal:
    symbol: str
    side: str                       # BUY / SELL
    quantity: int
    reason: str = ""
    order_type: str = "MARKET"
    price: Optional[float] = None
    is_exit: bool = False           # exits bypass the per-order risk gate


class StrategyContext:
    """Read-only view the engine gives strategies each tick."""

    def __init__(self, broker, session_start: Optional[time] = None, ai=None, maxlen: int = 5000):
        self.broker = broker
        self.ai = ai
        self.session_start = session_start
        self.now = datetime.now()
        self.backtest = False  # set True by the backtester (disables look-ahead sources)
        self._prices: dict[str, deque] = {}
        self._bars: dict[str, deque] = {}
        self._maxlen = maxlen

    def record_tick(self, symbol: str, price: float) -> None:
        buf = self._prices.setdefault(symbol, deque(maxlen=self._maxlen))
        buf.append(price)

    def prices(self, symbol: str) -> list[float]:
        return list(self._prices.get(symbol, []))

    # ---- OHLCV bars (for VWAP/RSI/ATR/Supertrend strategies) ----
    def record_bar(self, symbol: str, bar: dict) -> None:
        buf = self._bars.setdefault(symbol, deque(maxlen=self._maxlen))
        buf.append({
            "date": bar.get("date"),
            "open": float(bar.get("open", bar.get("close", 0.0))),
            "high": float(bar.get("high", bar.get("close", 0.0))),
            "low": float(bar.get("low", bar.get("close", 0.0))),
            "close": float(bar.get("close", 0.0)),
            "volume": float(bar.get("volume", 0.0) or 0.0),
        })

    def bars(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(list(self._bars.get(symbol, [])))

    def n_bars(self, symbol: str) -> int:
        return len(self._bars.get(symbol, []))

    def ltp(self, symbol: str) -> float:
        return self.broker.ltp(symbol)

    def position(self, symbol: str) -> Position:
        return self.broker.positions().get(symbol, Position(symbol=symbol))


class Strategy:
    type: str = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg.get("name", self.type)
        self.symbol = cfg["symbol"]
        self.quantity = int(cfg.get("quantity", 1))

    def on_tick(self, ctx: StrategyContext) -> list[Signal]:  # pragma: no cover - interface
        raise NotImplementedError

"""Event-driven backtest: replays candles through the SAME strategy + paper
broker + risk-engine code used live, so behaviour matches production.

Data source priority (set in MarketData.historical): Kite (paid) -> yfinance
(free, real NSE history) -> synthetic. Costs + slippage are modelled so the
net P&L reflects what you'd actually keep, not an optimistic raw-price figure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dtime

import pandas as pd

from ..brokers.paper import PaperBroker
from ..config import Config
from ..market import MarketData
from ..strategies import StrategyContext, build_strategy
from .costs import CostModel


class _ReplayMD:
    """Market-data stub: serves a single 'current' price set by the loop."""

    def __init__(self):
        self._price: dict[str, float] = {}

    def connect(self):  # noqa: D401
        pass

    @property
    def using_live_data(self):
        return False

    def set(self, symbol: str, price: float):
        self._price[symbol] = price

    def ltp(self, symbol: str) -> float:
        return self._price.get(symbol, 0.0)


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    source: str
    bars: int
    trades: int
    gross_pnl: float
    total_costs: float
    net_pnl: float
    max_drawdown: float
    equity_curve: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (f"[{self.strategy}/{self.symbol}] data={self.source} bars={self.bars} "
                f"trades={self.trades} | gross=Rs.{self.gross_pnl:,.0f} "
                f"costs=Rs.{self.total_costs:,.0f} -> NET=Rs.{self.net_pnl:,.0f} "
                f"(netDD=Rs.{self.max_drawdown:,.0f})")


def backtest(config: Config, strategy_cfg: dict, days: int = 5) -> BacktestResult:
    symbol = strategy_cfg["symbol"]
    md_live = MarketData(config)
    md_live.connect()
    candles: pd.DataFrame = md_live.historical(symbol, interval="minute", days=days)
    source = getattr(md_live, "last_source", "synthetic")

    replay = _ReplayMD()
    broker = PaperBroker(config, replay)
    strat = build_strategy(strategy_cfg)
    ctx = StrategyContext(broker, session_start=dtime(9, 15))
    ctx.backtest = True  # disables look-ahead sources (fundamentals/news/VIX)
    costs = CostModel.from_config(config)

    equity: list[float] = []
    peak = 0.0
    max_dd = 0.0
    trades = 0
    total_costs = 0.0

    for _, row in candles.iterrows():
        price = float(row["close"])
        if "date" in row and pd.notna(row["date"]):
            ts = pd.to_datetime(row["date"])
            ctx.now = ts.to_pydatetime()
        # Feed the full OHLCV bar so VWAP/RSI/ATR/Supertrend strategies work.
        ctx.record_bar(symbol, {
            "date": row.get("date"), "open": row.get("open", price),
            "high": row.get("high", price), "low": row.get("low", price),
            "close": price, "volume": row.get("volume", 0.0),
        })
        replay.set(symbol, price)
        for sig in strat.on_tick(ctx):
            # Apply adverse slippage to the fill, then book costs on execution.
            fill = costs.fill_with_slippage(price, sig.side)
            replay.set(symbol, fill)
            res = broker.place_order(sig.symbol, sig.side, sig.quantity, "MARKET")
            replay.set(symbol, price)
            if res.ok and res.status == "FILLED":
                trades += 1
                total_costs += costs.charges(res.fill_price, sig.quantity, sig.side)
        net = broker.total_pnl() - total_costs
        equity.append(round(net, 2))
        peak = max(peak, net)
        max_dd = min(max_dd, net - peak)

    gross = broker.realized_pnl() + broker.unrealized_pnl()
    return BacktestResult(
        strategy=strat.name, symbol=symbol, source=source, bars=len(candles), trades=trades,
        gross_pnl=round(gross, 2), total_costs=round(total_costs, 2),
        net_pnl=round(gross - total_costs, 2), max_drawdown=round(max_dd, 2),
        equity_curve=equity,
    )

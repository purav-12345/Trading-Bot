"""Multi-stock portfolio backtest for HAASEY, with real metrics.

Feeds real historical bars for the ENTIRE universe simultaneously and lets the
bot pick which stocks to trade on its own, from one capital pool. Reports
Net P&L, Max Drawdown, Sharpe, win-rate and per-stock trade counts.

Speed: for HAASEY it pre-resamples each symbol to the strategy's decision
timeframe ONCE and drives the loop at that cadence (no per-tick resampling),
so 30-90 day runs are feasible. Backtest uses technical+momentum only
(fundamentals/news/VIX would be look-ahead bias) — so it also costs ZERO API.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..data import default_universe
from ..market import MarketData
from ..strategies import build_strategy
from ..strategies.base import StrategyContext
from ..strategies.haasey import _resample
from .costs import CostModel
from .runner import _ReplayMD


@dataclass
class PortfolioResult:
    symbols_considered: int
    decision_bars: int
    trades: int
    wins: int
    closed: int              # number of round-trips (for win-rate)
    gross_pnl: float
    total_costs: float
    net_pnl: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe: float
    per_symbol_trades: dict = field(default_factory=dict)
    final_positions: dict = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return round(100 * self.wins / self.closed, 1) if self.closed else 0.0

    def summary(self) -> str:
        picks = ", ".join(f"{s}:{n}" for s, n in
                          sorted(self.per_symbol_trades.items(), key=lambda x: -x[1])) or "none"
        return (
            f"[HAASEY portfolio] universe={self.symbols_considered} | decision_bars={self.decision_bars}\n"
            f"  trades={self.trades}  win-rate={self.win_rate}%  "
            f"gross=Rs.{self.gross_pnl:,.0f}  costs=Rs.{self.total_costs:,.0f}  "
            f"NET=Rs.{self.net_pnl:,.0f}\n"
            f"  max drawdown=Rs.{self.max_drawdown:,.0f} ({self.max_drawdown_pct:.1f}%)  "
            f"Sharpe(annualised)={self.sharpe:.2f}\n"
            f"  bot traded: {picks}\n"
            f"  final holdings: {self.final_positions}"
        )


def _sharpe(equity_by_day: list[float], capital: float) -> float:
    if len(equity_by_day) < 3:
        return 0.0
    eq = np.array(equity_by_day, dtype=float)
    daily_pnl = np.diff(eq)
    rets = daily_pnl / capital
    sd = rets.std(ddof=1)
    if sd == 0:
        return 0.0
    return float((rets.mean() / sd) * np.sqrt(252))


def fetch_decision_data(config: Config, haasey_cfg: dict, days: int = 30,
                        max_symbols: int | None = None):
    """Fetch + pre-resample the whole universe (and regime index) ONCE.

    Returns (data, tradeable, decision_minutes). Reused by portfolio_backtest and
    the walk-forward harness so we don't refetch (and rate-limit) per fold.
    """
    md = MarketData(config)
    md.connect()
    cfg = dict(haasey_cfg)
    uni = cfg.get("universe")
    if not uni or uni == "auto":
        uni = default_universe()
    if max_symbols:
        uni = uni[:max_symbols]

    decision_minutes = int(cfg.get("decision_bar_minutes", 15))
    fetch_list = list(uni)
    regime_symbol = cfg.get("regime_symbol", "NIFTY 50")
    if cfg.get("use_regime", True):
        fetch_list.append(regime_symbol)

    data: dict[str, pd.DataFrame] = {}
    for sym in fetch_list:
        try:
            raw = md.historical(sym, interval="minute", days=days)
        except Exception:
            raw = None
        if raw is None or raw.empty or "date" not in raw.columns:
            continue
        dec = _resample(raw, decision_minutes)
        if not dec.empty:
            dec["ts"] = pd.to_datetime(dec["date"])
            data[sym] = dec.set_index("ts")
    tradeable = [s for s in uni if s in data]
    if not tradeable:
        raise RuntimeError("No historical data fetched for the universe.")
    return data, tradeable, decision_minutes


def run_portfolio_over(config: Config, haasey_cfg: dict, data: dict, tradeable: list,
                       timeline=None, equity_csv: str | None = None,
                       live_mode: bool = False) -> PortfolioResult:
    """Run HAASEY over pre-fetched decision-bar `data`, optionally restricted to a
    `timeline` subset (used by walk-forward to slice train/test windows)."""
    cfg = dict(haasey_cfg)
    capital = float(cfg.get("capital", 20000))
    replay = _ReplayMD()
    from ..brokers.paper import PaperBroker
    broker = PaperBroker(config, replay)
    broker.cash = capital

    cfg["universe"] = tradeable
    strat = build_strategy(cfg)
    strat.set_prebucketed(True)  # bars fed are already completed decision bars
    ctx = StrategyContext(broker)
    ctx.backtest = not live_mode
    costs = CostModel.from_config(config)
    equity_rows: list[tuple] = []

    if timeline is None:
        timeline = sorted(set().union(*[df.index for df in data.values()]))
    per_symbol = Counter()
    total_costs = 0.0
    trades = wins = closed = 0
    equity_curve = []           # net equity at each decision bar
    day_end_equity: dict = {}   # date -> last net equity that day

    for t in timeline:
        for sym, df in data.items():
            if t in df.index:
                r = df.loc[t]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[-1]
                ctx.record_bar(sym, {"date": t, "open": r["open"], "high": r["high"],
                                     "low": r["low"], "close": r["close"],
                                     "volume": r.get("volume", 0.0)})
                replay.set(sym, float(r["close"]))
        ctx.now = t.to_pydatetime()
        for sig in strat.on_tick(ctx):
            price = replay.ltp(sig.symbol)
            fill = costs.fill_with_slippage(price, sig.side)
            replay.set(sig.symbol, fill)
            realized_before = broker.realized_pnl()
            res = broker.place_order(sig.symbol, sig.side, sig.quantity, "MARKET")
            replay.set(sig.symbol, price)
            if res.ok and res.status == "FILLED":
                trades += 1
                per_symbol[sig.symbol] += 1
                total_costs += costs.charges(res.fill_price, sig.quantity, sig.side)
                if sig.is_exit:  # a closed round-trip: did it win?
                    closed += 1
                    if broker.realized_pnl() > realized_before:
                        wins += 1
        net = broker.total_pnl() - total_costs
        equity_curve.append(net)
        day_end_equity[t.date()] = net
        if equity_csv:
            open_pos = sum(1 for p in broker.positions().values() if p.quantity)
            equity_rows.append((t, round(net, 2), trades, open_pos))

    if equity_csv:
        import csv as _csv
        with open(equity_csv, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["ts", "realized_pnl", "unrealized_pnl", "total_pnl",
                        "open_positions", "trade_count", "locked", "lock_reason"])
            for t, net, tr, op in equity_rows:
                w.writerow([t.isoformat(), "", "", net, op, tr, 0, ""])

    # metrics
    peak = 0.0
    max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        max_dd = min(max_dd, e - peak)
    sharpe = _sharpe([capital + v for v in day_end_equity.values()], capital)
    gross = broker.realized_pnl() + broker.unrealized_pnl()
    final_positions = {s: p.quantity for s, p in broker.positions().items() if p.quantity}

    return PortfolioResult(
        symbols_considered=len(tradeable),
        decision_bars=len(timeline),
        trades=trades, wins=wins, closed=closed,
        gross_pnl=round(gross, 2), total_costs=round(total_costs, 2),
        net_pnl=round(gross - total_costs, 2),
        max_drawdown=round(max_dd, 2),
        max_drawdown_pct=round(100 * max_dd / capital, 2),
        sharpe=round(sharpe, 2),
        per_symbol_trades=dict(per_symbol), final_positions=final_positions,
    )


def portfolio_backtest(config: Config, haasey_cfg: dict, days: int = 30,
                       max_symbols: int | None = None,
                       equity_csv: str | None = None,
                       live_mode: bool = False) -> PortfolioResult:
    """Fetch the universe once and run HAASEY over the full window."""
    data, tradeable, _ = fetch_decision_data(config, haasey_cfg, days, max_symbols)
    return run_portfolio_over(config, haasey_cfg, data, tradeable,
                              equity_csv=equity_csv, live_mode=live_mode)

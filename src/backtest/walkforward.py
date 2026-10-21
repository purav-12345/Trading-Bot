"""Walk-forward validation — the guardrail that makes every optimization honest.

Splits real history into rolling folds: optimize params on a TRAIN window, then
evaluate those params on the NEXT, untouched TEST window. Reports in-sample (IS)
vs out-of-sample (OOS) side by side. A strategy is only trustworthy if its
**OOS** numbers hold up — a great IS number that collapses OOS is overfitting.

Fetches the universe once (via fetch_decision_data) and slices the decision-bar
timeline per fold. The test window includes a short warm-up lead-in so indicators
are primed; those warm-up bars produce no trades (insufficient history), so the
OOS P&L reflects test-window trades only.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..config import Config
from .portfolio import fetch_decision_data, run_portfolio_over

# Small grid so folds run fast; the point is OOS honesty, not exhaustive search.
DEFAULT_GRID = {
    "buy": [0.45, 0.55, 0.65],
    "cooldown_minutes": [60],
    "trail_atr_mult": [2.0, 2.5],
}


@dataclass
class Fold:
    idx: int
    train_bars: int
    test_bars: int
    best_params: dict
    is_net: float
    oos_net: float
    oos_sharpe: float
    oos_trades: int


def walk_forward(config: Config, haasey_cfg: dict, days: int = 60,
                 max_symbols: int | None = 12, train_bars: int = 250,
                 test_bars: int = 125, grid: dict | None = None) -> list[Fold]:
    grid = grid or DEFAULT_GRID
    data, tradeable, _ = fetch_decision_data(config, haasey_cfg, days, max_symbols)
    timeline = sorted(set().union(*[df.index for df in data.values()]))
    warmup = max(int(haasey_cfg.get("trend_sma", 50)), 60) + 2

    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    folds: list[Fold] = []
    start = 0
    while start + train_bars + test_bars <= len(timeline):
        train_tl = timeline[start:start + train_bars]
        test_start = start + train_bars
        test_tl = timeline[max(0, test_start - warmup):test_start + test_bars]

        # 1) optimize on TRAIN
        best_params, best_net = None, None
        for combo in combos:
            p = dict(zip(keys, combo))
            c = dict(haasey_cfg); c.update(p)
            r = run_portfolio_over(config, c, data, tradeable, timeline=train_tl)
            if best_net is None or r.net_pnl > best_net:
                best_params, best_net = p, r.net_pnl

        # 2) evaluate the winner on the untouched TEST window
        c = dict(haasey_cfg); c.update(best_params)
        oos = run_portfolio_over(config, c, data, tradeable, timeline=test_tl)
        folds.append(Fold(len(folds) + 1, train_bars, test_bars, best_params,
                          round(best_net, 2), oos.net_pnl, oos.sharpe, oos.trades))
        start += test_bars
    return folds


def format_folds(folds: list[Fold]) -> str:
    if not folds:
        return ("Not enough data for a walk-forward split. Try more --days or "
                "smaller --train-bars/--test-bars (free yfinance intraday ~60 days).")
    lines = [
        "",
        "===== WALK-FORWARD: in-sample (IS) vs OUT-OF-SAMPLE (OOS) =====",
        f"{'fold':>4}  {'best buy/cd/trail':>18}  {'IS net':>9}  {'OOS net':>9}  "
        f"{'OOS Sharpe':>10}  {'OOS trades':>10}",
        "-" * 74,
    ]
    tot = 0.0
    pos = 0
    for f in folds:
        p = f.best_params
        combo = f"{p['buy']:.2f}/{p['cooldown_minutes']}/{p['trail_atr_mult']}"
        lines.append(f"{f.idx:>4}  {combo:>18}  {f.is_net:>9,.0f}  {f.oos_net:>9,.0f}  "
                     f"{f.oos_sharpe:>10.2f}  {f.oos_trades:>10}")
        tot += f.oos_net
        pos += 1 if f.oos_net > 0 else 0
    n = len(folds)
    lines += [
        "-" * 74,
        f"  OOS folds positive: {pos}/{n}   total OOS net: Rs.{tot:,.0f}   "
        f"avg OOS/fold: Rs.{tot / n:,.0f}",
        "  VERDICT: trust the strategy ONLY if OOS is consistently positive across",
        "  folds. A high IS with weak/negative OOS = overfitting (do NOT go live).",
    ]
    return "\n".join(lines)

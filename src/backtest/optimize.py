"""Grid-search optimiser for HAASEY's efficiency parameters.

Sweeps ONLY the three "efficiency" knobs the design calls out —
buy threshold, cooldown_minutes, trail_atr_mult — NOT the strategy logic.
Reports each combo's trade count, net P&L, drawdown and Sharpe so you can
pick a robust setting instead of overfitting.

Rule of thumb (from the design): prefer the combo with the best net P&L that
keeps the trade count in a sane band (not too few, not churny) over the window.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..config import Config
from .portfolio import portfolio_backtest

DEFAULT_GRID = {
    "buy": [0.50, 0.55, 0.60, 0.65],
    "cooldown_minutes": [30, 60, 90, 120],
    "trail_atr_mult": [1.5, 2.0, 2.5, 3.0],
}


@dataclass
class GridRow:
    params: dict
    trades: int
    net_pnl: float
    max_drawdown: float
    sharpe: float
    win_rate: float


def grid_search(config: Config, base_cfg: dict, days: int = 30,
                max_symbols: int | None = None, grid: dict | None = None) -> list[GridRow]:
    grid = grid or DEFAULT_GRID
    keys = list(grid.keys())
    rows: list[GridRow] = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        cfg = dict(base_cfg)
        cfg.update(params)
        try:
            res = portfolio_backtest(config, cfg, days=days, max_symbols=max_symbols)
            rows.append(GridRow(params, res.trades, res.net_pnl,
                                res.max_drawdown, res.sharpe, res.win_rate))
        except Exception as exc:  # noqa: BLE001
            print(f"[optimize] {params} failed: {exc}")
    rows.sort(key=lambda r: r.net_pnl, reverse=True)
    return rows


def format_rows(rows: list[GridRow], top: int = 15) -> str:
    out = ["", f"{'buy':>5} {'cooldn':>6} {'trail':>5} | {'trades':>6} "
              f"{'netP&L':>9} {'maxDD':>8} {'Sharpe':>6} {'win%':>5}",
           "-" * 62]
    for r in rows[:top]:
        out.append(
            f"{r.params['buy']:>5.2f} {r.params['cooldown_minutes']:>6d} "
            f"{r.params['trail_atr_mult']:>5.1f} | {r.trades:>6d} "
            f"{r.net_pnl:>9,.0f} {r.max_drawdown:>8,.0f} {r.sharpe:>6.2f} {r.win_rate:>5.1f}"
        )
    return "\n".join(out)

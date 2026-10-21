"""Genetic multi-objective optimizer for HAASEY's efficiency parameters.

Evolves {buy, cooldown_minutes, trail_atr_mult, target_atr_mult} to balance
three competing objectives — maximize net P&L, minimize drawdown, control trade
count — via a scalarized fitness. Crucially it is **walk-forward gated**: the GA
optimizes on a TRAIN split, then the winner is reported on a separate untouched
TEST split. If TEST << TRAIN, it overfit (expected, given our OOS findings) and
must not be trusted.

This replaces hand-tuned params with searched ones — but honestly, since the GA
can only fit the efficiency knobs (fundamentals are frozen in backtest), it
inherits the same "no technical edge" conclusion. It exists to prove/quantify
that, not to conjure returns.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import Config
from .portfolio import fetch_decision_data, run_portfolio_over

# Gene bounds
BOUNDS = {
    "buy": (0.40, 0.75),
    "cooldown_minutes": (30, 150),
    "trail_atr_mult": (1.5, 3.5),
    "target_atr_mult": (2.0, 4.0),
}


@dataclass
class GAResult:
    best_params: dict
    is_net: float
    is_dd: float
    oos_net: float
    oos_dd: float
    oos_sharpe: float
    oos_trades: int
    generations: int

    def summary(self) -> str:
        p = self.best_params
        return (
            "\n===== GENETIC OPTIMIZER (walk-forward gated) =====\n"
            f"  best params: buy={p['buy']:.2f} cooldown={p['cooldown_minutes']} "
            f"trail={p['trail_atr_mult']:.2f} target={p['target_atr_mult']:.2f}\n"
            f"  IN-SAMPLE (train): net=Rs.{self.is_net:,.0f}  maxDD=Rs.{self.is_dd:,.0f}\n"
            f"  OUT-OF-SAMPLE (test): net=Rs.{self.oos_net:,.0f}  maxDD=Rs.{self.oos_dd:,.0f}  "
            f"Sharpe={self.oos_sharpe:.2f}  trades={self.oos_trades}\n"
            "  VERDICT: trust ONLY the OOS number. If OOS << IS, the GA overfit."
        )


def _random_genome() -> dict:
    return {
        "buy": round(random.uniform(*BOUNDS["buy"]), 2),
        "cooldown_minutes": random.randrange(BOUNDS["cooldown_minutes"][0],
                                             BOUNDS["cooldown_minutes"][1] + 1, 15),
        "trail_atr_mult": round(random.uniform(*BOUNDS["trail_atr_mult"]), 2),
        "target_atr_mult": round(random.uniform(*BOUNDS["target_atr_mult"]), 2),
    }


def _crossover(a: dict, b: dict) -> dict:
    return {k: random.choice([a[k], b[k]]) for k in a}


def _mutate(g: dict, rate: float = 0.3) -> dict:
    g = dict(g)
    for k, (lo, hi) in BOUNDS.items():
        if random.random() < rate:
            if k == "cooldown_minutes":
                g[k] = random.randrange(lo, hi + 1, 15)
            else:
                g[k] = round(min(hi, max(lo, g[k] + random.uniform(-0.1, 0.1) * (hi - lo))), 2)
    return g


def _fitness(config, base_cfg, data, tradeable, timeline, genome) -> tuple[float, object]:
    c = dict(base_cfg); c.update(genome)
    r = run_portfolio_over(config, c, data, tradeable, timeline=timeline)
    # scalarized multi-objective: reward net, penalize drawdown + over-trading
    fit = r.net_pnl + 0.5 * r.max_drawdown - 0.5 * max(0, r.trades - 120)
    return fit, r


def genetic_optimize(config: Config, base_cfg: dict, days: int = 45,
                     max_symbols: int | None = 10, pop_size: int = 12,
                     generations: int = 6, train_frac: float = 0.7,
                     seed: int = 42) -> GAResult:
    random.seed(seed)
    data, tradeable, _ = fetch_decision_data(config, base_cfg, days, max_symbols)
    timeline = sorted(set().union(*[df.index for df in data.values()]))
    split = int(len(timeline) * train_frac)
    warmup = max(int(base_cfg.get("trend_sma", 50)), 60) + 2
    train_tl = timeline[:split]
    test_tl = timeline[max(0, split - warmup):]

    pop = [_random_genome() for _ in range(pop_size)]
    best_genome, best_fit, best_r = None, None, None
    for _ in range(generations):
        scored = []
        for g in pop:
            fit, r = _fitness(config, base_cfg, data, tradeable, train_tl, g)
            scored.append((fit, g, r))
            if best_fit is None or fit > best_fit:
                best_fit, best_genome, best_r = fit, g, r
        scored.sort(key=lambda x: x[0], reverse=True)
        # elitism + tournament breeding
        survivors = [g for _, g, _ in scored[: max(2, pop_size // 2)]]
        children = []
        while len(children) < pop_size - len(survivors):
            a, b = random.sample(survivors, 2)
            children.append(_mutate(_crossover(a, b)))
        pop = survivors + children

    # Evaluate the winner OUT-OF-SAMPLE
    c = dict(base_cfg); c.update(best_genome)
    oos = run_portfolio_over(config, c, data, tradeable, timeline=test_tl)
    return GAResult(best_genome, best_r.net_pnl, best_r.max_drawdown,
                    oos.net_pnl, oos.max_drawdown, oos.sharpe, oos.trades, generations)

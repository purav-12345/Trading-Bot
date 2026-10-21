"""PPO reinforcement-learning agent — OPTIONAL, EXPERIMENTAL, baseline-gated.

Per symbol: train a PPO policy on the TRAIN split of the TradingEnv, roll it out
on the untouched TEST split, and sum the out-of-sample P&L. Compared to the
rule-based HAASEY on the same window. If PPO doesn't clearly beat the baseline
OOS, it is dropped — not shipped (per plan). Import-guarded (needs
stable-baselines3 + gymnasium). Kept intentionally small/fast; RL for a ₹20k
retail account is high-complexity, low-expected-payoff (acknowledged).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Config


@dataclass
class RLResult:
    ok: bool
    message: str = ""
    rl_oos_net: float = 0.0
    baseline_oos_net: float = 0.0
    symbols: int = 0

    @property
    def beats_baseline(self) -> bool:
        return self.ok and self.rl_oos_net > self.baseline_oos_net

    def summary(self) -> str:
        if not self.ok:
            return f"[PPO] {self.message}"
        verdict = ("beats baseline OOS -> keep experimenting (validate live)"
                   if self.beats_baseline else
                   "does NOT beat the rule baseline OOS -> DROP (as planned)")
        return (
            "\n===== PPO RL AGENT (out-of-sample, gated, experimental) =====\n"
            f"  symbols tested: {self.symbols}\n"
            f"  PPO OOS net     : Rs.{self.rl_oos_net:,.0f}\n"
            f"  rule baseline   : Rs.{self.baseline_oos_net:,.0f}\n"
            f"  VERDICT: {verdict}"
        )


def train_and_evaluate(config: Config, base_cfg: dict, days: int = 45,
                       max_symbols: int | None = 5, timesteps: int = 4000,
                       train_frac: float = 0.7) -> RLResult:
    try:
        from stable_baselines3 import PPO
    except Exception:
        return RLResult(False, "stable-baselines3 not installed.")
    from .rl_env import TradingEnv, _HAS_GYM
    if not _HAS_GYM:
        return RLResult(False, "gymnasium not installed.")
    from ..backtest.portfolio import fetch_decision_data, run_portfolio_over

    data, tradeable, _ = fetch_decision_data(config, base_cfg, days, max_symbols)
    timeline = sorted(set().union(*[df.index for df in data.values()]))
    split_ts = timeline[int(len(timeline) * train_frac)]
    capital = float(base_cfg.get("capital", 20000))
    notional = capital / max(1, len(tradeable))

    rl_net = 0.0
    used = 0
    for sym in tradeable:
        df = data[sym]
        train_df = df[df.index <= split_ts]
        test_df = df[df.index > split_ts]
        if len(train_df) < 150 or len(test_df) < 40:
            continue
        try:
            env = TradingEnv(train_df)
            model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64)
            model.learn(total_timesteps=timesteps)
            tenv = TradingEnv(test_df)
            obs, _ = tenv.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, _trunc, _ = tenv.step(action)
                total_reward += r
            rl_net += total_reward * notional   # fractional reward -> Rs on notional
            used += 1
        except Exception:
            continue

    if used == 0:
        return RLResult(False, "not enough data to train/evaluate.")

    test_tl = [t for t in timeline if t > split_ts]
    baseline = run_portfolio_over(config, base_cfg, data, tradeable, timeline=test_tl)
    return RLResult(ok=True, rl_oos_net=round(rl_net, 2),
                    baseline_oos_net=round(baseline.net_pnl, 2), symbols=used)

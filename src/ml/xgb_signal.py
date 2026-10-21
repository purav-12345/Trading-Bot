"""XGBoost signal model — trained + validated out-of-sample, gated by a baseline.

For each stock: build technical features, train XGBoost on the TRAIN split to
predict "next bar up > threshold", predict on the untouched TEST split, and
simulate a long/flat strategy on those predictions (with costs). Aggregate OOS
net is compared to the rule-based HAASEY on the SAME test window.

HONEST GATE: if XGBoost does not beat the rule baseline OOS, it is scrapped —
not shipped. SHAP shows which features drive predictions (random-looking top
features = no real signal). Import-guarded (needs xgboost/shap).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from .features import FEATURE_COLS, make_dataset, build_features


@dataclass
class MLResult:
    ok: bool
    message: str = ""
    xgb_oos_net: float = 0.0
    baseline_oos_net: float = 0.0
    symbols: int = 0
    top_features: list = field(default_factory=list)
    beats_baseline: bool = False

    def summary(self) -> str:
        if not self.ok:
            return f"[XGBoost] {self.message}"
        feats = ", ".join(f"{n}={v:.3f}" for n, v in self.top_features[:6])
        verdict = ("BEATS baseline OOS -> candidate to ship (still validate live)"
                   if self.beats_baseline else
                   "does NOT beat the rule baseline OOS -> SCRAP (no real edge)")
        return (
            "\n===== XGBoost SIGNAL MODEL (out-of-sample, gated) =====\n"
            f"  symbols tested: {self.symbols}\n"
            f"  XGBoost OOS net : Rs.{self.xgb_oos_net:,.0f}\n"
            f"  rule baseline   : Rs.{self.baseline_oos_net:,.0f}\n"
            f"  top features (mean|SHAP|): {feats}\n"
            f"  VERDICT: {verdict}"
        )


def _simulate_long_flat(prices: np.ndarray, proba: np.ndarray, costs, notional: float) -> float:
    """Long when proba>0.5, flat otherwise. P&L over the test window minus costs."""
    pnl, prev = 0.0, 0
    qty = max(1, int(notional / prices[0])) if len(prices) else 1
    for i in range(len(prices) - 1):
        pos = 1 if proba[i] > 0.5 else 0
        pnl += pos * (prices[i + 1] - prices[i]) * qty
        if pos != prev:
            pnl -= costs.charges(prices[i], qty, "BUY" if pos else "SELL")
        prev = pos
    return pnl


def train_and_evaluate(config: Config, base_cfg: dict, days: int = 45,
                       max_symbols: int | None = 10, threshold: float = 0.002,
                       train_frac: float = 0.7) -> MLResult:
    try:
        import xgboost as xgb
    except Exception:
        return MLResult(False, "xgboost not installed (pip install xgboost).")
    from ..backtest.portfolio import fetch_decision_data, run_portfolio_over
    from ..backtest.costs import CostModel

    data, tradeable, _ = fetch_decision_data(config, base_cfg, days, max_symbols)
    costs = CostModel.from_config(config)
    capital = float(base_cfg.get("capital", 20000))
    notional = capital / max(1, len(tradeable))

    timeline = sorted(set().union(*[df.index for df in data.values()]))
    split_ts = timeline[int(len(timeline) * train_frac)]

    xgb_net = 0.0
    shap_accum: dict[str, float] = {c: 0.0 for c in FEATURE_COLS}
    shap_n = 0
    used = 0
    for sym in tradeable:
        df = data[sym]
        X, y = make_dataset(df, horizon=1, threshold=threshold)
        if len(X) < 120 or y.nunique() < 2:
            continue
        tr = X.index <= split_ts
        te = X.index > split_ts
        if tr.sum() < 60 or te.sum() < 20 or y[tr].nunique() < 2:
            continue
        model = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08,
                                  subsample=0.9, colsample_bytree=0.9,
                                  eval_metric="logloss", verbosity=0)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        prices = df.loc[X[te].index, "close"].to_numpy()
        xgb_net += _simulate_long_flat(prices, proba, costs, notional)
        used += 1
        # SHAP importance (mean|value|) on the test slice
        try:
            import shap
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(X[te])
            mean_abs = np.abs(sv).mean(axis=0)
            for c, v in zip(FEATURE_COLS, mean_abs):
                shap_accum[c] += float(v)
            shap_n += 1
        except Exception:
            pass

    if used == 0:
        return MLResult(False, "not enough data to train/evaluate.")

    # rule baseline on the SAME out-of-sample window
    test_tl = [t for t in timeline if t > split_ts]
    baseline = run_portfolio_over(config, base_cfg, data, tradeable, timeline=test_tl)

    top = sorted(((c, shap_accum[c] / max(1, shap_n)) for c in FEATURE_COLS),
                 key=lambda x: -x[1])
    return MLResult(
        ok=True, xgb_oos_net=round(xgb_net, 2),
        baseline_oos_net=round(baseline.net_pnl, 2), symbols=used,
        top_features=top, beats_baseline=xgb_net > baseline.net_pnl,
    )

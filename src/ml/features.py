"""Feature engineering for the ML signal model.

Turns a symbol's decision-bar OHLCV into a feature matrix (technical indicators
+ returns + volume) and a binary target: does the NEXT bar's return exceed a
threshold? Kept purely technical/price-derived so it is honestly backtestable
(no look-ahead from point-in-time fundamentals).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategies import indicators as ind

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "rsi", "macd", "macd_slope",
    "atr_pct", "vwap_dist", "sma_dist", "vol_ratio",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the feature columns for a decision-bar DataFrame."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["rsi"] = ind.rsi(close, 14) / 100.0
    macd = ind.macd(df)
    out["macd"] = macd / close
    out["macd_slope"] = macd.diff() / close
    out["atr_pct"] = ind.atr(df, 14) / close
    out["vwap_dist"] = (close - ind.vwap(df)) / close
    out["sma_dist"] = (close - close.rolling(20).mean()) / close
    vol = df.get("volume")
    if vol is None:
        vol = pd.Series(0.0, index=df.index)
    out["vol_ratio"] = vol / (vol.rolling(20).mean() + 1e-9)
    return out


def make_dataset(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.002):
    """Return (X, y) where y = 1 if close `horizon` bars ahead is > threshold up."""
    feats = build_features(df)
    fwd_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    y = (fwd_ret > threshold).astype(int)
    data = feats.copy()
    data["_y"] = y
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    X = data[FEATURE_COLS]
    y = data["_y"]
    return X, y

"""Gymnasium trading environment for the RL agent (single symbol, long/flat).

State  = technical feature vector (from features.build_features) + current position.
Action = 0 (flat / exit) or 1 (long / enter).
Reward = mark-to-market equity change minus a transaction cost on position flips
         (so the agent is penalized for churn, like the real bot).

Import-guarded: only usable when `gymnasium` is installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLS, build_features

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:  # pragma: no cover
    _HAS_GYM = False
    gym = object  # type: ignore


if _HAS_GYM:
    class TradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, df: pd.DataFrame, cost_bps: float = 6.0):
            super().__init__()
            feats = build_features(df).replace([np.inf, -np.inf], np.nan).dropna()
            self.prices = df.loc[feats.index, "close"].to_numpy(dtype=float)
            self.feats = feats[FEATURE_COLS].to_numpy(dtype=float)
            self.n = len(self.prices)
            self.cost = cost_bps / 10000.0
            self.action_space = spaces.Discrete(2)  # 0 flat, 1 long
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(FEATURE_COLS) + 1,), dtype=np.float32)
            self._i = 0
            self._pos = 0

        def _obs(self):
            return np.append(self.feats[self._i], float(self._pos)).astype(np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._i = 0
            self._pos = 0
            return self._obs(), {}

        def step(self, action):
            action = int(action)
            ret = (self.prices[self._i + 1] / self.prices[self._i] - 1.0) \
                if self._i + 1 < self.n else 0.0
            reward = action * ret
            if action != self._pos:
                reward -= self.cost  # penalize flipping position
            self._pos = action
            self._i += 1
            terminated = self._i >= self.n - 1
            return self._obs(), float(reward), terminated, False, {}

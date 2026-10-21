"""Tests for the parts that protect money: paper P&L accounting, the
per-order gate, and the daily kill switch (profit target + max loss)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from src.brokers.paper import PaperBroker
from src.config import Config, Secrets
from src.risk.engine import RiskEngine
from src.store.db import Store
from src.strategies import build_strategy
from src.strategies.base import StrategyContext


class StubMD:
    def __init__(self, price=100.0):
        self.price = price

    def connect(self):
        pass

    def ltp(self, symbol):
        return self.price


def make_config(**risk):
    base_risk = {
        "daily_profit_target": 30000, "daily_max_loss": 2000,
        "max_trades_per_day": 50, "max_open_positions": 5,
        "per_trade_risk_rupees": 1000, "max_capital_per_trade": 50000,
        "square_off_on_limit": True, "lock_until_manual_reset": True,
    }
    base_risk.update(risk)
    settings = {"risk": base_risk, "account": {"paper_starting_cash": 100000}}
    return Config(settings, Secrets())


@pytest.fixture()
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(Path(path))
    yield s
    s.close()
    os.unlink(path)


def test_paper_pnl_round_trip():
    md = StubMD(100.0)
    b = PaperBroker(make_config(), md)
    b.place_order("X", "BUY", 10, "MARKET")
    md.price = 110.0
    assert b.unrealized_pnl() == pytest.approx(100.0)   # +10 * 10
    b.place_order("X", "SELL", 10, "MARKET")
    assert b.realized_pnl() == pytest.approx(100.0)
    assert b.positions()["X"].is_flat


def test_max_loss_triggers_lock_and_squareoff(store):
    cfg = make_config(daily_max_loss=500)
    md = StubMD(100.0)
    b = PaperBroker(cfg, md)
    risk = RiskEngine(cfg, b, store)

    b.place_order("X", "BUY", 100, "MARKET")  # 100 shares @ 100
    md.price = 94.0                            # -6 * 100 = -600 (beyond -500)
    assert risk.total_pnl() == pytest.approx(-600.0)

    still_ok = risk.check_daily_limits()
    assert still_ok is False
    assert store.is_locked() is True
    assert b.positions()["X"].is_flat          # squared off
    # New entries are now refused.
    decision = risk.evaluate_order("X", "BUY", 1, 94.0)
    assert decision.allowed is False


def test_networth_floor_triggers_lock(store):
    # HAASEY-style: start 20k, floor 15k -> a 5k+ loss must lock the day.
    cfg = make_config(daily_max_loss=999999, daily_profit_target=999999,
                      starting_capital=20000, capital_floor=15000, capital_ceiling=25000)
    md = StubMD(100.0)
    b = PaperBroker(cfg, md)
    risk = RiskEngine(cfg, b, store)
    b.place_order("X", "BUY", 1000, "MARKET")   # 1000 @ 100
    md.price = 94.0                              # -6 * 1000 = -6000 -> worth 14000
    assert risk.net_worth() == pytest.approx(14000.0)
    assert risk.check_daily_limits() is False
    assert store.is_locked() is True


def test_networth_ceiling_triggers_lock(store):
    cfg = make_config(daily_max_loss=999999, daily_profit_target=999999,
                      starting_capital=20000, capital_floor=15000, capital_ceiling=25000)
    md = StubMD(100.0)
    b = PaperBroker(cfg, md)
    risk = RiskEngine(cfg, b, store)
    b.place_order("X", "BUY", 1000, "MARKET")
    md.price = 106.0                             # +6000 -> worth 26000 >= 25000
    assert risk.check_daily_limits() is False
    assert store.is_locked() is True


def test_profit_target_triggers_lock(store):
    cfg = make_config(daily_profit_target=1000)
    md = StubMD(100.0)
    b = PaperBroker(cfg, md)
    risk = RiskEngine(cfg, b, store)
    b.place_order("X", "BUY", 100, "MARKET")
    md.price = 112.0                            # +12 * 100 = +1200 > 1000
    assert risk.check_daily_limits() is False
    assert store.is_locked() is True


def test_capital_cap_resizes_quantity(store):
    cfg = make_config(max_capital_per_trade=10000)
    md = StubMD(100.0)
    b = PaperBroker(cfg, md)
    risk = RiskEngine(cfg, b, store)
    d = risk.evaluate_order("X", "BUY", 1000, 100.0)  # wants 1000 -> Rs.100000
    assert d.allowed is True
    assert d.quantity == 100                          # capped to Rs.10000 / 100


def test_exit_always_allowed_even_when_locked(store):
    cfg = make_config()
    md = StubMD(100.0)
    b = PaperBroker(cfg, md)
    risk = RiskEngine(cfg, b, store)
    store.lock_day("test lock")
    d = risk.evaluate_order("X", "SELL", 10, 100.0, is_exit=True)
    assert d.allowed is True


def test_heuristic_parser_extracts_correct_numbers():
    from src.agent import StrategyPlanner

    p = StrategyPlanner().propose(
        "trade GAIL buy at 150 sell at 155 qty 500, stop at 30000 profit 2000 loss"
    )
    patch = p.config_patch
    assert patch["risk"]["daily_profit_target"] == 30000
    assert patch["risk"]["daily_max_loss"] == 2000
    assert patch["strategy"]["symbol"] == "GAIL"
    assert patch["strategy"]["quantity"] == 500
    assert patch["strategy"]["type"] == "threshold"
    assert patch["strategy"]["entry_price"] == 150.0
    assert patch["strategy"]["target_price"] == 155.0


def test_indicators_on_ohlcv():
    import numpy as np
    import pandas as pd

    from src.strategies import indicators as ind

    n = 120
    close = pd.Series(np.linspace(100, 130, n) + np.sin(np.arange(n)) * 2)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01 09:15", periods=n, freq="1min"),
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": np.random.randint(1000, 5000, n).astype(float),
    })
    # RSI bounded 0..100
    r = ind.rsi(df["close"], 14)
    assert 0 <= float(r.iloc[-1]) <= 100
    # ATR positive
    assert float(ind.atr(df, 14).iloc[-1]) > 0
    # VWAP within the price range
    vw = float(ind.vwap(df).iloc[-1])
    assert df["low"].min() <= vw <= df["high"].max()
    # Supertrend trend is +1 on a rising series
    assert int(ind.supertrend(df, 10, 3.0).iloc[-1]) == 1
    # VWAP with zero volume falls back gracefully (no NaN)
    df0 = df.copy(); df0["volume"] = 0.0
    assert not np.isnan(float(ind.vwap(df0).iloc[-1]))


def test_new_strategies_run_via_backtest():
    """Smoke test: each bar-based strategy runs end-to-end on synthetic data."""
    from src.backtest import backtest

    cfg = make_config()
    cfg.settings["account"] = {"paper_starting_cash": 100000}
    for stype in ("orb_vwap", "supertrend_vwap", "rsi_reversion"):
        sc = {"name": f"t_{stype}", "type": stype, "symbol": "GAIL", "quantity": 5,
              "trend_sma": 20, "min_bars": 20, "rsi_period": 2}
        res = backtest(cfg, sc, days=2)
        assert res.bars > 0  # it processed candles without crashing


def test_ml_feature_builder():
    import numpy as np
    import pandas as pd

    from src.ml.features import FEATURE_COLS, build_features, make_dataset

    n = 200
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="15min")
    close = 100 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame({"date": idx, "open": close, "high": close + 1,
                       "low": close - 1, "close": close,
                       "volume": np.random.randint(1000, 5000, n).astype(float)})
    feats = build_features(df)
    assert list(feats.columns) == FEATURE_COLS
    X, y = make_dataset(df, horizon=1, threshold=0.002)
    assert len(X) == len(y) and set(y.unique()).issubset({0, 1})
    assert list(X.columns) == FEATURE_COLS


def test_dynamic_sizing():
    from src.risk.sizing import dynamic_size, kelly_fraction

    # Kelly: positive edge -> positive fraction; no edge -> 0
    assert kelly_fraction(0.6, 2.0) > 0
    assert kelly_fraction(0.3, 1.0) == 0.0
    # Regime scaling: bear (0.5x) sizes smaller than bull (1.0x)
    bull = dynamic_size(20000, 100, 2.0, 0.7, 0.02, 0.20, regime_mult=1.0)
    bear = dynamic_size(20000, 100, 2.0, 0.7, 0.02, 0.20, regime_mult=0.5)
    assert bear <= bull
    # Never exceeds the per-position cap (20% of 20000 / 100 = 40 shares)
    assert dynamic_size(20000, 100, 0.01, 1.0, 0.99, 0.20) <= 40


def test_multitimeframe_confluence():
    import numpy as np
    import pandas as pd

    from src.strategies import build_strategy

    strat = build_strategy({"name": "h", "type": "haasey_v3", "universe": ["X"],
                            "htf_minutes": 60, "htf_sma": 5})
    idx = pd.date_range("2024-01-01 09:15", periods=600, freq="1min")
    up = pd.DataFrame({"date": idx, "open": np.linspace(100, 140, 600),
                       "high": np.linspace(100, 140, 600) + 1,
                       "low": np.linspace(100, 140, 600) - 1,
                       "close": np.linspace(100, 140, 600),
                       "volume": 1000.0})
    down = up.copy(); down["close"] = np.linspace(140, 100, 600)
    assert strat._htf_ok(up) is True       # higher timeframe rising -> allowed
    assert strat._htf_ok(down) is False    # higher timeframe falling -> blocked


def test_regime_classifier():
    import numpy as np
    import pandas as pd

    from src.data.regime import classify_regime

    n = 80
    up = pd.DataFrame({"close": np.linspace(100, 130, n)})      # steady uptrend
    down = pd.DataFrame({"close": np.linspace(130, 100, n)})    # steady downtrend

    assert classify_regime(up, vix=12).label == "BULL"
    assert classify_regime(down, vix=12).label == "BEAR"
    # High VIX forces BEAR / no new entries even in an uptrend
    bear_vol = classify_regime(up, vix=30)
    assert bear_vol.label == "BEAR" and bear_vol.allow_new_entries is False
    # Too little data => UNKNOWN but does not block trading
    assert classify_regime(pd.DataFrame({"close": [100, 101]}), vix=12).allow_new_entries is True


def test_paper_limit_order_gating():
    md = StubMD(100.0)
    b = PaperBroker(make_config(), md)
    # BUY limit below market -> not reachable -> PENDING (no fill)
    r = b.place_order("X", "BUY", 10, "LIMIT", 95.0)
    assert not r.ok and r.status == "PENDING"
    # price falls to 94 -> limit 95 now reachable -> fills at the limit price
    md.price = 94.0
    r2 = b.place_order("X", "BUY", 10, "LIMIT", 95.0)
    assert r2.ok and r2.status == "FILLED" and r2.fill_price == 95.0


def test_walkforward_report_formatting():
    from src.backtest.walkforward import Fold, format_folds

    folds = [Fold(1, 250, 125, {"buy": 0.55, "cooldown_minutes": 60,
                                 "trail_atr_mult": 2.5}, 100.0, -50.0, -0.5, 20)]
    out = format_folds(folds)
    assert "OUT-OF-SAMPLE" in out and "overfitting" in out
    assert format_folds([]).startswith("Not enough data")


def test_threshold_strategy_buy_then_sell():
    md = StubMD(150.0)
    b = PaperBroker(make_config(), md)
    strat = build_strategy({
        "name": "t", "type": "threshold", "symbol": "GAIL", "quantity": 10,
        "entry_price": 150.0, "target_price": 155.0, "stoploss_price": 147.0,
        "re_enter": True,
    })
    ctx = StrategyContext(b)

    md.price = 150.0
    sigs = strat.on_tick(ctx)
    assert sigs and sigs[0].side == "BUY"
    b.place_order("GAIL", "BUY", 10, "MARKET")   # simulate the fill

    md.price = 156.0
    sigs = strat.on_tick(ctx)
    assert sigs and sigs[0].side == "SELL" and sigs[0].is_exit

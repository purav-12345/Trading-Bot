"""Strategy registry. Each strategy only PROPOSES signals; the risk engine
has the final say before anything is sent to the broker."""
from .base import Signal, Strategy, StrategyContext
from .threshold import ThresholdStrategy
from .momentum import SupertrendORBStrategy
from .ma_crossover import MACrossoverStrategy
from .orb_vwap import ORBVwapStrategy
from .supertrend_vwap import SupertrendVwapStrategy
from .rsi_reversion import RSIReversionStrategy
from .haasey import HaaseyStrategy
from .haasey_v3 import HaaseyV3Strategy

REGISTRY = {
    "threshold": ThresholdStrategy,
    "supertrend_orb": SupertrendORBStrategy,
    "ma_crossover": MACrossoverStrategy,
    "orb_vwap": ORBVwapStrategy,
    "supertrend_vwap": SupertrendVwapStrategy,
    "rsi_reversion": RSIReversionStrategy,
    "haasey": HaaseyStrategy,
    "haasey_v3": HaaseyV3Strategy,
}


def build_strategy(cfg: dict):
    stype = cfg.get("type")
    if stype not in REGISTRY:
        raise ValueError(f"Unknown strategy type: {stype!r}. Known: {list(REGISTRY)}")
    return REGISTRY[stype](cfg)


__all__ = ["Signal", "Strategy", "StrategyContext", "REGISTRY", "build_strategy"]

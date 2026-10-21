from .base import Broker, OrderResult, Position
from .paper import PaperBroker

__all__ = ["Broker", "OrderResult", "Position", "PaperBroker", "make_broker"]


def make_broker(config, market_data):
    """Factory: live -> Zerodha, otherwise the safe paper broker."""
    if config.secrets.is_live and config.secrets.has_kite:
        from .zerodha import ZerodhaBroker
        return ZerodhaBroker(config, market_data)
    return PaperBroker(config, market_data)

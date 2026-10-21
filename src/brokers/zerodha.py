"""Zerodha Kite Connect adapter — REAL money when TRADING_MODE=live.

Requires KITE_API_KEY + a daily KITE_ACCESS_TOKEN (see README for the
login flow) and, per SEBI rules, a STATIC IP for order placement.

Order placement via Kite is free; the data API subscription (~Rs.500/mo)
is needed for live quotes/historical. We resolve tradingsymbols to
instrument tokens lazily.
"""
from __future__ import annotations

from typing import Optional

from .base import Broker, OrderResult, Position


class ZerodhaBroker(Broker):
    name = "zerodha"

    def __init__(self, config, market_data):
        self.config = config
        self.md = market_data
        self._kite = None
        self.exchange = config.account.get("exchange", "NSE")
        self.product_code = config.account.get("product", "MIS")

    def connect(self) -> None:
        from kiteconnect import KiteConnect

        s = self.config.secrets
        if not (s.kite_api_key and s.kite_access_token):
            raise RuntimeError(
                "Live mode needs KITE_API_KEY and KITE_ACCESS_TOKEN. "
                "Run `python -m src.main kite-login` to generate today's token."
            )
        self._kite = KiteConnect(api_key=s.kite_api_key)
        self._kite.set_access_token(s.kite_access_token)
        self.md.connect()

    @property
    def kite(self):
        if self._kite is None:
            self.connect()
        return self._kite

    def ltp(self, symbol: str) -> float:
        return self.md.ltp(symbol)

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> OrderResult:
        side = side.upper()
        k = self.kite
        try:
            order_id = k.place_order(
                variety=k.VARIETY_REGULAR,
                exchange=getattr(k, f"EXCHANGE_{self.exchange}", self.exchange),
                tradingsymbol=symbol,
                transaction_type=k.TRANSACTION_TYPE_BUY if side == "BUY" else k.TRANSACTION_TYPE_SELL,
                quantity=int(quantity),
                product=getattr(k, f"PRODUCT_{self.product_code}", self.product_code),
                order_type=k.ORDER_TYPE_LIMIT if order_type.upper() == "LIMIT" else k.ORDER_TYPE_MARKET,
                price=price if order_type.upper() == "LIMIT" else None,
                validity=k.VALIDITY_DAY,
            )
            fill = price if price else self.ltp(symbol)
            return OrderResult(True, str(order_id), symbol, side, quantity, float(fill), "PENDING",
                               "submitted to exchange")
        except Exception as exc:  # noqa: BLE001 - surface broker errors verbatim
            return OrderResult(False, "", symbol, side, quantity, 0.0, "REJECTED", str(exc))

    def positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        try:
            data = self.kite.positions().get("net", [])
        except Exception:
            return out
        for p in data:
            sym = p["tradingsymbol"]
            out[sym] = Position(
                symbol=sym,
                quantity=int(p.get("quantity", 0)),
                avg_price=float(p.get("average_price", 0) or 0),
                last_price=float(p.get("last_price", 0) or 0),
                realized_pnl=float(p.get("realised", 0) or 0),
            )
        return out

    def square_off_all(self) -> list[OrderResult]:
        results = []
        for sym, pos in self.positions().items():
            if pos.quantity != 0:
                side = "SELL" if pos.quantity > 0 else "BUY"
                results.append(self.place_order(sym, side, abs(pos.quantity), "MARKET"))
        return results

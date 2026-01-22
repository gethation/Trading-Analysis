# broker_ccxt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ccxt

from core import Intent


@dataclass
class MarketRules:
    min_qty: float = 0.0
    qty_step: float = 0.0  # 0 表示不處理


def _floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return (x // step) * step


class CCXTBroker:
    def __init__(self, exchange: ccxt.Exchange, symbol: str, quote: str = "USDT"):
        self.ex = exchange
        self.symbol = symbol
        self.quote = quote
        self.rules = self._load_market_rules()

    def _load_market_rules(self) -> MarketRules:
        # 盡量從 markets 抓限制；不同交易所格式略不同，先做最小安全處理
        self.ex.load_markets()
        m = self.ex.market(self.symbol)
        limits = (m.get("limits") or {}).get("amount") or {}
        min_qty = float(limits.get("min") or 0.0)
        precision = (m.get("precision") or {}).get("amount")
        # precision 是小數位數 -> 轉 step
        qty_step = 0.0
        if isinstance(precision, int) and precision >= 0:
            qty_step = 10 ** (-precision)
        return MarketRules(min_qty=min_qty, qty_step=qty_step)

    def round_qty(self, qty: float) -> float:
        qty2 = _floor_to_step(float(qty), self.rules.qty_step)
        if qty2 < self.rules.min_qty:
            return 0.0
        return qty2

    def fetch_cash(self) -> float:
        bal = self.ex.fetch_balance()
        free = bal.get("free", {}).get(self.quote)
        if free is None:
            free = (bal.get(self.quote) or {}).get("free", 0.0)
        return float(free or 0.0)

    def fetch_position_qty(self) -> float:
        # 交易所差異很大；Binance futures 常可用 fetch_positions
        fn = getattr(self.ex, "fetch_positions", None)
        if fn is None:
            return 0.0
        ps = self.ex.fetch_positions([self.symbol])
        if not ps:
            return 0.0
        p = ps[0]
        contracts = float(p.get("contracts") or 0.0)
        side = (p.get("side") or "").lower()
        return contracts if side == "long" else (-contracts if side == "short" else 0.0)

    def cancel_all(self):
        fn = getattr(self.ex, "cancel_all_orders", None)
        if fn is not None:
            fn(self.symbol)
            return
        for o in self.ex.fetch_open_orders(self.symbol):
            self.ex.cancel_order(o["id"], self.symbol)

    def close_position_market(self, position_qty: float):
        if position_qty == 0:
            return
        side = "sell" if position_qty > 0 else "buy"
        self.ex.create_order(
            self.symbol,
            "market",
            side,
            abs(position_qty),
            None,
            {"reduceOnly": True},
        )

    def place_intent(self, intent: Intent):
        if intent.kind == "cancel_all":
            self.cancel_all()
            return
        if intent.kind == "close":
            pos = self.fetch_position_qty()
            self.close_position_market(pos)
            return

        assert intent.kind == "order"
        qty = self.round_qty(intent.qty or 0.0)
        if qty <= 0:
            return

        params = {"reduceOnly": bool(intent.reduce_only)}
        self.ex.create_order(
            self.symbol,
            intent.type,
            intent.side,
            qty,
            intent.price,
            params,
        )
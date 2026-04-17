from __future__ import annotations

import math
import os
from dataclasses import dataclass

from ...domain import OrderIntent, OrderSide, RunMode
from .market_data import require_env


@dataclass
class AlpacaBrokerConfig:
    api_key: str
    secret_key: str
    dry_run: bool = False

    @classmethod
    def from_env(cls, dry_run: bool = False) -> "AlpacaBrokerConfig":
        return cls(
            api_key=require_env("ALPACA_API_KEY"),
            secret_key=require_env("ALPACA_SECRET_KEY"),
            dry_run=dry_run or os.getenv("ALPACA_DRY_RUN", "false").lower() == "true",
        )


class AlpacaTradingBroker:
    def __init__(self, config: AlpacaBrokerConfig):
        from alpaca.trading.client import TradingClient

        self.config = config
        self.paper_client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=True,
        )
        self.live_client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=False,
        )

    def _client_for_mode(self, mode: RunMode):
        return self.paper_client if mode is RunMode.PAPER else self.live_client

    def client_for_mode(self, mode: RunMode):
        return self._client_for_mode(mode)

    def _current_position_qty(self, mode: RunMode, symbol: str) -> float:
        client = self._client_for_mode(mode)
        try:
            position = client.get_open_position(symbol)
            return float(position.qty)
        except Exception:
            return 0.0

    def _available_cash(self, mode: RunMode) -> float:
        client = self._client_for_mode(mode)
        try:
            account = client.get_account()
            return float(getattr(account, "cash", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _floor_qty(self, quantity: float, precision: int = 6) -> float:
        scale = 10**precision
        return math.floor(max(quantity, 0.0) * scale) / scale

    def submit(self, intent: OrderIntent, mode: RunMode) -> str:
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        position_qty = self._current_position_qty(mode, intent.symbol)

        if intent.side == OrderSide.BUY:
            if intent.size_fraction is not None:
                spend = round(self._available_cash(mode) * max(intent.size_fraction, 0.0), 2)
            else:
                spend = round(intent.notional_usd, 2)
            if spend <= 0:
                return f"{mode.value}-noop-insufficient-cash"
            order = MarketOrderRequest(
                symbol=intent.symbol,
                notional=spend,
                side=AlpacaOrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        elif intent.side == OrderSide.SELL:
            if position_qty <= 0:
                return f"{mode.value}-noop-no-position"
            sell_qty = abs(position_qty)
            if intent.size_fraction is not None:
                sell_qty = abs(position_qty) * max(intent.size_fraction, 0.0)
            sell_qty = self._floor_qty(sell_qty, precision=6)
            if sell_qty <= 0:
                return f"{mode.value}-noop-no-position"
            order = MarketOrderRequest(
                symbol=intent.symbol,
                qty=sell_qty,
                side=AlpacaOrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        else:
            return f"{mode.value}-noop-hold"

        if self.config.dry_run:
            return f"{mode.value}-dry-run"

        result = self._client_for_mode(mode).submit_order(order_data=order)
        return getattr(result, "id", f"{mode.value}-submitted")

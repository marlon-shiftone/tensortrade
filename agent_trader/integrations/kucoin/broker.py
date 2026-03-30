from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...domain import OrderIntent, OrderSide, RunMode, utc_now_iso
from .market_data import build_kucoin_exchange, normalize_symbol, split_symbol


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class KuCoinBrokerConfig:
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    dry_run: bool = False
    paper_state_path: str = "kucoin/state/paper_portfolio.json"
    paper_cash_usd: float = 10000.0
    enable_rate_limit: bool = True

    @classmethod
    def from_env(
        cls,
        state_dir: Path,
        dry_run: bool = False,
    ) -> "KuCoinBrokerConfig":
        return cls(
            api_key=os.getenv("KUCOIN_API_KEY", ""),
            secret_key=os.getenv("KUCOIN_SECRET_KEY", ""),
            passphrase=os.getenv("KUCOIN_PASSPHRASE", ""),
            dry_run=dry_run or os.getenv("KUCOIN_DRY_RUN", "false").lower() == "true",
            paper_state_path=str(state_dir / "paper_portfolio.json"),
            paper_cash_usd=float(os.getenv("KUCOIN_PAPER_CASH", "10000")),
            enable_rate_limit=os.getenv("KUCOIN_ENABLE_RATE_LIMIT", "true").lower() == "true",
        )

    def ensure_live_credentials(self) -> None:
        missing = [
            name
            for name, value in (
                ("KUCOIN_API_KEY", self.api_key),
                ("KUCOIN_SECRET_KEY", self.secret_key),
                ("KUCOIN_PASSPHRASE", self.passphrase),
            )
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Credenciais KuCoin ausentes para live: {joined}")


class KuCoinPaperLedger:
    def __init__(self, path: str | Path, initial_cash_usd: float = 10000.0):
        self.path = Path(path)
        self.initial_cash_usd = float(initial_cash_usd)

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "initial_cash_usd": self.initial_cash_usd,
            "cash_usd": self.initial_cash_usd,
            "realized_pnl": 0.0,
            "positions": {},
            "trade_history": [],
            "equity_curve": [],
        }

    def load(self) -> dict[str, Any]:
        self._ensure_parent()
        if not self.path.exists():
            state = self._empty_state()
            self.save(state)
            return state

        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, state: dict[str, Any]) -> None:
        self._ensure_parent()
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)

    def _equity(self, state: dict[str, Any], symbol_prices: dict[str, float]) -> float:
        equity = _safe_float(state.get("cash_usd"))
        for symbol, position in state.get("positions", {}).items():
            price = _safe_float(symbol_prices.get(symbol), _safe_float(position.get("last_price")))
            quantity = _safe_float(position.get("qty"))
            equity += quantity * max(price, 0.0)
            position["last_price"] = price
        return equity

    def _append_equity_point(
        self,
        state: dict[str, Any],
        symbol_prices: dict[str, float],
        source: str,
    ) -> float:
        equity = self._equity(state, symbol_prices)
        state.setdefault("equity_curve", []).append(
            {
                "at": utc_now_iso(),
                "equity": equity,
                "source": source,
            }
        )
        return equity

    def record_mark_to_market(
        self,
        symbol_prices: dict[str, float],
        source: str = "mark_to_market",
    ) -> dict[str, Any]:
        state = self.load()
        self._append_equity_point(state, symbol_prices, source=source)
        self.save(state)
        return state

    def submit(self, intent: OrderIntent) -> str:
        symbol = normalize_symbol(intent.symbol)
        reference_price = _safe_float(intent.reference_price)
        if reference_price <= 0:
            raise ValueError("reference_price obrigatorio para ordens paper da KuCoin")

        state = self.load()
        position = state.setdefault("positions", {}).get(symbol, {})
        current_qty = _safe_float(position.get("qty"))

        if intent.side is OrderSide.BUY:
            if current_qty > 0:
                return "paper-noop-already-long"

            spend = min(_safe_float(intent.notional_usd), _safe_float(state.get("cash_usd")))
            if spend <= 0:
                return "paper-noop-insufficient-cash"

            quantity = spend / reference_price
            state["cash_usd"] = _safe_float(state.get("cash_usd")) - spend
            state["positions"][symbol] = {
                "qty": quantity,
                "cost_basis_usd": spend,
                "last_price": reference_price,
                "opened_at": utc_now_iso(),
            }
            state.setdefault("trade_history", []).append(
                {
                    "at": utc_now_iso(),
                    "symbol": symbol,
                    "side": "buy",
                    "qty": quantity,
                    "price": reference_price,
                    "notional": spend,
                    "realized_pnl": 0.0,
                }
            )
            reference = f"paper-buy-{len(state['trade_history'])}"
            self._append_equity_point(state, {symbol: reference_price}, source=reference)
            self.save(state)
            return reference

        if intent.side is OrderSide.SELL:
            if current_qty <= 0:
                return "paper-noop-no-position"

            cost_basis = _safe_float(position.get("cost_basis_usd"))
            proceeds = current_qty * reference_price
            realized_pnl = proceeds - cost_basis
            state["cash_usd"] = _safe_float(state.get("cash_usd")) + proceeds
            state["realized_pnl"] = _safe_float(state.get("realized_pnl")) + realized_pnl
            state["positions"].pop(symbol, None)
            state.setdefault("trade_history", []).append(
                {
                    "at": utc_now_iso(),
                    "symbol": symbol,
                    "side": "sell",
                    "qty": current_qty,
                    "price": reference_price,
                    "notional": proceeds,
                    "realized_pnl": realized_pnl,
                    "cost_basis_usd": cost_basis,
                }
            )
            reference = f"paper-sell-{len(state['trade_history'])}"
            self._append_equity_point(state, {symbol: reference_price}, source=reference)
            self.save(state)
            return reference

        return "paper-noop-hold"


class KuCoinTradingBroker:
    def __init__(self, config: KuCoinBrokerConfig):
        self.config = config
        self.paper_ledger = KuCoinPaperLedger(
            path=config.paper_state_path,
            initial_cash_usd=config.paper_cash_usd,
        )
        self.live_client = None

        if config.api_key and config.secret_key and config.passphrase:
            self.live_client = build_kucoin_exchange(
                api_key=config.api_key,
                secret_key=config.secret_key,
                passphrase=config.passphrase,
                enable_rate_limit=config.enable_rate_limit,
            )

    def _client_for_live(self):
        if self.live_client is None:
            self.config.ensure_live_credentials()
            self.live_client = build_kucoin_exchange(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                passphrase=self.config.passphrase,
                enable_rate_limit=self.config.enable_rate_limit,
            )
        return self.live_client

    def client_for_mode(self, mode: RunMode):
        return self.paper_ledger if mode is RunMode.PAPER else self._client_for_live()

    def _current_live_position_qty(self, symbol: str) -> float:
        exchange = self._client_for_live()
        base_asset, _quote_asset = split_symbol(symbol)
        balance = exchange.fetch_balance()
        totals = balance.get("total", {})
        return _safe_float(totals.get(base_asset))

    def submit(self, intent: OrderIntent, mode: RunMode) -> str:
        symbol = normalize_symbol(intent.symbol)

        if mode is RunMode.PAPER:
            return self.paper_ledger.submit(intent)

        if self.config.dry_run:
            return f"{mode.value}-dry-run"

        exchange = self._client_for_live()
        position_qty = self._current_live_position_qty(symbol)

        if intent.side is OrderSide.BUY:
            if position_qty > 0:
                return f"{mode.value}-noop-already-long"
            result = exchange.create_market_buy_order_with_cost(
                symbol,
                round(intent.notional_usd, 8),
            )
            return str(
                result.get("id")
                or result.get("clientOrderId")
                or result.get("info", {}).get("orderId")
                or f"{mode.value}-submitted"
            )

        if intent.side is OrderSide.SELL:
            if position_qty <= 0:
                return f"{mode.value}-noop-no-position"
            result = exchange.create_order(symbol, "market", "sell", position_qty)
            return str(
                result.get("id")
                or result.get("clientOrderId")
                or result.get("info", {}).get("orderId")
                or f"{mode.value}-submitted"
            )

        return f"{mode.value}-noop-hold"

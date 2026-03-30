from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...domain import ModelRecord, PerformanceSnapshot, RunMode
from ...features import engineer_features
from .broker import KuCoinBrokerConfig, KuCoinPaperLedger
from .market_data import KuCoinHistoricalDataClient, build_kucoin_exchange, normalize_symbol, split_symbol


@dataclass
class KuCoinMetricsConfig:
    after_days: int = 30
    trades_limit: int = 500
    equity_curve_limit: int = 500


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _annualization_factor() -> float:
    return float(np.sqrt(252))


def _compute_sharpe_from_equity_curve(equity: np.ndarray) -> float:
    if equity.size < 2:
        return 0.0
    returns = np.diff(equity) / np.where(equity[:-1] == 0, 1.0, equity[:-1])
    if returns.size < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float((returns.mean() / std) * _annualization_factor())


def _compute_max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = np.where(rolling_max == 0, 0.0, (rolling_max - equity) / rolling_max)
    return float(drawdowns.max()) if drawdowns.size else 0.0


def _compute_feature_drift(reference: dict[str, Any], current_features: pd.DataFrame) -> float:
    if not reference:
        return 0.0

    reference_means = reference.get("feature_means", {})
    reference_stds = reference.get("feature_stds", {})
    if not reference_means:
        return 0.0

    recent = current_features.tail(min(50, len(current_features)))
    if recent.empty:
        return 0.0

    z_scores = []
    for feature_name, train_mean in reference_means.items():
        if feature_name not in recent.columns:
            continue
        current_mean = float(recent[feature_name].mean())
        train_std = float(reference_stds.get(feature_name, 0.0))
        denom = train_std if abs(train_std) > 1e-8 else 1.0
        z_scores.append(abs(current_mean - float(train_mean)) / denom)

    if not z_scores:
        return 0.0

    return float(np.mean(z_scores))


def _compute_trade_stats(trades: list[dict[str, Any]]) -> tuple[int, float, float]:
    if not trades:
        return 0, 0.0, 0.0

    ordered_trades = sorted(trades, key=lambda trade: trade.get("timestamp") or trade.get("at", ""))
    active_buy_notional = 0.0
    profitable_round_trips = 0
    total_round_trips = 0
    turnover = 0.0

    for trade in ordered_trades:
        side = str(trade.get("side", "")).lower()
        notional = _safe_float(
            trade.get("notional"),
            _safe_float(trade.get("cost"), _safe_float(trade.get("amount")) * _safe_float(trade.get("price"))),
        )
        turnover += notional
        if side == "buy":
            active_buy_notional += notional
            continue

        if side == "sell" and active_buy_notional > 0:
            total_round_trips += 1
            if notional > active_buy_notional:
                profitable_round_trips += 1
            active_buy_notional = 0.0

    hit_rate = profitable_round_trips / total_round_trips if total_round_trips else 0.0
    return total_round_trips, hit_rate, turnover


class KuCoinLiveStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        self._ensure_parent()
        if not self.path.exists():
            state = {
                "initial_equity": None,
                "equity_curve": [],
            }
            self.save(state)
            return state

        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, state: dict[str, Any]) -> None:
        self._ensure_parent()
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)

    def append_equity(self, equity: float, limit: int) -> dict[str, Any]:
        state = self.load()
        if state.get("initial_equity") is None:
            state["initial_equity"] = equity

        state.setdefault("equity_curve", []).append(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "equity": equity,
            }
        )
        if limit > 0:
            state["equity_curve"] = state["equity_curve"][-limit:]
        self.save(state)
        return state


class KuCoinMetricsCollector:
    def __init__(
        self,
        broker_config: KuCoinBrokerConfig,
        market_data_client: KuCoinHistoricalDataClient,
        state_dir: str | Path,
        config: KuCoinMetricsConfig | None = None,
    ):
        self.broker_config = broker_config
        self.market_data_client = market_data_client
        self.state_dir = Path(state_dir)
        self.config = config or KuCoinMetricsConfig()
        self.paper_ledger = KuCoinPaperLedger(
            path=broker_config.paper_state_path,
            initial_cash_usd=broker_config.paper_cash_usd,
        )
        self.live_state = KuCoinLiveStateStore(self.state_dir / "live_account_state.json")
        self.live_client = None

    def _live_exchange(self):
        if self.live_client is None:
            self.broker_config.ensure_live_credentials()
            self.live_client = build_kucoin_exchange(
                api_key=self.broker_config.api_key,
                secret_key=self.broker_config.secret_key,
                passphrase=self.broker_config.passphrase,
                enable_rate_limit=self.broker_config.enable_rate_limit,
            )
        return self.live_client

    def _collect_features(
        self,
        symbol: str,
        timeframe_amount: int,
        timeframe_unit: str,
        lookback_bars: int,
    ) -> tuple[pd.DataFrame, float]:
        bars = self.market_data_client.load_recent_bars(
            symbol=symbol,
            timeframe_amount=timeframe_amount,
            timeframe_unit=timeframe_unit,
            lookback_bars=lookback_bars,
        )
        feature_df = engineer_features(bars)
        latest_close = float(feature_df["close"].iloc[-1])
        return feature_df, latest_close

    def _paper_snapshot(
        self,
        model: ModelRecord,
        symbol: str,
        feature_df: pd.DataFrame,
        latest_close: float,
    ) -> PerformanceSnapshot:
        ledger_state = self.paper_ledger.record_mark_to_market(
            symbol_prices={normalize_symbol(symbol): latest_close},
            source="metrics",
        )
        equity_curve = np.array(
            [float(point["equity"]) for point in ledger_state.get("equity_curve", [])],
            dtype=float,
        )
        current_equity = equity_curve[-1] if equity_curve.size else _safe_float(ledger_state.get("cash_usd"))
        net_profit = current_equity - _safe_float(ledger_state.get("initial_cash_usd"))
        trades, hit_rate, turnover = _compute_trade_stats(ledger_state.get("trade_history", []))
        feature_drift = _compute_feature_drift(model.metadata.get("feature_reference", {}), feature_df)
        paper_days = 0
        if ledger_state.get("equity_curve"):
            dates = {
                datetime.fromisoformat(point["at"]).date()
                for point in ledger_state["equity_curve"]
            }
            paper_days = len(dates)

        return PerformanceSnapshot(
            symbol=normalize_symbol(symbol),
            net_profit=net_profit,
            rolling_sharpe=_compute_sharpe_from_equity_curve(equity_curve),
            max_drawdown=_compute_max_drawdown(equity_curve),
            hit_rate=hit_rate,
            turnover=turnover,
            feature_drift=feature_drift,
            paper_days=paper_days,
            trades=trades,
            metadata={
                "collector_mode": RunMode.PAPER.value,
                "current_equity": current_equity,
                "realized_pnl": _safe_float(ledger_state.get("realized_pnl")),
                "equity_points": int(equity_curve.size),
                "feature_rows": len(feature_df),
            },
        )

    def _live_snapshot(
        self,
        model: ModelRecord,
        symbol: str,
        feature_df: pd.DataFrame,
        latest_close: float,
    ) -> PerformanceSnapshot:
        exchange = self._live_exchange()
        normalized_symbol = normalize_symbol(symbol)
        base_asset, quote_asset = split_symbol(normalized_symbol)
        balance = exchange.fetch_balance()
        totals = balance.get("total", {})
        base_total = _safe_float(totals.get(base_asset))
        quote_total = _safe_float(totals.get(quote_asset))
        current_equity = quote_total + (base_total * latest_close)
        live_state = self.live_state.append_equity(
            equity=current_equity,
            limit=self.config.equity_curve_limit,
        )
        equity_curve = np.array(
            [float(point["equity"]) for point in live_state.get("equity_curve", [])],
            dtype=float,
        )
        initial_equity = _safe_float(live_state.get("initial_equity"), default=current_equity)
        net_profit = current_equity - initial_equity

        since_ms = int(
            (datetime.now(timezone.utc) - timedelta(days=self.config.after_days)).timestamp() * 1000
        )
        trades_payload = exchange.fetch_my_trades(
            normalized_symbol,
            since=since_ms,
            limit=self.config.trades_limit,
        )
        trades, hit_rate, turnover = _compute_trade_stats(trades_payload)
        feature_drift = _compute_feature_drift(model.metadata.get("feature_reference", {}), feature_df)
        tracked_days = 0
        if live_state.get("equity_curve"):
            dates = {
                datetime.fromisoformat(point["at"]).date()
                for point in live_state["equity_curve"]
            }
            tracked_days = len(dates)

        return PerformanceSnapshot(
            symbol=normalized_symbol,
            net_profit=net_profit,
            rolling_sharpe=_compute_sharpe_from_equity_curve(equity_curve),
            max_drawdown=_compute_max_drawdown(equity_curve),
            hit_rate=hit_rate,
            turnover=turnover,
            feature_drift=feature_drift,
            paper_days=tracked_days,
            trades=trades,
            metadata={
                "collector_mode": RunMode.LIVE.value,
                "current_equity": current_equity,
                "base_asset": base_asset,
                "base_total": base_total,
                "quote_asset": quote_asset,
                "quote_total": quote_total,
                "equity_points": int(equity_curve.size),
                "trades_evaluated": len(trades_payload),
                "feature_rows": len(feature_df),
            },
        )

    def collect_snapshot(
        self,
        model: ModelRecord,
        symbol: str,
        timeframe_amount: int,
        timeframe_unit: str,
        lookback_bars: int,
        mode: RunMode,
    ) -> PerformanceSnapshot:
        feature_df, latest_close = self._collect_features(
            symbol=symbol,
            timeframe_amount=timeframe_amount,
            timeframe_unit=timeframe_unit,
            lookback_bars=lookback_bars,
        )
        if mode is RunMode.PAPER:
            return self._paper_snapshot(
                model=model,
                symbol=symbol,
                feature_df=feature_df,
                latest_close=latest_close,
            )
        return self._live_snapshot(
            model=model,
            symbol=symbol,
            feature_df=feature_df,
            latest_close=latest_close,
        )

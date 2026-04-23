from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...domain import ModelRecord, PerformanceSnapshot, RunMode
from ...features import engineer_features
from ...storage import JsonlStore
from .market_data import AlpacaHistoricalDataClient


@dataclass
class AlpacaMetricsConfig:
    history_period: str = "1M"
    history_timeframe: str = "1D"
    orders_limit: int = 500
    after_days: int = 30
    execution_history_path: str | None = None
    decision_lookback_limit: int = 500


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _annualization_factor(timeframe: str) -> float:
    timeframe_normalized = timeframe.strip().lower()
    if timeframe_normalized.endswith("min"):
        return float(np.sqrt(252 * 390))
    if timeframe_normalized.endswith("h"):
        return float(np.sqrt(252 * 6.5))
    if timeframe_normalized.endswith("w"):
        return float(np.sqrt(52))
    return float(np.sqrt(252))


def _compute_sharpe_from_equity_curve(equity: np.ndarray, timeframe: str) -> float:
    if equity.size < 2:
        return 0.0
    returns = np.diff(equity) / np.where(equity[:-1] == 0, 1.0, equity[:-1])
    if returns.size < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float((returns.mean() / std) * _annualization_factor(timeframe))


def _compute_max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = np.divide(
        rolling_max - equity,
        rolling_max,
        out=np.zeros_like(equity, dtype=float),
        where=rolling_max != 0,
    )
    return float(drawdowns.max()) if drawdowns.size else 0.0


def _filled_qty(order) -> float:
    qty = _safe_float(getattr(order, "filled_qty", None))
    if qty > 0:
        return qty
    return _safe_float(getattr(order, "qty", None))


def _filled_notional(order) -> float:
    qty = _filled_qty(order)
    price = _safe_float(getattr(order, "filled_avg_price", None))
    notional = _safe_float(getattr(order, "notional", None))
    if notional > 0:
        return notional
    return qty * price


def _compute_trade_stats(orders: list[Any]) -> tuple[int, float, float]:
    filled_orders = [
        order for order in orders
        if getattr(order, "filled_at", None) is not None and _filled_qty(order) > 0
    ]
    filled_orders.sort(key=lambda order: getattr(order, "filled_at"))

    active_buy_notional = 0.0
    profitable_round_trips = 0
    total_round_trips = 0
    turnover = 0.0

    for order in filled_orders:
        side = getattr(getattr(order, "side", None), "value", "").lower()
        turnover += _filled_notional(order)
        if side == "buy":
            active_buy_notional += _filled_notional(order)
            continue

        if side == "sell" and active_buy_notional > 0:
            sell_notional = _filled_notional(order)
            total_round_trips += 1
            if sell_notional > active_buy_notional:
                profitable_round_trips += 1
            active_buy_notional = 0.0

    hit_rate = profitable_round_trips / total_round_trips if total_round_trips else 0.0
    return total_round_trips, hit_rate, turnover


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


class AlpacaMetricsCollector:
    def __init__(
        self,
        trading_client,
        market_data_client: AlpacaHistoricalDataClient,
        config: AlpacaMetricsConfig | None = None,
    ):
        self.trading_client = trading_client
        self.market_data_client = market_data_client
        self.config = config or AlpacaMetricsConfig()

    def _execution_decision_stats(self, model: ModelRecord, mode: RunMode) -> dict[str, Any]:
        path = getattr(self.config, "execution_history_path", None)
        if not path:
            return {
                "decision_records": 0,
                "submitted_decisions": 0,
                "buy_decisions": 0,
                "sell_decisions": 0,
                "hold_decisions": 0,
                "noop_decisions": 0,
                "flat_sell_noops": 0,
                "max_consecutive_flat_sell_noops": 0,
                "latest_decision_at": None,
            }

        store_path = Path(path)
        if not store_path.exists():
            return {
                "decision_records": 0,
                "submitted_decisions": 0,
                "buy_decisions": 0,
                "sell_decisions": 0,
                "hold_decisions": 0,
                "noop_decisions": 0,
                "flat_sell_noops": 0,
                "max_consecutive_flat_sell_noops": 0,
                "latest_decision_at": None,
            }

        window_start = self._model_window_start(model)
        records = JsonlStore(store_path).read_all()
        filtered_records = []
        for record in records:
            if record.get("model_id") != model.model_id or record.get("mode") != mode.value:
                continue
            payload = record.get("payload", {})
            if "result" not in payload:
                continue
            try:
                recorded_at = datetime.fromisoformat(record.get("recorded_at"))
            except (TypeError, ValueError):
                continue
            if recorded_at.tzinfo is None:
                recorded_at = recorded_at.replace(tzinfo=timezone.utc)
            recorded_at = recorded_at.astimezone(timezone.utc)
            if recorded_at < window_start:
                continue
            filtered_records.append((recorded_at, record))

        filtered_records.sort(key=lambda item: item[0])
        limit = max(1, int(getattr(self.config, "decision_lookback_limit", 500)))
        filtered_records = filtered_records[-limit:]

        stats = {
            "decision_records": 0,
            "submitted_decisions": 0,
            "buy_decisions": 0,
            "sell_decisions": 0,
            "hold_decisions": 0,
            "noop_decisions": 0,
            "flat_sell_noops": 0,
            "max_consecutive_flat_sell_noops": 0,
            "max_consecutive_holds": 0,
            "latest_decision_at": None,
        }
        consecutive_flat_sell_noops = 0
        consecutive_holds = 0

        for recorded_at, record in filtered_records:
            result = record.get("payload", {}).get("result", {})
            intent = result.get("intent", {})
            side = str(intent.get("side", "hold")).lower()
            reference = str(result.get("reference", "") or result.get("reason", "")).lower()

            stats["decision_records"] += 1
            if result.get("submitted"):
                stats["submitted_decisions"] += 1
            if side == "buy":
                stats["buy_decisions"] += 1
            elif side == "sell":
                stats["sell_decisions"] += 1
                consecutive_holds = 0
            else:
                stats["hold_decisions"] += 1
                consecutive_holds += 1
                stats["max_consecutive_holds"] = max(
                    stats["max_consecutive_holds"],
                    consecutive_holds,
                )

            if "noop" in reference:
                stats["noop_decisions"] += 1

            if side == "sell" and "noop-no-position" in reference:
                stats["flat_sell_noops"] += 1
                consecutive_flat_sell_noops += 1
                stats["max_consecutive_flat_sell_noops"] = max(
                    stats["max_consecutive_flat_sell_noops"],
                    consecutive_flat_sell_noops,
                )
            else:
                consecutive_flat_sell_noops = 0
                if side != "hold":
                    consecutive_holds = 0

            stats["latest_decision_at"] = recorded_at.isoformat()

        return stats

    def _model_window_start(self, model: ModelRecord) -> datetime:
        fallback_start = datetime.now(timezone.utc) - timedelta(days=self.config.after_days)
        created_at = getattr(model, "created_at", None)
        if not created_at:
            return fallback_start
        try:
            created_at_dt = datetime.fromisoformat(created_at)
        except ValueError:
            return fallback_start
        if created_at_dt.tzinfo is None:
            created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)
        return max(fallback_start, created_at_dt.astimezone(timezone.utc))

    def _portfolio_history(self, model: ModelRecord):
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        window_start = self._model_window_start(model)
        request = GetPortfolioHistoryRequest(
            timeframe=self.config.history_timeframe,
            start=window_start,
            end=datetime.now(timezone.utc),
            extended_hours=False,
        )
        return self.trading_client.get_portfolio_history(request)

    def _closed_orders(self, symbol: str, model: ModelRecord) -> list[Any]:
        from alpaca.common.enums import Sort
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        window_start = self._model_window_start(model)
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=self.config.orders_limit,
            after=window_start,
            direction=Sort.ASC,
            nested=False,
            symbols=[symbol],
        )
        return self.trading_client.get_orders(filter=request)

    def _account_equity(self) -> tuple[float, float]:
        account = self.trading_client.get_account()
        equity = _safe_float(getattr(account, "equity", None))
        last_equity = _safe_float(getattr(account, "last_equity", None), default=equity)
        return equity, last_equity

    def collect_snapshot(
        self,
        model: ModelRecord,
        symbol: str,
        timeframe_amount: int,
        timeframe_unit: str,
        lookback_bars: int,
    ) -> PerformanceSnapshot:
        portfolio_history = self._portfolio_history(model)
        timestamps = np.array(getattr(portfolio_history, "timestamp", []) or [])
        equity_values = np.array(getattr(portfolio_history, "equity", []) or [], dtype=float)
        profit_loss = np.array(getattr(portfolio_history, "profit_loss", []) or [], dtype=float)

        if equity_values.size == 0:
            equity, last_equity = self._account_equity()
            equity_values = np.array([last_equity, equity], dtype=float)
            profit_loss = np.array([0.0, equity - last_equity], dtype=float)
            timestamps = np.array([0, 1], dtype=int)

        net_profit = float(profit_loss[-1]) if profit_loss.size else float(equity_values[-1] - equity_values[0])
        rolling_sharpe = _compute_sharpe_from_equity_curve(equity_values, self.config.history_timeframe)
        max_drawdown = _compute_max_drawdown(equity_values)

        orders = self._closed_orders(symbol, model)
        trades, hit_rate, turnover = _compute_trade_stats(orders)
        execution_stats = self._execution_decision_stats(model=model, mode=RunMode.PAPER)

        bars = self.market_data_client.load_recent_bars(
            symbol=symbol,
            timeframe_amount=timeframe_amount,
            timeframe_unit=timeframe_unit,
            lookback_bars=lookback_bars,
        )
        feature_df = engineer_features(bars)
        feature_reference = model.metadata.get("feature_reference", {})
        feature_drift = _compute_feature_drift(feature_reference, feature_df)

        paper_days = 0
        if timestamps.size:
            paper_days = max(1, int(len(np.unique(pd.to_datetime(timestamps, unit="s").date))))

        return PerformanceSnapshot(
            symbol=symbol,
            net_profit=net_profit,
            rolling_sharpe=rolling_sharpe,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            turnover=turnover,
            feature_drift=feature_drift,
            paper_days=paper_days,
            trades=trades,
            metadata={
                "model_created_at": model.created_at,
                "metrics_window_start": self._model_window_start(model).isoformat(),
                "history_period": self.config.history_period,
                "history_timeframe": self.config.history_timeframe,
                "equity_points": int(equity_values.size),
                "orders_evaluated": len(orders),
                "feature_rows": len(feature_df),
                **execution_stats,
            },
        )

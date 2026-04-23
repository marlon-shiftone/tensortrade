from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...domain import OrderIntent, OrderSide, PerformanceSnapshot, utc_now_iso
from ...metrics import HealthThresholds, evaluate_snapshot


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_sharpe_from_equity_curve(equity: np.ndarray) -> float:
    if equity.size < 2:
        return 0.0
    returns = np.diff(equity) / np.where(equity[:-1] == 0, 1.0, equity[:-1])
    if returns.size < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float((returns.mean() / std) * np.sqrt(252))


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


def _compute_trade_stats(trades: list[dict[str, Any]]) -> tuple[int, float, float]:
    if not trades:
        return 0, 0.0, 0.0

    ordered_trades = sorted(trades, key=lambda trade: trade.get("at", ""))
    active_buy_notional = 0.0
    profitable_round_trips = 0
    total_round_trips = 0
    turnover = 0.0

    for trade in ordered_trades:
        side = str(trade.get("side", "")).lower()
        notional = _safe_float(trade.get("notional"))
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


def compute_feature_drift(reference: dict[str, Any], current_features: pd.DataFrame) -> float:
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


@dataclass
class ChallengerPromotionPolicy:
    evaluation_iterations: int = 10
    evaluation_min_trades: int = 2
    challenger_wins_required: int = 3
    max_evaluation_iterations: int = 120
    max_zero_trade_iterations: int = 30
    allow_rebase_promotion: bool = True


@dataclass
class OnlineTrainingState:
    champion_model_id: str | None = None
    challenger_model_id: str | None = None
    evaluation_iterations: int = 0
    challenger_created_at: str | None = None
    last_seen_bar_at: str | None = None
    last_processed_bar_at: str | None = None
    last_cycle_status: str | None = None
    last_challenger_retired_at: str | None = None
    last_challenger_retired_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion_model_id": self.champion_model_id,
            "challenger_model_id": self.challenger_model_id,
            "evaluation_iterations": self.evaluation_iterations,
            "challenger_created_at": self.challenger_created_at,
            "last_seen_bar_at": self.last_seen_bar_at,
            "last_processed_bar_at": self.last_processed_bar_at,
            "last_cycle_status": self.last_cycle_status,
            "last_challenger_retired_at": self.last_challenger_retired_at,
            "last_challenger_retired_reason": self.last_challenger_retired_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OnlineTrainingState":
        return cls(
            champion_model_id=payload.get("champion_model_id"),
            challenger_model_id=payload.get("challenger_model_id"),
            evaluation_iterations=payload.get("evaluation_iterations", 0),
            challenger_created_at=payload.get("challenger_created_at"),
            last_seen_bar_at=payload.get("last_seen_bar_at"),
            last_processed_bar_at=payload.get("last_processed_bar_at"),
            last_cycle_status=payload.get("last_cycle_status"),
            last_challenger_retired_at=payload.get("last_challenger_retired_at"),
            last_challenger_retired_reason=payload.get("last_challenger_retired_reason"),
        )


class OnlineTrainingStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> OnlineTrainingState:
        self._ensure_parent()
        if not self.path.exists():
            return OnlineTrainingState()

        with self.path.open("r", encoding="utf-8") as handle:
            return OnlineTrainingState.from_dict(json.load(handle))

    def save(self, state: OnlineTrainingState) -> dict[str, Any]:
        payload = state.to_dict()
        self._ensure_parent()
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return payload


class ShadowPaperLedger:
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

    def reset(self) -> dict[str, Any]:
        state = self._empty_state()
        self.save(state)
        return state

    def _equity(self, state: dict[str, Any], symbol_prices: dict[str, float]) -> float:
        equity = _safe_float(state.get("cash_usd"))
        for symbol, position in state.get("positions", {}).items():
            price = _safe_float(symbol_prices.get(symbol), _safe_float(position.get("last_price")))
            quantity = _safe_float(position.get("qty"))
            equity += quantity * max(price, 0.0)
            position["last_price"] = price
        return equity

    def record_mark_to_market(
        self,
        symbol_prices: dict[str, float],
        source: str = "shadow_mark_to_market",
    ) -> dict[str, Any]:
        state = self.load()
        equity = self._equity(state, symbol_prices)
        state.setdefault("equity_curve", []).append(
            {
                "at": utc_now_iso(),
                "equity": equity,
                "source": source,
            }
        )
        self.save(state)
        return state

    def submit(self, intent: OrderIntent) -> str:
        reference_price = _safe_float(intent.reference_price)
        if reference_price <= 0:
            raise ValueError("reference_price obrigatorio para shadow paper ledger")

        state = self.load()
        symbol = intent.symbol.upper()
        position = state.setdefault("positions", {}).get(symbol, {})
        current_qty = _safe_float(position.get("qty"))

        if intent.side is OrderSide.BUY:
            if current_qty > 0 and intent.size_fraction is None:
                return "shadow-noop-already-long"
            requested_spend = (
                _safe_float(state.get("cash_usd")) * _safe_float(intent.size_fraction)
                if intent.size_fraction is not None
                else _safe_float(intent.notional_usd)
            )
            spend = min(requested_spend, _safe_float(state.get("cash_usd")))
            if spend <= 0:
                return "shadow-noop-insufficient-cash"
            quantity = spend / reference_price
            new_qty = current_qty + quantity
            new_cost_basis = _safe_float(position.get("cost_basis_usd")) + spend
            state["cash_usd"] = _safe_float(state.get("cash_usd")) - spend
            state["positions"][symbol] = {
                "qty": new_qty,
                "cost_basis_usd": new_cost_basis,
                "last_price": reference_price,
                "opened_at": position.get("opened_at") or utc_now_iso(),
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
        elif intent.side is OrderSide.SELL:
            if current_qty <= 0:
                return "shadow-noop-no-position"
            sell_qty = current_qty
            if intent.size_fraction is not None:
                sell_qty = min(current_qty, current_qty * _safe_float(intent.size_fraction))
            if sell_qty <= 0:
                return "shadow-noop-no-position"
            cost_basis = _safe_float(position.get("cost_basis_usd"))
            cost_basis_sold = cost_basis * (sell_qty / current_qty)
            proceeds = sell_qty * reference_price
            realized_pnl = proceeds - cost_basis_sold
            state["cash_usd"] = _safe_float(state.get("cash_usd")) + proceeds
            state["realized_pnl"] = _safe_float(state.get("realized_pnl")) + realized_pnl
            remaining_qty = current_qty - sell_qty
            if remaining_qty <= 1e-12:
                state["positions"].pop(symbol, None)
            else:
                state["positions"][symbol] = {
                    "qty": remaining_qty,
                    "cost_basis_usd": cost_basis - cost_basis_sold,
                    "last_price": reference_price,
                    "opened_at": position.get("opened_at") or utc_now_iso(),
                }
            state.setdefault("trade_history", []).append(
                {
                    "at": utc_now_iso(),
                    "symbol": symbol,
                    "side": "sell",
                    "qty": sell_qty,
                    "price": reference_price,
                    "notional": proceeds,
                    "realized_pnl": realized_pnl,
                    "cost_basis_usd": cost_basis_sold,
                }
            )
        else:
            return "shadow-noop-hold"

        self.save(state)
        return f"shadow-{len(state['trade_history'])}"

    def build_snapshot(
        self,
        symbol: str,
        feature_drift: float,
        metadata: dict[str, Any] | None = None,
    ) -> PerformanceSnapshot:
        state = self.load()
        equity_curve = np.array(
            [float(point["equity"]) for point in state.get("equity_curve", [])],
            dtype=float,
        )
        current_equity = equity_curve[-1] if equity_curve.size else _safe_float(state.get("cash_usd"))
        trades, hit_rate, turnover = _compute_trade_stats(state.get("trade_history", []))
        paper_days = 0
        if state.get("equity_curve"):
            dates = {
                datetime.fromisoformat(point["at"]).date()
                for point in state["equity_curve"]
            }
            paper_days = len(dates)

        return PerformanceSnapshot(
            symbol=symbol,
            net_profit=current_equity - _safe_float(state.get("initial_cash_usd")),
            rolling_sharpe=_compute_sharpe_from_equity_curve(equity_curve),
            max_drawdown=_compute_max_drawdown(equity_curve),
            hit_rate=hit_rate,
            turnover=turnover,
            feature_drift=feature_drift,
            paper_days=paper_days,
            trades=trades,
            metadata=metadata or {},
        )


def evaluate_challenger_promotion(
    champion_snapshot: PerformanceSnapshot,
    challenger_snapshot: PerformanceSnapshot,
    evaluation_iterations: int,
    thresholds: HealthThresholds,
    policy: ChallengerPromotionPolicy,
) -> dict[str, Any]:
    champion_health = evaluate_snapshot(champion_snapshot, thresholds)
    challenger_health = evaluate_snapshot(challenger_snapshot, thresholds)

    comparisons = {
        "net_profit": challenger_snapshot.net_profit >= champion_snapshot.net_profit,
        "rolling_sharpe": challenger_snapshot.rolling_sharpe >= champion_snapshot.rolling_sharpe,
        "hit_rate": challenger_snapshot.hit_rate >= champion_snapshot.hit_rate,
        "max_drawdown": challenger_snapshot.max_drawdown <= champion_snapshot.max_drawdown,
    }
    challenger_wins = sum(1 for won in comparisons.values() if won)
    required_wins = policy.challenger_wins_required
    if not champion_health.healthy:
        required_wins = max(2, required_wins - 1)

    eligible = all(
        [
            evaluation_iterations >= policy.evaluation_iterations,
            challenger_snapshot.trades >= policy.evaluation_min_trades,
            challenger_health.healthy,
            challenger_wins >= required_wins,
        ]
    )

    rebase_eligible = all(
        [
            policy.allow_rebase_promotion,
            evaluation_iterations >= policy.evaluation_iterations,
            champion_health.requires_retraining,
            challenger_snapshot.trades == 0,
            challenger_snapshot.feature_drift <= thresholds.max_feature_drift,
            challenger_snapshot.max_drawdown <= thresholds.max_drawdown_retrain,
            challenger_snapshot.turnover <= thresholds.max_turnover,
        ]
    )

    activity_rebase_eligible = all(
        [
            policy.allow_rebase_promotion,
            evaluation_iterations >= 1,
            champion_health.requires_retraining,
            champion_snapshot.turnover <= 0.0,
            challenger_snapshot.turnover > 0.0,
            challenger_snapshot.max_drawdown <= thresholds.max_drawdown_retrain,
            challenger_snapshot.feature_drift <= max(
                thresholds.max_feature_drift * 2.0,
                champion_snapshot.feature_drift + 0.15,
            ),
        ]
    )

    replacement_reasons = []
    if (
        evaluation_iterations >= policy.max_zero_trade_iterations
        and challenger_snapshot.trades == 0
    ):
        replacement_reasons.append(
            (
                "challenger sem trades apos "
                f"{evaluation_iterations} iteracoes "
                f"(limite={policy.max_zero_trade_iterations})"
            )
        )
    if (
        evaluation_iterations >= policy.evaluation_iterations
        and challenger_health.requires_retraining
        and challenger_snapshot.trades < policy.evaluation_min_trades
    ):
        replacement_reasons.append(
            "challenger requer retreinamento e nao atingiu minimo de trades para promocao"
        )
    if evaluation_iterations >= policy.max_evaluation_iterations and not eligible:
        replacement_reasons.append(
            (
                "challenger excedeu janela maxima de avaliacao "
                f"({policy.max_evaluation_iterations}) sem promocao"
            )
        )

    replacement_required = bool(replacement_reasons)

    return {
        "eligible": eligible,
        "rebase_eligible": rebase_eligible,
        "activity_rebase_eligible": activity_rebase_eligible,
        "replacement_required": replacement_required,
        "replacement_reasons": replacement_reasons,
        "evaluation_iterations": evaluation_iterations,
        "policy": {
            "evaluation_iterations": policy.evaluation_iterations,
            "evaluation_min_trades": policy.evaluation_min_trades,
            "challenger_wins_required": required_wins,
            "max_evaluation_iterations": policy.max_evaluation_iterations,
            "max_zero_trade_iterations": policy.max_zero_trade_iterations,
            "allow_rebase_promotion": policy.allow_rebase_promotion,
        },
        "comparisons": comparisons,
        "challenger_wins": challenger_wins,
        "champion_evaluation": champion_health.to_dict(),
        "challenger_evaluation": challenger_health.to_dict(),
    }

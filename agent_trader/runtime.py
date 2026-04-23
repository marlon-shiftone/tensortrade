from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .domain import ModelRecord, ModelStage, OrderIntent, OrderSide, RunMode
from .registry import ModelRegistry


class ModelExecutor(Protocol):
    def predict_order(
        self,
        features: Mapping[str, Any],
        model: ModelRecord,
        context: Mapping[str, Any] | None = None,
    ) -> OrderIntent:
        ...


class RiskManager(Protocol):
    def allow(
        self,
        intent: OrderIntent,
        mode: RunMode,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        ...


class BrokerClient(Protocol):
    def submit(self, intent: OrderIntent, mode: RunMode) -> str:
        ...


@dataclass
class StaticSignalModelExecutor:
    symbol: str
    side: OrderSide = OrderSide.HOLD
    notional_usd: float = 0.0
    confidence: float = 0.0
    reason: str = "static signal executor"

    def predict_order(
        self,
        features: Mapping[str, Any],
        model: ModelRecord,
        context: Mapping[str, Any] | None = None,
    ) -> OrderIntent:
        return OrderIntent(
            model_id=model.model_id,
            symbol=self.symbol,
            side=self.side,
            notional_usd=self.notional_usd,
            confidence=self.confidence,
            reason=self.reason,
        )


@dataclass
class AllowAllRiskManager:
    def allow(
        self,
        intent: OrderIntent,
        mode: RunMode,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        return True


@dataclass
class MemoryBroker:
    submissions: list[dict[str, Any]] = field(default_factory=list)

    def submit(self, intent: OrderIntent, mode: RunMode) -> str:
        reference = f"{mode.value}-{len(self.submissions) + 1}"
        self.submissions.append(
            {
                "reference": reference,
                "mode": mode.value,
                "intent": intent.to_dict(),
            }
        )
        return reference


class TradingRuntime:
    def __init__(
        self,
        registry: ModelRegistry,
        model_executor: ModelExecutor,
        risk_manager: RiskManager,
        broker: BrokerClient,
    ):
        self.registry = registry
        self.model_executor = model_executor
        self.risk_manager = risk_manager
        self.broker = broker

    def _ensure_mode_allowed(self, model: ModelRecord, mode: RunMode) -> None:
        if mode is RunMode.PAPER:
            return

        if not model.human_approved_for_live:
            raise PermissionError(
                f"modelo {model.model_id} nao possui aprovacao humana para operar em live"
            )

        if model.stage not in {ModelStage.LIVE_APPROVED, ModelStage.LIVE_ACTIVE}:
            raise PermissionError(
                f"modelo {model.model_id} nao esta em stage habilitado para live: {model.stage.value}"
            )

    def start(self, model_id: str, mode: RunMode) -> ModelRecord:
        model = self.registry.get_model(model_id)
        self._ensure_mode_allowed(model, mode)

        if mode is RunMode.PAPER:
            return self.registry.mark_paper_ready(model_id, notes="runtime paper iniciado")

        return self.registry.activate_live(model_id, notes="runtime live iniciado")

    def run_once(
        self,
        model_id: str,
        mode: RunMode,
        features: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        model = self.registry.get_model(model_id)
        self._ensure_mode_allowed(model, mode)
        intent = self.model_executor.predict_order(features, model, context=context)

        if intent.side == OrderSide.HOLD:
            hold_reason = intent.reason or "modelo optou por hold"
            return {
                "submitted": False,
                "reason": hold_reason,
                "intent": intent.to_dict(),
            }

        position_qty = context.get("position_qty") if context else None
        try:
            position_qty_value = float(position_qty) if position_qty is not None else None
        except (TypeError, ValueError):
            position_qty_value = None
        if intent.side == OrderSide.SELL and position_qty_value is not None and position_qty_value <= 0:
            return {
                "submitted": False,
                "reason": "sell bloqueado antes do broker: sem posicao aberta",
                "intent": intent.to_dict(),
            }

        if not self.risk_manager.allow(intent, mode, context=context):
            return {
                "submitted": False,
                "reason": "bloqueado pela camada de risco",
                "intent": intent.to_dict(),
            }

        try:
            reference = self.broker.submit(intent, mode)
        except Exception as exc:
            return {
                "submitted": False,
                "reason": "erro ao enviar ordem para o broker",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "intent": intent.to_dict(),
            }

        reference_text = str(reference)
        if "noop" in reference_text or "dry-run" in reference_text:
            return {
                "submitted": False,
                "reference": reference_text,
                "reason": reference_text,
                "intent": intent.to_dict(),
            }

        return {
            "submitted": True,
            "reference": reference_text,
            "intent": intent.to_dict(),
        }

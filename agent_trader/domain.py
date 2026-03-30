from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class ModelStage(str, Enum):
    CANDIDATE = "candidate"
    PAPER_READY = "paper_ready"
    LIVE_APPROVED = "live_approved"
    LIVE_ACTIVE = "live_active"
    RETIRED = "retired"


class SupervisorAction(str, Enum):
    KEEP_RUNNING = "keep_running"
    RETRAIN = "retrain"
    PROMOTE_TO_LIVE = "promote_to_live"
    ROLLBACK = "rollback"
    PAUSE = "pause"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class PerformanceSnapshot:
    symbol: str
    net_profit: float = 0.0
    rolling_sharpe: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    turnover: float = 0.0
    feature_drift: float = 0.0
    paper_days: int = 0
    trades: int = 0
    as_of: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "net_profit": self.net_profit,
            "rolling_sharpe": self.rolling_sharpe,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "turnover": self.turnover,
            "feature_drift": self.feature_drift,
            "paper_days": self.paper_days,
            "trades": self.trades,
            "as_of": self.as_of,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceSnapshot":
        return cls(
            symbol=data["symbol"],
            net_profit=data.get("net_profit", 0.0),
            rolling_sharpe=data.get("rolling_sharpe", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            hit_rate=data.get("hit_rate", 0.0),
            turnover=data.get("turnover", 0.0),
            feature_drift=data.get("feature_drift", 0.0),
            paper_days=data.get("paper_days", 0),
            trades=data.get("trades", 0),
            as_of=data.get("as_of", utc_now_iso()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ModelRecord:
    model_id: str
    artifact_path: str
    trainer: str = "tensortrade"
    stage: ModelStage = ModelStage.CANDIDATE
    created_at: str = field(default_factory=utc_now_iso)
    human_approved_for_live: bool = False
    human_approved_by: str | None = None
    human_approved_at: str | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "artifact_path": self.artifact_path,
            "trainer": self.trainer,
            "stage": self.stage.value,
            "created_at": self.created_at,
            "human_approved_for_live": self.human_approved_for_live,
            "human_approved_by": self.human_approved_by,
            "human_approved_at": self.human_approved_at,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRecord":
        return cls(
            model_id=data["model_id"],
            artifact_path=data["artifact_path"],
            trainer=data.get("trainer", "tensortrade"),
            stage=ModelStage(data.get("stage", ModelStage.CANDIDATE.value)),
            created_at=data.get("created_at", utc_now_iso()),
            human_approved_for_live=data.get("human_approved_for_live", False),
            human_approved_by=data.get("human_approved_by"),
            human_approved_at=data.get("human_approved_at"),
            notes=data.get("notes", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OrderIntent:
    model_id: str
    symbol: str
    side: OrderSide
    notional_usd: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    reference_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "notional_usd": self.notional_usd,
            "confidence": self.confidence,
            "reason": self.reason,
            "reference_price": self.reference_price,
        }


@dataclass
class SupervisorReport:
    action: SupervisorAction
    reasons: list[str]
    evaluation: dict[str, Any] = field(default_factory=dict)
    requires_human_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reasons": self.reasons,
            "evaluation": self.evaluation,
            "requires_human_review": self.requires_human_review,
        }

from __future__ import annotations

from dataclasses import dataclass

from .domain import PerformanceSnapshot


@dataclass
class HealthThresholds:
    min_paper_sharpe: float = 0.50
    min_live_sharpe: float = 1.00
    max_drawdown_retrain: float = 0.15
    max_drawdown_pause: float = 0.25
    min_hit_rate: float = 0.48
    max_feature_drift: float = 0.35
    min_paper_days_for_live: int = 10
    min_paper_trades_for_live: int = 50
    max_turnover: float = 10.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "min_paper_sharpe": self.min_paper_sharpe,
            "min_live_sharpe": self.min_live_sharpe,
            "max_drawdown_retrain": self.max_drawdown_retrain,
            "max_drawdown_pause": self.max_drawdown_pause,
            "min_hit_rate": self.min_hit_rate,
            "max_feature_drift": self.max_feature_drift,
            "min_paper_days_for_live": self.min_paper_days_for_live,
            "min_paper_trades_for_live": self.min_paper_trades_for_live,
            "max_turnover": self.max_turnover,
        }


@dataclass
class HealthEvaluation:
    healthy: bool
    allow_live_candidate: bool
    requires_retraining: bool
    pause_trading: bool
    reasons: list[str]
    thresholds: HealthThresholds

    def to_dict(self) -> dict[str, object]:
        return {
            "healthy": self.healthy,
            "allow_live_candidate": self.allow_live_candidate,
            "requires_retraining": self.requires_retraining,
            "pause_trading": self.pause_trading,
            "reasons": self.reasons,
            "thresholds": self.thresholds.to_dict(),
        }


def evaluate_snapshot(
    snapshot: PerformanceSnapshot,
    thresholds: HealthThresholds | None = None,
) -> HealthEvaluation:
    thresholds = thresholds or HealthThresholds()
    reasons = []
    requires_retraining = False
    pause_trading = False

    if snapshot.rolling_sharpe < thresholds.min_paper_sharpe:
        requires_retraining = True
        reasons.append(
            f"rolling_sharpe={snapshot.rolling_sharpe:.3f} abaixo do minimo de papel "
            f"{thresholds.min_paper_sharpe:.3f}"
        )

    if snapshot.max_drawdown > thresholds.max_drawdown_retrain:
        requires_retraining = True
        reasons.append(
            f"max_drawdown={snapshot.max_drawdown:.3f} acima do limite de retreino "
            f"{thresholds.max_drawdown_retrain:.3f}"
        )

    if snapshot.hit_rate < thresholds.min_hit_rate and snapshot.trades > 0:
        requires_retraining = True
        reasons.append(
            f"hit_rate={snapshot.hit_rate:.3f} abaixo do minimo {thresholds.min_hit_rate:.3f}"
        )

    if snapshot.feature_drift > thresholds.max_feature_drift:
        requires_retraining = True
        reasons.append(
            f"feature_drift={snapshot.feature_drift:.3f} acima do limite "
            f"{thresholds.max_feature_drift:.3f}"
        )

    if snapshot.turnover > thresholds.max_turnover:
        reasons.append(
            f"turnover={snapshot.turnover:.3f} acima do teto {thresholds.max_turnover:.3f}"
        )

    if snapshot.max_drawdown > thresholds.max_drawdown_pause:
        pause_trading = True
        reasons.append(
            f"max_drawdown={snapshot.max_drawdown:.3f} acima do limite de pausa "
            f"{thresholds.max_drawdown_pause:.3f}"
        )

    allow_live_candidate = all(
        [
            snapshot.paper_days >= thresholds.min_paper_days_for_live,
            snapshot.trades >= thresholds.min_paper_trades_for_live,
            snapshot.rolling_sharpe >= thresholds.min_live_sharpe,
            snapshot.max_drawdown <= thresholds.max_drawdown_retrain,
            snapshot.feature_drift <= thresholds.max_feature_drift,
        ]
    )

    healthy = not requires_retraining and not pause_trading

    if allow_live_candidate:
        reasons.append("modelo elegivel para avaliacao humana antes de live")

    return HealthEvaluation(
        healthy=healthy,
        allow_live_candidate=allow_live_candidate,
        requires_retraining=requires_retraining,
        pause_trading=pause_trading,
        reasons=reasons,
        thresholds=thresholds,
    )

from __future__ import annotations

from dataclasses import dataclass

from .domain import ModelRecord, PerformanceSnapshot, RunMode, SupervisorAction, SupervisorReport
from .metrics import HealthThresholds, evaluate_snapshot


@dataclass
class SupervisorState:
    consecutive_bad_windows: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"consecutive_bad_windows": self.consecutive_bad_windows}


@dataclass
class SupervisorPolicy:
    bad_windows_before_retrain: int = 3
    bad_windows_before_pause_live: int = 2


class Supervisor:
    def __init__(
        self,
        thresholds: HealthThresholds | None = None,
        policy: SupervisorPolicy | None = None,
    ):
        self.thresholds = thresholds or HealthThresholds()
        self.policy = policy or SupervisorPolicy()

    def review(
        self,
        model: ModelRecord,
        snapshot: PerformanceSnapshot,
        mode: RunMode,
        state: SupervisorState | None = None,
    ) -> tuple[SupervisorReport, SupervisorState]:
        state = state or SupervisorState()
        evaluation = evaluate_snapshot(snapshot, self.thresholds)
        reasons = list(evaluation.reasons)

        if evaluation.pause_trading:
            state.consecutive_bad_windows += 1
            return (
                SupervisorReport(
                    action=SupervisorAction.PAUSE,
                    reasons=reasons or ["pausa preventiva acionada"],
                    evaluation=evaluation.to_dict(),
                ),
                state,
            )

        if evaluation.requires_retraining:
            state.consecutive_bad_windows += 1

            if mode is RunMode.LIVE and state.consecutive_bad_windows >= self.policy.bad_windows_before_pause_live:
                return (
                    SupervisorReport(
                        action=SupervisorAction.PAUSE,
                        reasons=reasons + ["modelo live deteriorando em janelas consecutivas"],
                        evaluation=evaluation.to_dict(),
                    ),
                    state,
                )

            if state.consecutive_bad_windows >= self.policy.bad_windows_before_retrain:
                return (
                    SupervisorReport(
                        action=SupervisorAction.RETRAIN,
                        reasons=reasons + ["retreino recomendado pelo supervisor"],
                        evaluation=evaluation.to_dict(),
                    ),
                    state,
                )

            return (
                SupervisorReport(
                    action=SupervisorAction.KEEP_RUNNING,
                    reasons=reasons + ["watch mode: aguardando confirmacao de degradacao"],
                    evaluation=evaluation.to_dict(),
                ),
                state,
            )

        state.consecutive_bad_windows = 0

        if mode is RunMode.PAPER and evaluation.allow_live_candidate:
            if model.human_approved_for_live:
                return (
                    SupervisorReport(
                        action=SupervisorAction.PROMOTE_TO_LIVE,
                        reasons=reasons + ["modelo elegivel e com aprovacao humana para live"],
                        evaluation=evaluation.to_dict(),
                    ),
                    state,
                )

            return (
                SupervisorReport(
                    action=SupervisorAction.KEEP_RUNNING,
                    reasons=reasons + ["modelo elegivel para live, aguardando autorizacao humana"],
                    evaluation=evaluation.to_dict(),
                    requires_human_review=True,
                ),
                state,
            )

        return (
            SupervisorReport(
                action=SupervisorAction.KEEP_RUNNING,
                reasons=reasons or ["modelo saudavel nas metricas atuais"],
                evaluation=evaluation.to_dict(),
            ),
            state,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .domain import PerformanceSnapshot, RunMode, SupervisorAction
from .registry import ModelRegistry
from .storage import PerformanceSnapshotStore, SupervisorReportStore, SupervisorStateStore
from .supervisor import Supervisor


RetrainCallback = Callable[[PerformanceSnapshot], dict]


@dataclass
class SupervisorLoopResult:
    report_record: dict
    state: dict
    retrain_result: dict | None = None

    def to_dict(self) -> dict:
        return {
            "report_record": self.report_record,
            "state": self.state,
            "retrain_result": self.retrain_result,
        }


class SupervisorLoop:
    def __init__(
        self,
        project: str,
        registry: ModelRegistry,
        snapshot_store: PerformanceSnapshotStore,
        report_store: SupervisorReportStore,
        state_store: SupervisorStateStore,
        supervisor: Supervisor,
    ):
        self.project = project
        self.registry = registry
        self.snapshot_store = snapshot_store
        self.report_store = report_store
        self.state_store = state_store
        self.supervisor = supervisor

    def run_once(
        self,
        model_id: str,
        mode: RunMode,
        retrain_callback: RetrainCallback | None = None,
    ) -> SupervisorLoopResult:
        snapshot = self.snapshot_store.latest_snapshot(model_id=model_id, mode=mode)
        if snapshot is None:
            raise RuntimeError(
                f"nenhum snapshot persistido para model_id={model_id} mode={mode.value}"
            )

        model = self.registry.get_model(model_id)
        state = self.state_store.load(model_id=model_id, mode=mode)
        report, next_state = self.supervisor.review(
            model=model,
            snapshot=snapshot,
            mode=mode,
            state=state,
        )
        saved_state = self.state_store.save(model_id=model_id, mode=mode, state=next_state)
        report_record = self.report_store.append_report(
            project=self.project,
            model_id=model_id,
            mode=mode,
            report=report,
            snapshot=snapshot,
            extra={"supervisor_state": saved_state},
        )

        retrain_result = None
        if report.action is SupervisorAction.RETRAIN and retrain_callback:
            retrain_result = retrain_callback(snapshot)

        return SupervisorLoopResult(
            report_record=report_record,
            state=saved_state,
            retrain_result=retrain_result,
        )

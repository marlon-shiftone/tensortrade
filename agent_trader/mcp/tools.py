from __future__ import annotations

from ..domain import PerformanceSnapshot, RunMode
from ..registry import ModelRegistry
from ..supervisor import Supervisor


class SupervisorTools:
    def __init__(self, registry: ModelRegistry, supervisor: Supervisor):
        self.registry = registry
        self.supervisor = supervisor

    def register_candidate(
        self,
        model_id: str,
        artifact_path: str,
        trainer: str = "tensortrade",
        notes: str = "",
    ) -> dict[str, object]:
        return self.registry.register_candidate(
            model_id=model_id,
            artifact_path=artifact_path,
            trainer=trainer,
            notes=notes,
        ).to_dict()

    def approve_for_live(
        self,
        model_id: str,
        approved_by: str,
        notes: str = "",
    ) -> dict[str, object]:
        return self.registry.approve_for_live(
            model_id=model_id,
            approved_by=approved_by,
            notes=notes,
        ).to_dict()

    def activate_live(self, model_id: str, notes: str = "") -> dict[str, object]:
        return self.registry.activate_live(model_id, notes=notes).to_dict()

    def review_model(
        self,
        model_id: str,
        mode: RunMode,
        snapshot: PerformanceSnapshot,
    ) -> dict[str, object]:
        model = self.registry.get_model(model_id)
        report, state = self.supervisor.review(model=model, snapshot=snapshot, mode=mode)
        return {
            "model": model.to_dict(),
            "snapshot": snapshot.to_dict(),
            "report": report.to_dict(),
            "state": state.to_dict(),
        }

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain import ModelRecord, ModelStage, utc_now_iso


class ModelRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "models": {},
            "paper_model_id": None,
            "live_model_id": None,
            "history": [],
        }

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

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

    def _append_event(self, state: dict[str, Any], action: str, model_id: str, notes: str = "") -> None:
        state["history"].append(
            {
                "at": utc_now_iso(),
                "action": action,
                "model_id": model_id,
                "notes": notes,
            }
        )

    def state(self) -> dict[str, Any]:
        return self.load()

    def list_models(self) -> list[ModelRecord]:
        state = self.load()
        return [ModelRecord.from_dict(payload) for payload in state["models"].values()]

    def get_model(self, model_id: str) -> ModelRecord:
        state = self.load()
        payload = state["models"].get(model_id)
        if not payload:
            raise KeyError(f"modelo nao encontrado: {model_id}")
        return ModelRecord.from_dict(payload)

    def register_candidate(
        self,
        model_id: str,
        artifact_path: str,
        trainer: str = "tensortrade",
        metadata: dict[str, Any] | None = None,
        notes: str = "",
    ) -> ModelRecord:
        state = self.load()
        if model_id in state["models"]:
            raise ValueError(f"modelo ja registrado: {model_id}")

        record = ModelRecord(
            model_id=model_id,
            artifact_path=artifact_path,
            trainer=trainer,
            stage=ModelStage.CANDIDATE,
            metadata=metadata or {},
            notes=notes,
        )
        state["models"][model_id] = record.to_dict()
        self._append_event(state, "register_candidate", model_id, notes)
        self.save(state)
        return record

    def mark_paper_ready(self, model_id: str, notes: str = "") -> ModelRecord:
        state = self.load()
        record = self.get_model(model_id)
        record.stage = ModelStage.PAPER_READY
        if notes:
            record.notes = notes
        state["models"][model_id] = record.to_dict()
        state["paper_model_id"] = model_id
        self._append_event(state, "mark_paper_ready", model_id, notes)
        self.save(state)
        return record

    def approve_for_live(self, model_id: str, approved_by: str, notes: str = "") -> ModelRecord:
        state = self.load()
        record = self.get_model(model_id)
        record.stage = ModelStage.LIVE_APPROVED
        record.human_approved_for_live = True
        record.human_approved_by = approved_by
        record.human_approved_at = utc_now_iso()
        if notes:
            record.notes = notes
        state["models"][model_id] = record.to_dict()
        self._append_event(state, "approve_for_live", model_id, notes)
        self.save(state)
        return record

    def activate_live(self, model_id: str, notes: str = "") -> ModelRecord:
        state = self.load()
        record = self.get_model(model_id)

        if not record.human_approved_for_live:
            raise PermissionError(
                f"modelo {model_id} nao pode entrar em live sem aprovacao humana"
            )

        previous_live_model_id = state.get("live_model_id")
        if previous_live_model_id and previous_live_model_id != model_id:
            previous = self.get_model(previous_live_model_id)
            previous.stage = ModelStage.PAPER_READY
            state["models"][previous_live_model_id] = previous.to_dict()

        record.stage = ModelStage.LIVE_ACTIVE
        if notes:
            record.notes = notes
        state["models"][model_id] = record.to_dict()
        state["live_model_id"] = model_id
        self._append_event(state, "activate_live", model_id, notes)
        self.save(state)
        return record

    def retire_model(self, model_id: str, notes: str = "") -> ModelRecord:
        state = self.load()
        record = self.get_model(model_id)
        record.stage = ModelStage.RETIRED
        if notes:
            record.notes = notes
        state["models"][model_id] = record.to_dict()
        if state.get("paper_model_id") == model_id:
            state["paper_model_id"] = None
        if state.get("live_model_id") == model_id:
            state["live_model_id"] = None
        self._append_event(state, "retire_model", model_id, notes)
        self.save(state)
        return record

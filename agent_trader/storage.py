from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .json_utils import json_default
from .domain import PerformanceSnapshot, RunMode, SupervisorReport, utc_now_iso
from .supervisor import SupervisorState


class JsonlStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        self._ensure_parent()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=json_default))
            handle.write("\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records


class PerformanceSnapshotStore:
    def __init__(self, path: str | Path):
        self.store = JsonlStore(path)

    def append_snapshot(
        self,
        project: str,
        model_id: str,
        mode: RunMode,
        snapshot: PerformanceSnapshot,
        source: str = "manual",
        notes: str = "",
    ) -> dict[str, Any]:
        record = {
            "recorded_at": utc_now_iso(),
            "project": project,
            "model_id": model_id,
            "mode": mode.value,
            "source": source,
            "notes": notes,
            "snapshot": snapshot.to_dict(),
        }
        self.store.append(record)
        return record

    def recent_records(
        self,
        model_id: str | None = None,
        mode: RunMode | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        records = self.store.read_all()
        if model_id:
            records = [record for record in records if record.get("model_id") == model_id]
        if mode:
            records = [record for record in records if record.get("mode") == mode.value]
        if limit is not None:
            records = records[-limit:]
        return records

    def latest_snapshot(
        self,
        model_id: str,
        mode: RunMode,
    ) -> PerformanceSnapshot | None:
        records = self.recent_records(model_id=model_id, mode=mode, limit=1)
        if not records:
            return None
        return PerformanceSnapshot.from_dict(records[-1]["snapshot"])


class ExecutionResultStore:
    def __init__(self, path: str | Path):
        self.store = JsonlStore(path)

    def append_result(
        self,
        project: str,
        model_id: str,
        mode: RunMode,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "recorded_at": utc_now_iso(),
            "project": project,
            "model_id": model_id,
            "mode": mode.value,
            "payload": payload,
        }
        self.store.append(record)
        return record


class TrainingRunStore:
    def __init__(self, path: str | Path):
        self.store = JsonlStore(path)

    def append_run(
        self,
        project: str,
        model_id: str,
        trainer: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "recorded_at": utc_now_iso(),
            "project": project,
            "model_id": model_id,
            "trainer": trainer,
            "payload": payload,
        }
        self.store.append(record)
        return record


class SupervisorReportStore:
    def __init__(self, path: str | Path):
        self.store = JsonlStore(path)

    def append_report(
        self,
        project: str,
        model_id: str,
        mode: RunMode,
        report: SupervisorReport,
        snapshot: PerformanceSnapshot,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "recorded_at": utc_now_iso(),
            "project": project,
            "model_id": model_id,
            "mode": mode.value,
            "snapshot": snapshot.to_dict(),
            "report": report.to_dict(),
            "extra": extra or {},
        }
        self.store.append(record)
        return record


class SupervisorStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_all(self) -> dict[str, Any]:
        self._ensure_parent()
        if not self.path.exists():
            return {"paper": {}, "live": {}}

        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load(self, model_id: str, mode: RunMode) -> SupervisorState:
        payload = self._load_all()
        state_payload = payload.get(mode.value, {}).get(model_id, {})
        return SupervisorState(
            consecutive_bad_windows=state_payload.get("consecutive_bad_windows", 0)
        )

    def save(self, model_id: str, mode: RunMode, state: SupervisorState) -> dict[str, Any]:
        payload = self._load_all()
        payload.setdefault(mode.value, {})
        payload[mode.value][model_id] = state.to_dict()

        self._ensure_parent()
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return payload[mode.value][model_id]

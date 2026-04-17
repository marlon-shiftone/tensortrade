from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .env import load_default_env_files
from .metrics import HealthThresholds


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
    root_dir: Path = ROOT_DIR
    project_name: Optional[str] = None
    project_dir: Optional[Path] = None
    state_dir: Optional[Path] = None
    artifacts_dir: Optional[Path] = None
    reports_dir: Optional[Path] = None
    configs_dir: Optional[Path] = None
    registry_path: Optional[Path] = None
    metrics_history_path: Optional[Path] = None
    execution_history_path: Optional[Path] = None
    supervisor_reports_path: Optional[Path] = None
    supervisor_state_path: Optional[Path] = None
    training_history_path: Optional[Path] = None
    default_trainer: str = os.getenv("AGENT_TRADER_DEFAULT_TRAINER", "tensortrade")
    thresholds: HealthThresholds = field(default_factory=HealthThresholds)

    def __post_init__(self) -> None:
        if self.project_name is None:
            self.project_name = os.getenv("AGENT_TRADER_PROJECT", "alpaca")

        if self.project_dir is None:
            project_dir_env = os.getenv("AGENT_TRADER_PROJECT_DIR")
            self.project_dir = Path(project_dir_env) if project_dir_env else self.root_dir / self.project_name

        load_default_env_files(self.root_dir, self.project_dir)

        if self.state_dir is None:
            self.state_dir = self.project_dir / "state"
        if self.artifacts_dir is None:
            self.artifacts_dir = self.project_dir / "artifacts"
        if self.reports_dir is None:
            self.reports_dir = self.project_dir / "reports"
        if self.configs_dir is None:
            self.configs_dir = self.project_dir / "configs"
        if self.registry_path is None:
            self.registry_path = self.state_dir / "model_registry.json"
        if self.metrics_history_path is None:
            self.metrics_history_path = self.reports_dir / "performance_snapshots.jsonl"
        if self.execution_history_path is None:
            self.execution_history_path = self.reports_dir / "execution_results.jsonl"
        if self.supervisor_reports_path is None:
            self.supervisor_reports_path = self.reports_dir / "supervisor_reports.jsonl"
        if self.supervisor_state_path is None:
            self.supervisor_state_path = self.state_dir / "supervisor_state.json"
        if self.training_history_path is None:
            self.training_history_path = self.reports_dir / "training_runs.jsonl"

    def ensure_state_dir(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

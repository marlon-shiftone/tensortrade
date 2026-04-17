from __future__ import annotations

import os
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export "):].strip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if value and value[0] in {"'", '"'} and value[-1] == value[0]:
        value = value[1:-1]

    return key, value


def load_env_file(path: str | Path, override: bool = False) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_env_line(line)
            if parsed is None:
                continue

            key, value = parsed
            if override or key not in os.environ:
                os.environ[key] = value
                loaded[key] = value

    return loaded


def load_default_env_files(root_dir: str | Path, project_dir: str | Path | None = None) -> list[Path]:
    root = Path(root_dir)
    candidates = [root / ".env"]
    if project_dir is not None:
        candidates.append(Path(project_dir) / "configs" / ".env")

    loaded_paths: list[Path] = []
    for candidate in candidates:
        loaded = load_env_file(candidate, override=False)
        if loaded:
            loaded_paths.append(candidate)
    return loaded_paths

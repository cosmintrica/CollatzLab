from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_env_file(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = _strip_wrapping_quotes(value.strip())
        os.environ[key] = value


def write_env_updates(root: Path, updates: dict[str, str]) -> Path:
    env_path = root / ".env"
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    remaining = {key: str(value) for key, value in updates.items()}
    rewritten: list[str] = []

    for raw_line in existing_lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            rewritten.append(raw_line)
            continue
        key, _value = raw_line.split("=", 1)
        normalized_key = key.strip()
        if normalized_key in remaining:
            rewritten.append(f"{normalized_key}={remaining.pop(normalized_key)}")
        else:
            rewritten.append(raw_line)

    for key, value in remaining.items():
        rewritten.append(f"{key}={value}")

    env_path.write_text("\n".join(rewritten).rstrip() + "\n", encoding="utf-8")
    for key, value in updates.items():
        os.environ[key] = str(value)
    return env_path


@dataclass(frozen=True)
class Settings:
    workspace_root: Path
    data_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    research_dir: Path
    db_path: Path
    api_host: str
    api_port: int
    llm_autopilot_enabled: bool
    llm_autopilot_interval_seconds: int
    llm_autopilot_max_tasks: int

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(os.getenv("COLLATZ_LAB_ROOT", Path.cwd())).resolve()
        _load_env_file(root)
        root = Path(os.getenv("COLLATZ_LAB_ROOT", root)).resolve()
        data_dir = root / "data"
        artifacts_dir = root / "artifacts"
        reports_dir = root / "reports"
        research_dir = root / "research"
        return cls(
            workspace_root=root,
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
            reports_dir=reports_dir,
            research_dir=research_dir,
            db_path=data_dir / "lab.db",
            api_host=os.getenv("COLLATZ_LAB_HOST", "127.0.0.1"),
            api_port=int(os.getenv("COLLATZ_LAB_PORT", "8000")),
            llm_autopilot_enabled=os.getenv("COLLATZ_LLM_AUTOPILOT_ENABLED", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            llm_autopilot_interval_seconds=max(
                60,
                int(os.getenv("COLLATZ_LLM_AUTOPILOT_INTERVAL_SECONDS", "7200")),
            ),
            llm_autopilot_max_tasks=max(
                1,
                min(5, int(os.getenv("COLLATZ_LLM_AUTOPILOT_MAX_TASKS", "2"))),
            ),
        )

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.artifacts_dir,
            self.artifacts_dir / "runs",
            self.artifacts_dir / "tasks",
            self.artifacts_dir / "reviews",
            self.artifacts_dir / "validations",
            self.reports_dir,
            self.research_dir,
            self.research_dir / "claims",
            self.research_dir / "directions",
            self.research_dir / "sources",
        ):
            path.mkdir(parents=True, exist_ok=True)

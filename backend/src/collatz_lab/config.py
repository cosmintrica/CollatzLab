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
        )

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.artifacts_dir,
            self.artifacts_dir / "runs",
            self.artifacts_dir / "reviews",
            self.artifacts_dir / "validations",
            self.reports_dir,
            self.research_dir,
            self.research_dir / "claims",
            self.research_dir / "directions",
            self.research_dir / "sources",
        ):
            path.mkdir(parents=True, exist_ok=True)

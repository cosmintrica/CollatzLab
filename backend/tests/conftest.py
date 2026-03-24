from __future__ import annotations

from pathlib import Path

import pytest

from collatz_lab.runtime_bootstrap import ensure_darwin_duplicate_openmp_ok

ensure_darwin_duplicate_openmp_ok()

from collatz_lab.config import Settings
from collatz_lab.repository import LabRepository


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("COLLATZ_LAB_ROOT", str(tmp_path))
    return Settings.from_env()


@pytest.fixture()
def repository(settings: Settings) -> LabRepository:
    repo = LabRepository(settings)
    repo.init()
    return repo

from pathlib import Path

from collatz_lab.config import Settings, write_env_updates


def test_settings_loads_dotenv(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text(
        "\n".join(
            [
                "COLLATZ_LAB_PORT=8123",
                "GEMINI_API_KEY=test-key",
                "COLLATZ_LLM_ENABLED=1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("COLLATZ_LAB_ROOT", str(workspace))
    monkeypatch.delenv("COLLATZ_LAB_PORT", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("COLLATZ_LLM_ENABLED", raising=False)
    monkeypatch.delenv("COLLATZ_LLM_AUTOPILOT_ENABLED", raising=False)
    monkeypatch.delenv("COLLATZ_LLM_AUTOPILOT_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("COLLATZ_LLM_AUTOPILOT_MAX_TASKS", raising=False)

    settings = Settings.from_env()

    assert settings.workspace_root == workspace.resolve()
    assert settings.api_port == 8123
    assert settings.llm_autopilot_enabled is False
    assert settings.llm_autopilot_interval_seconds == 7200
    assert settings.llm_autopilot_max_tasks == 2


def test_write_env_updates_persists_and_updates_process(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env_path = workspace / ".env"
    env_path.write_text("COLLATZ_LLM_ENABLED=0\n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    written = write_env_updates(
        workspace,
        {
            "COLLATZ_LLM_ENABLED": "1",
            "GEMINI_MODEL": "gemini-2.5-flash",
            "GEMINI_API_KEY": "secret-key",
        },
    )

    text = written.read_text(encoding="utf-8")
    assert "COLLATZ_LLM_ENABLED=1" in text
    assert "GEMINI_MODEL=gemini-2.5-flash" in text
    assert "GEMINI_API_KEY=secret-key" in text
    assert written == env_path

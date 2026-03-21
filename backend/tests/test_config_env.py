from pathlib import Path

from collatz_lab.config import Settings


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

    settings = Settings.from_env()

    assert settings.workspace_root == workspace.resolve()
    assert settings.api_port == 8123

"""``gpu_sieve_metal_runtime`` stays inert off macOS and respects ``COLLATZ_GPU_SIEVE_BACKEND``."""

from __future__ import annotations

from unittest import mock

import pytest


def test_native_metal_unavailable_off_darwin():
    with mock.patch("platform.system", return_value="Linux"):
        from collatz_lab import gpu_sieve_metal_runtime as m

        m.metal_sieve_chunk_binary_path.cache_clear()
        assert m.native_metal_sieve_available() is False
        assert m.should_use_native_metal_sieve() is False


def test_forced_metal_without_binary_raises(monkeypatch, tmp_path):
    with mock.patch("platform.system", return_value="Darwin"):
        from collatz_lab import gpu_sieve_metal_runtime as m

        m.metal_sieve_chunk_binary_path.cache_clear()
        monkeypatch.setenv("COLLATZ_GPU_SIEVE_BACKEND", "metal")
        monkeypatch.delenv("COLLATZ_METAL_SIEVE_BINARY", raising=False)
        monkeypatch.setattr(
            m,
            "_default_metal_chunk_binary_paths",
            lambda: [tmp_path / "no_metal_sieve_chunk_here"],
        )
        m.metal_sieve_chunk_binary_path.cache_clear()
        with pytest.raises(ValueError, match="metal_sieve_chunk"):
            m.should_use_native_metal_sieve()


def test_mps_mode_never_metal(monkeypatch):
    with mock.patch("platform.system", return_value="Darwin"):
        from collatz_lab import gpu_sieve_metal_runtime as m

        monkeypatch.setenv("COLLATZ_GPU_SIEVE_BACKEND", "mps")
        m.metal_sieve_chunk_binary_path.cache_clear()
        assert m.should_use_native_metal_sieve() is False


def test_metal_chunk_size_respects_cap(monkeypatch):
    from collatz_lab import gpu_sieve_metal_runtime as m

    monkeypatch.setenv("COLLATZ_METAL_SIEVE_CHUNK_MAX", "8388608")
    monkeypatch.setenv("COLLATZ_METAL_SIEVE_CHUNK_SIZE", "16777216")
    assert m.metal_sieve_chunk_max_odds() == 8_388_608
    assert m._metal_sieve_chunk_size() == 8_388_608
    monkeypatch.delenv("COLLATZ_METAL_SIEVE_CHUNK_SIZE", raising=False)
    monkeypatch.setenv("COLLATZ_METAL_SIEVE_CHUNK_AUTO", "0")
    assert m._metal_sieve_chunk_size() == min(4_194_304, 8_388_608)


def test_metal_auto_chunk_ladder(monkeypatch):
    from collatz_lab import gpu_sieve_metal_runtime as m

    with mock.patch("platform.system", return_value="Darwin"):
        monkeypatch.delenv("COLLATZ_METAL_SIEVE_CHUNK_SIZE", raising=False)
        monkeypatch.setenv("COLLATZ_METAL_SIEVE_CHUNK_AUTO", "1")
        monkeypatch.setenv("COLLATZ_METAL_SIEVE_USE_CALIBRATION", "0")
        monkeypatch.setenv("COLLATZ_METAL_SIEVE_CHUNK_MAX", "16777216")
        monkeypatch.setattr(m, "_estimate_bytes_available_for_metal_steps_buffer", lambda: 32 << 30)
        assert m._metal_sieve_chunk_size() == 16_777_216
        # 64 MiB → max_odds ≈ 1.4M → ladder picks 1 048 576
        monkeypatch.setattr(m, "_estimate_bytes_available_for_metal_steps_buffer", lambda: 64 << 20)
        assert m._metal_sieve_chunk_size() == 1_048_576


def test_metal_auto_disabled_falls_back_four_million(monkeypatch):
    from collatz_lab import gpu_sieve_metal_runtime as m

    with mock.patch("platform.system", return_value="Darwin"):
        monkeypatch.delenv("COLLATZ_METAL_SIEVE_CHUNK_SIZE", raising=False)
        monkeypatch.setenv("COLLATZ_METAL_SIEVE_CHUNK_AUTO", "0")
        monkeypatch.setenv("COLLATZ_METAL_SIEVE_CHUNK_MAX", "16777216")
        assert m._metal_sieve_chunk_size() == 4_194_304

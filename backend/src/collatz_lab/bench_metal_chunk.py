"""
Metal ``gpu-sieve`` chunk sweep for throughput tuning (macOS + ``metal_sieve_chunk``).

Used by ``scripts/profile_metal_sieve_chunk.py`` and the dashboard API
(``POST /api/bench/metal-chunk/run``).
"""

from __future__ import annotations

import json
import os
import platform
import threading
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .repository import LabRepository


def _odd_count(start: int, end: int) -> int:
    first_odd = start if start & 1 else start + 1
    if first_odd > end:
        return 0
    return ((end - first_odd) // 2) + 1


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_metal_chunk_benchmark(
    *,
    quick: bool = True,
    linear_end: int = 0,
    reps: int = 3,
    warmup: int = 1,
    chunks_csv: str = "",
    write_calibration: bool = False,
    pipeline_ab: bool = False,
    preset: str = "",
) -> dict[str, Any]:
    """Run sweep; raises ``RuntimeError`` if Metal is unavailable or not Darwin."""
    if platform.system() != "Darwin":
        raise RuntimeError("Metal chunk benchmark requires macOS (Darwin).")

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["COLLATZ_GPU_SIEVE_BACKEND"] = "metal"
    for k in ("COLLATZ_MPS_SIEVE_BATCH_SIZE", "COLLATZ_MPS_SYNC_EVERY"):
        os.environ.pop(k, None)

    from .gpu_sieve_metal_runtime import (
        compute_range_metrics_gpu_sieve_metal,
        metal_sieve_chunk_binary_path,
        metal_sieve_chunk_max_odds,
        native_metal_sieve_available,
        should_use_native_metal_sieve,
        shutdown_metal_stdio_transport,
    )
    from .hardware import GPU_SIEVE_KERNEL
    from .services import compute_range_metrics

    if not native_metal_sieve_available():
        raise RuntimeError(
            "metal_sieve_chunk is not available. Build: "
            "bash scripts/native_sieve_kit/metal/build_metal_sieve_chunk.sh"
        )
    try:
        should_use_native_metal_sieve()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    if linear_end > 0:
        end = linear_end
    else:
        end = 12_000_000 if quick else 48_000_000
    start = 1
    odds = _odd_count(start, end)

    if chunks_csv.strip():
        chunk_grid = [int(x.strip()) for x in chunks_csv.split(",") if x.strip()]
    else:
        chunk_grid = [1_048_576, 2_097_152, 4_194_304, 8_388_608, 16_777_216]

    max_allowed = metal_sieve_chunk_max_odds()
    chunk_grid = sorted({c for c in chunk_grid if 4096 <= c <= max_allowed})
    if not chunk_grid:
        raise RuntimeError(f"No valid chunk sizes in grid (max_allowed={max_allowed}).")

    try:
        ref = compute_range_metrics(1, 8000, kernel="cpu-sieve")
        metal_chk = compute_range_metrics(1, 8000, kernel=GPU_SIEVE_KERNEL)
        parity_ok = (
            metal_chk.max_total_stopping_time == ref.max_total_stopping_time
            and metal_chk.processed == ref.processed
        )

        rows: list[dict[str, Any]] = []

        import time as time_mod

        def time_once_perf(chunk_sz: int) -> float:
            os.environ["COLLATZ_METAL_SIEVE_CHUNK_SIZE"] = str(chunk_sz)
            t0 = time_mod.perf_counter()
            compute_range_metrics_gpu_sieve_metal(start, end)
            return time_mod.perf_counter() - t0

        for c in chunk_grid:
            for _ in range(warmup):
                time_once_perf(c)
            secs = [time_once_perf(c) for _ in range(reps)]
            med = sorted(secs)[len(secs) // 2]
            m_per_s = odds / med if med > 0 else 0.0
            rows.append(
                {
                    "metal_chunk_size": c,
                    "seconds_median": round(med, 6),
                    "odd_per_sec": round(m_per_s, 1),
                    "odd_per_sec_millions": round(m_per_s / 1e6, 3),
                    "reps_seconds": [round(s, 6) for s in secs],
                }
            )

        best = max(rows, key=lambda r: r["odd_per_sec"])

        pipeline_rows: list[dict[str, Any]] = []
        if pipeline_ab:
            bc = int(best["metal_chunk_size"])
            for pl in ("0", "1"):
                shutdown_metal_stdio_transport()
                os.environ["COLLATZ_METAL_SIEVE_STDIO_PIPELINE"] = pl
                os.environ["COLLATZ_METAL_SIEVE_CHUNK_SIZE"] = str(bc)
                for _ in range(warmup):
                    time_once_perf(bc)
                secs = [time_once_perf(bc) for _ in range(reps)]
                med = sorted(secs)[len(secs) // 2]
                m_per_s = odds / med if med > 0 else 0.0
                pipeline_rows.append(
                    {
                        "stdio_pipeline": pl == "1",
                        "seconds_median": round(med, 6),
                        "odd_per_sec_millions": round(m_per_s / 1e6, 3),
                    }
                )
            os.environ.pop("COLLATZ_METAL_SIEVE_STDIO_PIPELINE", None)

        cal_path_str: str | None = None
        if write_calibration:
            from .metal_chunk_calibration import write_metal_chunk_calibration

            cal_path = write_metal_chunk_calibration(
                chunk_odds=int(best["metal_chunk_size"]),
                interval={"start": start, "end": end, "odd_seeds": odds},
                winner_row=dict(best),
                platform=platform.system(),
            )
            cal_path_str = str(cal_path)

        helper = metal_sieve_chunk_binary_path()
        return {
            "benchmark_preset": (preset or "").strip() or None,
            "platform": platform.system(),
            "helper_path": str(helper) if helper else None,
            "interval": {"start": start, "end": end, "odd_seeds": odds},
            "runtime_max_chunk": max_allowed,
            "small_range_parity_ok": parity_ok,
            "winner": best,
            "runs": rows,
            "pipeline_ab": pipeline_rows or None,
            "calibration_written": cal_path_str,
        }
    finally:
        # Long-lived stdio helper keeps a grow-only steps buffer (~4 bytes × largest chunk); kill it so RAM returns to the OS.
        try:
            shutdown_metal_stdio_transport()
        except Exception:
            pass


# --- Single background job (API + UI) -----------------------------------------

_active_lock = threading.Lock()
_active: dict[str, Any] | None = None


def metal_benchmark_public_status() -> dict[str, Any]:
    from .gpu_sieve_metal_runtime import native_metal_sieve_available

    with _active_lock:
        job = dict(_active) if _active else None
    sys = platform.system()
    return {
        "darwin": sys == "Darwin",
        # Where the Python API runs (Docker/Linux will show non-Darwin even if the browser is on a Mac).
        "server_system": sys,
        "metal_available": native_metal_sieve_available() if sys == "Darwin" else False,
        "active_job": job,
    }


def metal_benchmark_try_start(repository: LabRepository, params: dict[str, Any]) -> dict[str, Any]:
    """Start background benchmark if idle. Returns ``{started, job_id?, message?}``."""
    global _active
    if platform.system() != "Darwin":
        return {
            "started": False,
            "error_code": "wrong_platform",
            "message": "Metal benchmark runs only on macOS.",
        }

    with _active_lock:
        if _active is not None:
            return {
                "started": False,
                "error_code": "busy",
                "message": "A benchmark is already running.",
                "job_id": _active.get("job_id"),
            }
        job_id = str(uuid.uuid4())
        started_at = _iso_now()
        _active = {"job_id": job_id, "started_at": started_at}

    thread = threading.Thread(
        target=_metal_benchmark_worker,
        args=(repository, job_id, started_at, params),
        name="collatz-metal-bench",
        daemon=True,
    )
    thread.start()
    return {"started": True, "job_id": job_id, "started_at": started_at}


def _metal_benchmark_worker(
    repository: LabRepository,
    job_id: str,
    started_at: str,
    params: dict[str, Any],
) -> None:
    global _active
    params_json = json.dumps(params, separators=(",", ":"))
    err_msg = ""
    result: dict[str, Any] | None = None
    try:
        result = run_metal_chunk_benchmark(
            quick=bool(params.get("quick", True)),
            linear_end=int(params.get("linear_end") or 0),
            reps=max(1, min(12, int(params.get("reps", 3)))),
            warmup=max(0, min(5, int(params.get("warmup", 1)))),
            chunks_csv=str(params.get("chunks_csv") or ""),
            write_calibration=bool(params.get("write_calibration", False)),
            pipeline_ab=bool(params.get("pipeline_ab", False)),
            preset=str(params.get("preset") or ""),
        )
    except Exception as exc:
        err_msg = str(exc)
    finished_at = _iso_now()
    try:
        repository.save_metal_benchmark_run(
            job_id=job_id,
            created_at=started_at,
            finished_at=finished_at,
            status="completed" if result is not None else "failed",
            params_json=params_json,
            result_json=json.dumps(result, indent=2) if result is not None else None,
            error_message=err_msg,
        )
    except Exception:
        pass
    finally:
        with _active_lock:
            _active = None

"""
Native **Metal** implementation of ``gpu-sieve`` (odd-only descent) on **macOS only**.

- **Modular:** imported only from ``compute_range_metrics_gpu_sieve`` on Darwin when selected;
  no Metal / Swift symbols at import time on Linux or Windows.
- **Helper binary:** ``metal_sieve_chunk`` (see ``scripts/native_sieve_kit/metal/``).
- **Transport:** by default a **long-lived ``--stdio`` process** (one JSON line in / one out per chunk)
  when the binary advertises ``"stdio": true`` on ``--ping``; otherwise one subprocess per chunk.
- **Selection:** ``COLLATZ_GPU_SIEVE_BACKEND`` = ``auto`` (default), ``mps``, or ``metal``.

``auto`` uses Metal when the helper exists and responds to ``--ping``; otherwise PyTorch MPS.
If Metal fails at runtime with ``auto``, falls back to MPS. With ``metal`` forced, errors propagate.

**Optional headless Mac:** ``COLLATZ_GPU_SIEVE_METAL_WITHOUT_TORCH=1`` allows ``gpu-sieve`` via native
Metal when PyTorch MPS is not installed (see ``collect_gpu_capabilities`` + ``compute_range_metrics_gpu_sieve``).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import platform
import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_stdio_lock = threading.Lock()
_stdio_proc: subprocess.Popen | None = None
_stdio_bin: str | None = None


def _workspace_root() -> Path:
    return Path(os.getenv("COLLATZ_LAB_ROOT", Path.cwd())).resolve()


def _default_metal_chunk_binary_paths() -> list[Path]:
    root = _workspace_root()
    pkg = Path(__file__).resolve().parent
    repo_via_file = pkg.parent.parent.parent
    out: list[Path] = []
    env_bin = os.getenv("COLLATZ_METAL_SIEVE_BINARY", "").strip()
    if env_bin:
        out.append(Path(env_bin).expanduser())
    out.append(root / "scripts" / "native_sieve_kit" / "metal" / "metal_sieve_chunk")
    out.append(repo_via_file / "scripts" / "native_sieve_kit" / "metal" / "metal_sieve_chunk")
    return out


@lru_cache(maxsize=1)
def metal_sieve_chunk_binary_path() -> Path | None:
    """First existing, executable ``metal_sieve_chunk`` path, or ``None``."""
    for p in _default_metal_chunk_binary_paths():
        if not p or not p.is_file():
            continue
        if os.access(p, os.X_OK):
            return p
    return None


@lru_cache(maxsize=4)
def _metal_ping_payload(bin_path_resolved: str) -> dict[str, Any]:
    try:
        p = subprocess.run(
            [bin_path_resolved, "--ping"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if p.returncode != 0:
            return {}
        return json.loads(p.stdout.strip() or "{}")
    except Exception:
        return {}


def native_metal_sieve_available() -> bool:
    """True only on macOS when the helper binary exists and ``--ping`` succeeds."""
    if platform.system() != "Darwin":
        return False
    bin_path = metal_sieve_chunk_binary_path()
    if bin_path is None:
        return False
    return bool(_metal_ping_payload(str(bin_path.resolve())).get("ok"))


def gpu_sieve_metal_without_torch_allowed() -> bool:
    """When True, ``gpu-sieve`` may run on native Metal without PyTorch MPS (Darwin only)."""
    return os.getenv("COLLATZ_GPU_SIEVE_METAL_WITHOUT_TORCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def gpu_sieve_backend_mode() -> str:
    raw = os.getenv("COLLATZ_GPU_SIEVE_BACKEND", "auto").strip().lower()
    if raw in {"auto", "mps", "metal"}:
        return raw
    return "auto"


def should_use_native_metal_sieve() -> bool:
    """Whether ``compute_range_metrics_gpu_sieve`` should try the Metal helper first."""
    if platform.system() != "Darwin":
        return False
    mode = gpu_sieve_backend_mode()
    if mode == "mps":
        return False
    if mode == "metal":
        if not native_metal_sieve_available():
            raise ValueError(
                "COLLATZ_GPU_SIEVE_BACKEND=metal but metal_sieve_chunk is missing or not executable. "
                "Build: bash scripts/native_sieve_kit/metal/build_metal_sieve_chunk.sh"
            )
        return True
    return native_metal_sieve_available()


def _prefer_metal_stdio(bin_path: Path) -> bool:
    if os.getenv("COLLATZ_METAL_SIEVE_STDIO", "1").strip().lower() in {"0", "false", "no"}:
        return False
    return bool(_metal_ping_payload(str(bin_path.resolve())).get("stdio"))


def _close_stdio_pipes(proc: subprocess.Popen) -> None:
    """Close parent-side pipe ends after the child is gone (avoid FD leaks)."""
    for attr in ("stdin", "stdout", "stderr"):
        stream = getattr(proc, attr, None)
        try:
            if stream is not None and hasattr(stream, "closed") and not stream.closed:
                stream.close()
        except Exception:
            pass


def _terminate_metal_stdio_child(proc: subprocess.Popen | None) -> None:
    """Stop ``metal_sieve_chunk --stdio`` politely, then SIGTERM/SIGKILL; close pipes when dead."""
    if proc is None:
        return
    if proc.poll() is not None:
        _close_stdio_pipes(proc)
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.write('{"op":"quit"}\n')
            proc.stdin.flush()
    except Exception:
        pass
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        pass
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass
    _close_stdio_pipes(proc)


def _shutdown_metal_stdio_server() -> None:
    global _stdio_proc, _stdio_bin
    with _stdio_lock:
        proc = _stdio_proc
        _stdio_proc = None
        _stdio_bin = None
    _terminate_metal_stdio_child(proc)


atexit.register(_shutdown_metal_stdio_server)


def shutdown_metal_stdio_transport() -> None:
    """Terminate the persistent ``metal_sieve_chunk --stdio`` child if any.

    The Swift helper keeps a **grow-only** per-seed ``steps`` buffer (~4 bytes × largest chunk
    processed). While the child stays alive (for throughput), that RAM is not returned to the OS.
    Call this after a heavy sweep/run, when changing env vars read at helper startup
    (e.g. ``COLLATZ_METAL_SIEVE_STDIO_PIPELINE``), or from ``POST .../stdio-shutdown`` when idle.
    """
    _shutdown_metal_stdio_server()


def _ensure_metal_stdio_server_unlocked(bin_path: Path) -> subprocess.Popen:
    global _stdio_proc, _stdio_bin
    key = str(bin_path.resolve())
    if _stdio_proc is not None and _stdio_proc.poll() is None and _stdio_bin == key:
        return _stdio_proc
    if _stdio_proc is not None:
        old = _stdio_proc
        _stdio_proc = None
        _stdio_bin = None
        _terminate_metal_stdio_child(old)

    _stdio_proc = subprocess.Popen(
        [key, "--stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    _stdio_bin = key
    return _stdio_proc


def _run_metal_chunk_oneshot(bin_path: Path, first_odd: int, count: int) -> dict[str, Any]:
    p = subprocess.run(
        [
            str(bin_path),
            "--first-odd",
            str(first_odd),
            "--count",
            str(count),
        ],
        capture_output=True,
        text=True,
        timeout=86400,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"metal_sieve_chunk failed (exit {p.returncode}): {p.stderr[:2000]!r}"
        )
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    if not line.startswith("{"):
        raise RuntimeError(f"metal_sieve_chunk: no JSON on stdout: {p.stdout[:500]!r}")
    return json.loads(line)


def _run_metal_chunk_stdio(bin_path: Path, first_odd: int, count: int) -> dict[str, Any]:
    line_out: str
    with _stdio_lock:
        proc = _ensure_metal_stdio_server_unlocked(bin_path)
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("metal_sieve_chunk stdio pipes missing")
        req = json.dumps({"first_odd": int(first_odd), "count": int(count)}) + "\n"
        proc.stdin.write(req)
        proc.stdin.flush()
        line_out = proc.stdout.readline()
        if proc.poll() is not None:
            err = ""
            try:
                if proc.stderr and not proc.stderr.closed:
                    err = proc.stderr.read()[:2000]
            except Exception:
                err = ""
            rc = proc.returncode
            if rc is None:
                rc = proc.poll()
            raise RuntimeError(
                "metal_sieve_chunk stdio process exited mid-request "
                f"(returncode={rc!r}, stderr={err!r}, stdout_line={line_out[:200]!r})"
            )
    line_out = line_out.strip()
    if not line_out.startswith("{"):
        raise RuntimeError(f"metal_sieve_chunk stdio: bad line {line_out[:300]!r}")
    payload = json.loads(line_out)
    if payload.get("error"):
        raise RuntimeError(f"metal_sieve_chunk stdio error: {payload.get('error')!r}")
    return payload


def _run_metal_chunk(bin_path: Path, first_odd: int, count: int) -> dict[str, Any]:
    if _prefer_metal_stdio(bin_path):
        try:
            return _run_metal_chunk_stdio(bin_path, first_odd, count)
        except Exception as exc:
            logger.warning("Metal stdio chunk failed; falling back to one-shot subprocess: %s", exc)
    return _run_metal_chunk_oneshot(bin_path, first_odd, count)


def diagnostics_native_metal_sieve() -> dict[str, Any]:
    """For dashboards / debugging (safe on all platforms)."""
    want: bool | None = None
    want_err: str | None = None
    bp = metal_sieve_chunk_binary_path()
    ping: dict[str, Any] = {}
    if bp:
        ping = dict(_metal_ping_payload(str(bp.resolve())))
    if platform.system() == "Darwin" and gpu_sieve_backend_mode() != "mps":
        try:
            want = should_use_native_metal_sieve()
        except ValueError as exc:
            want_err = str(exc)
    stdio_supported = bool(ping.get("stdio"))
    stdio_on = bool(bp and stdio_supported and _prefer_metal_stdio(bp))
    from .metal_chunk_calibration import get_metal_chunk_calibration_status

    diag = {
        "platform": platform.system(),
        "backend_mode": gpu_sieve_backend_mode(),
        "helper_path": str(bp) if bp else None,
        "available": native_metal_sieve_available(),
        "ping": ping,
        "stdio_supported": stdio_supported,
        "stdio_transport_active": stdio_on,
        "stdio_pipeline_env": os.getenv("COLLATZ_METAL_SIEVE_STDIO_PIPELINE", "1"),
        "metal_chunk_max_odds": metal_sieve_chunk_max_odds(),
        "metal_chunk_auto_enabled": metal_sieve_chunk_auto_enabled(),
        "metal_chunk_resolved_odds": resolve_metal_sieve_chunk_odds(),
        "metal_without_torch_env": gpu_sieve_metal_without_torch_allowed(),
        "would_use_metal": want,
        "would_use_metal_error": want_err,
    }
    diag.update(get_metal_chunk_calibration_status())
    return diag


def _metal_sieve_chunk_cap() -> int:
    """Upper bound for odds per Metal chunk (``steps`` buffer ≈ 4 × chunk bytes on the helper)."""
    try:
        cap = int(os.getenv("COLLATZ_METAL_SIEVE_CHUNK_MAX", "16777216"))
    except ValueError:
        cap = 16_777_216
    # Hard ceiling: UInt32 count in the helper; keep headroom below 2**32.
    return max(4096, min(67_108_864, cap))


def metal_sieve_chunk_max_odds() -> int:
    """Public cap for ``COLLATZ_METAL_SIEVE_CHUNK_SIZE`` (honours ``COLLATZ_METAL_SIEVE_CHUNK_MAX``)."""
    return _metal_sieve_chunk_cap()


def metal_sieve_chunk_auto_enabled() -> bool:
    """Whether unresolved ``COLLATZ_METAL_SIEVE_CHUNK_SIZE`` uses auto tuning (macOS only).

    Order: **fresh throughput calibration** file (``--write-calibration``), else **RAM/swap ladder**.
    """
    if platform.system() != "Darwin":
        return False
    raw = os.getenv("COLLATZ_METAL_SIEVE_CHUNK_AUTO", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


# Descending ladder: pick largest that fits memory-derived max (``steps`` ≈ 4 bytes × odds).
_METAL_CHUNK_AUTO_LADDER: tuple[int, ...] = (
    16_777_216,
    8_388_608,
    4_194_304,
    2_097_152,
    1_048_576,
    524_288,
)


def _estimate_bytes_available_for_metal_steps_buffer() -> int:
    """Conservative ``available`` RAM for sizing the helper's grow-only steps buffer."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        avail = int(vm.available)
        try:
            sw = psutil.swap_memory()
            if sw.total > 0 and sw.percent > 55.0:
                avail = min(avail, int(avail * 0.45))
        except Exception:
            pass
        return max(256 << 20, avail)
    except Exception:
        return 4 << 30


def _auto_metal_chunk_odds(cap: int) -> int:
    """Pick chunk odds: **throughput calibration** (if fresh), else RAM ladder.

    Calibration comes from ``data/metal_sieve_chunk_calibration.json`` written by
    ``profile_metal_sieve_chunk.py --write-calibration``. It is always clamped by
    the same RAM/swap ceiling as the ladder so we do not OOM.
    """
    from .metal_chunk_calibration import choose_chunk_from_calibration

    avail = _estimate_bytes_available_for_metal_steps_buffer()
    # steps buffer ≈ 4 * odds; leave headroom for Python, Metal driver, other tabs.
    max_odds = min(cap, max(524_288, avail // 48))
    cal = choose_chunk_from_calibration(cap, max_odds)
    if cal is not None:
        return cal
    for c in _METAL_CHUNK_AUTO_LADDER:
        if c <= max_odds:
            return max(4096, c)
    return max(4096, min(cap, 524_288))


def resolve_metal_sieve_chunk_odds() -> int:
    """Effective odds-per-chunk for Metal (explicit env, else auto or 4 Mi default)."""
    return _metal_sieve_chunk_size()


def _metal_sieve_chunk_size() -> int:
    """Odds per Metal chunk (separate knob from MPS if needed)."""
    cap = _metal_sieve_chunk_cap()
    try:
        v = int(os.getenv("COLLATZ_METAL_SIEVE_CHUNK_SIZE", "0"))
    except ValueError:
        v = 0
    if v > 0:
        return max(4096, min(cap, v))
    if metal_sieve_chunk_auto_enabled():
        return _auto_metal_chunk_odds(cap)
    # Native Metal has no MPS-style .any() sync cadence; larger chunks amortise
    # subprocess / stdio framing vs the ~1M default used for MPS.
    return min(4_194_304, cap)


def compute_range_metrics_gpu_sieve_metal(
    start: int,
    end: int,
    *,
    start_at: int | None = None,
    sample_limit: int = 12,
    profile: Any = None,
) -> Any:
    """Streaming Metal gpu-sieve: same aggregate contract as MPS path."""
    from .services import (
        INT64_MAX,
        AggregateMetrics,
        _OverflowPatch,
        metrics_descent_direct,
    )

    if platform.system() != "Darwin":
        raise RuntimeError("Metal gpu-sieve is only supported on macOS.")

    bin_path = metal_sieve_chunk_binary_path()
    if bin_path is None:
        raise RuntimeError("metal_sieve_chunk binary not found.")

    first = start_at if start_at is not None else start
    if first < start or first > end:
        raise ValueError("Checkpoint start is outside the run interval.")

    first_odd = first if first & 1 else first + 1
    if first_odd > end:
        return AggregateMetrics(
            processed=0,
            last_processed=end,
            max_total_stopping_time={"n": first, "value": 0},
            max_stopping_time={"n": first, "value": 0},
            max_excursion={"n": first, "value": 0},
            sample_records=[],
        )

    odd_count = ((end - first_odd) // 2) + 1
    chunk_size = _metal_sieve_chunk_size()

    best_total_val = -1
    best_total_seed = first_odd
    best_exc_val = -1
    best_exc_seed = first_odd
    patches: list[_OverflowPatch] = []

    offset = 0
    while offset < odd_count:
        chunk = min(chunk_size, odd_count - offset)
        chunk_first_odd = first_odd + 2 * offset
        payload = _run_metal_chunk(bin_path, chunk_first_odd, chunk)

        mt = payload["max_total_stopping_time"]
        me = payload["max_excursion"]
        chunk_best_tst = int(mt["value"])
        chunk_best_tst_n = int(mt["n"])
        chunk_best_exc = int(me["value"])
        chunk_best_exc_n = int(me["n"])

        for seed in payload.get("overflow_seeds") or []:
            seed = int(seed)
            m = metrics_descent_direct(seed)
            if m.max_excursion > INT64_MAX:
                patches.append(
                    _OverflowPatch(seed, m.total_stopping_time, m.stopping_time, m.max_excursion)
                )
            if m.total_stopping_time > chunk_best_tst:
                chunk_best_tst = m.total_stopping_time
                chunk_best_tst_n = seed
            exc_cap = INT64_MAX if m.max_excursion > INT64_MAX else int(m.max_excursion)
            if exc_cap > chunk_best_exc:
                chunk_best_exc = exc_cap
                chunk_best_exc_n = seed

        if chunk_best_tst > best_total_val:
            best_total_val = chunk_best_tst
            best_total_seed = chunk_best_tst_n
        if chunk_best_exc > best_exc_val:
            best_exc_val = chunk_best_exc
            best_exc_seed = chunk_best_exc_n

        offset += chunk

    max_total = {"n": best_total_seed, "value": best_total_val}
    max_stopping = {"n": best_total_seed, "value": best_total_val}
    max_excursion_agg = {"n": best_exc_seed, "value": best_exc_val}

    sample_records: list[dict] = []
    if best_total_val > 0:
        sample_records.append({"metric": "max_total_stopping_time", **max_total})
        sample_records.append({"metric": "max_stopping_time", **max_stopping})
    if best_exc_val > 0:
        sample_records.append({"metric": "max_excursion", **max_excursion_agg})

    aggregate = AggregateMetrics(
        processed=odd_count,
        last_processed=end,
        max_total_stopping_time=max_total,
        max_stopping_time=max_stopping,
        max_excursion=max_excursion_agg,
        sample_records=sample_records,
    )

    for patch in patches:
        if patch.max_excursion > aggregate.max_excursion["value"]:
            aggregate.max_excursion = {"n": patch.seed, "value": patch.max_excursion}
        if patch.total_stopping_time > aggregate.max_total_stopping_time["value"]:
            aggregate.max_total_stopping_time = {"n": patch.seed, "value": patch.total_stopping_time}
        if patch.stopping_time > aggregate.max_stopping_time["value"]:
            aggregate.max_stopping_time = {"n": patch.seed, "value": patch.stopping_time}
    return aggregate

#!/usr/bin/env python3
"""
Run the native Metal spike and the lab's gpu-sieve (MPS) on the **same odd workload**, then print a comparison.

From repo root (recommended: project venv activated):

  PYTHONPATH=backend/src python3 scripts/metal_native_spike/compare_with_lab.py

Or with explicit Python:

  python3 scripts/metal_native_spike/compare_with_lab.py --python /path/to/.venv/bin/python

If `--python` is omitted, the script prefers **`<repo>/.venv/bin/python`** when that file exists (so plain `python3 compare_with_lab.py` still picks up PyTorch MPS after `mac-dev-stack` / `pip install -e backend[mps]`).

Requires: Xcode Metal toolchain + swiftc; PyTorch with MPS for gpu-sieve.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def _odd_linear_end(base: int, odd_count: int) -> int:
    """Last odd seed when taking odd_count odds starting at base (base must be odd)."""
    b = base | 1
    return b + 2 * (odd_count - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Metal spike vs lab gpu-sieve (same odds).")
    ap.add_argument("--count", type=int, default=500_000, help="Number of odd seeds (default 500k).")
    ap.add_argument("--base", type=int, default=1, help="First odd seed (default 1).")
    ap.add_argument(
        "--python",
        default=None,
        help="Python for lab import (default: repo .venv/bin/python if present, else current interpreter).",
    )
    ap.add_argument(
        "--preset-env",
        choices=("tuned", "extreme", "default"),
        default="tuned",
        help="MPS batch/sync env before gpu-sieve (default: tuned 1M/128).",
    )
    ap.add_argument("--no-warmup", action="store_true", help="Skip small gpu-sieve warmup.")
    args = ap.parse_args()

    spike_dir = os.path.dirname(os.path.abspath(__file__))
    repo = _repo_root()
    venv_python = os.path.join(repo, ".venv", "bin", "python")
    chosen_py = args.python or (venv_python if os.path.isfile(venv_python) else sys.executable)
    # gpu-sieve imports torch in this process — re-exec to project venv if needed
    try:
        if os.path.realpath(chosen_py) != os.path.realpath(sys.executable):
            script_path = os.path.abspath(sys.argv[0])
            os.execv(chosen_py, [chosen_py, script_path, *sys.argv[1:]])
    except OSError as exc:
        print(f"error: could not switch to {chosen_py}: {exc}", file=sys.stderr)
        return 2

    run_sh = os.path.join(spike_dir, "run.sh")

    base = int(args.base) | 1
    oc = int(args.count)
    end = _odd_linear_end(base, oc)

    env_lab = os.environ.copy()
    env_lab["PYTHONPATH"] = os.path.join(repo, "backend", "src")
    presets = {
        "tuned": {
            "COLLATZ_MPS_SIEVE_BATCH_SIZE": "1048576",
            "COLLATZ_MPS_SYNC_EVERY": "128",
        },
        "extreme": {
            "COLLATZ_MPS_SIEVE_BATCH_SIZE": "2097152",
            "COLLATZ_MPS_SYNC_EVERY": "128",
        },
        "default": {},
    }
    for k, v in presets[args.preset_env].items():
        env_lab[k] = v

    print("== Native Metal spike (Swift + .metallib) ==", flush=True)
    try:
        p = subprocess.run(
            ["bash", run_sh, "--count", str(oc), "--base", str(base)],
            cwd=spike_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        print("error: bash or run.sh missing.", file=sys.stderr)
        return 2
    if p.returncode != 0:
        print(p.stderr, end="", file=sys.stderr)
        print(p.stdout, end="", file=sys.stderr)
        return p.returncode

    metal_line = ""
    for line in p.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            metal_line = line
    if not metal_line:
        print("error: no JSON line from metal_native_spike on stdout.", file=sys.stderr)
        print(p.stdout, file=sys.stderr)
        return 3
    metal = json.loads(metal_line)

    print(metal_line, flush=True)
    print(flush=True)

    # Lazy import after spike (Metal does not need Python torch)
    sys.path.insert(0, env_lab["PYTHONPATH"])
    from collatz_lab.hardware import GPU_SIEVE_KERNEL, gpu_execution_ready
    from collatz_lab.services import compute_range_metrics

    print(
        f"== Lab gpu-sieve [{args.preset_env}]  range [{base}, {end}]  (~{oc:,} odds) ==",
        flush=True,
    )
    if not gpu_execution_ready():
        print(
            "GPU not ready in this Python (no MPS/CUDA). Skipping gpu-sieve.\n"
            "Run this script on your Mac with the project venv where torch.mps works.",
            flush=True,
        )
        print(json.dumps({"metal": metal, "gpu_sieve": None}, indent=2))
        return 0

    if not args.no_warmup:
        compute_range_metrics(1, 3000, kernel=GPU_SIEVE_KERNEL)

    t0 = time.perf_counter()
    agg = compute_range_metrics(base, end, kernel=GPU_SIEVE_KERNEL)
    dt = time.perf_counter() - t0
    odds = int(agg.processed)
    lab = {
        "kernel": GPU_SIEVE_KERNEL,
        "preset": args.preset_env,
        "start": base,
        "end": end,
        "odd_seeds": odds,
        "seconds": round(dt, 6),
        "odd_per_sec": round(odds / dt, 0) if dt > 0 else 0.0,
    }
    print(json.dumps(lab), flush=True)
    print(flush=True)

    m_ops = float(metal.get("odd_per_sec", 0))
    l_ops = float(lab["odd_per_sec"])
    ratio = m_ops / l_ops if l_ops > 0 else 0.0
    print("== Summary ==", flush=True)
    print(f"  Metal odd/s:     {m_ops:,.0f}", flush=True)
    print(f"  gpu-sieve odd/s: {l_ops:,.0f}", flush=True)
    print(
        "  Note: Metal spike = one raw dispatch + uint64 odd-descent kernel only.\n"
        "        gpu-sieve = full lab path (PyTorch MPS, batching, sync, aggregates to CPU/Python).\n"
        "        Large ratios are expected; they bound “kernel ceiling” vs “current stack”, not identical work.",
        flush=True,
    )
    if l_ops <= 0:
        print("  (Cannot compare ratio — gpu-sieve odd/s is zero.)", flush=True)
    elif ratio >= 1.0:
        print(f"  Metal is ~{ratio:.2f}× faster than gpu-sieve (this workload).", flush=True)
    else:
        print(f"  gpu-sieve is ~{(1.0 / ratio):.2f}× faster than Metal spike (this workload).", flush=True)
    print(flush=True)
    print(json.dumps({"metal": metal, "gpu_sieve": lab, "metal_vs_mps_ratio": round(ratio, 4)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

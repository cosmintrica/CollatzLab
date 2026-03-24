# Kernel portability — guardrails (do not regress the baseline)

This note is **not** a feature spec. It defines **how** we extend kernels and hardware detection so we do **not** break the current stable path.

## Baseline regression target (must stay healthy)

Treat the following as the **primary compatibility bar** until we explicitly widen it with tests and sign-off:

- **OS:** Windows (x86_64)
- **CPU:** AMD (and Intel x86_64 in the same code paths where applicable)
- **GPU (optional):** NVIDIA + CUDA where Numba exposes the device today
- **Kernels in active use:** existing CPU parallel / direct paths and current Numba CUDA kernels (`gpu-*`) as wired in the worker and CLI

Any change that degrades throughput, changes numeric semantics, breaks checkpoint/resume, or drops GPU availability **without** an explicit opt-in path is a **regression**.

## API vs kernels (scope)

- **HTTP API (FastAPI):** transport and orchestration; largely OS-agnostic.
- **Kernels:** the numeric Collatz executors (CPU JIT, GPU JIT, batching, checkpoints). Portability work belongs **here** and in **`collatz_lab.hardware` / worker dispatch**, not in “adding endpoints” unless needed for capability reporting.

## Rules for kernel / hardware work

1. **CPU-first default** — Never require a GPU for core workflows. GPU remains optional and explicit in the run contract (see [`HARDWARE_AND_KERNELS.md`](./HARDWARE_AND_KERNELS.md)).
2. **No silent semantic change** — Same `kernel` name + same run parameters must produce the **same mathematical result** class (e.g. same stopping-time / excursion semantics) across platforms unless the change is versioned (new kernel name or explicit migration note) and covered by tests.
3. **Detect, then dispatch** — New platforms (Apple Silicon Metal, ROCm, etc.) add **detection** and **separate code paths**; avoid rewriting the Windows/NVIDIA path in place without a compatibility shim.
4. **Feature isolation** — Prefer `if backend_ready:` branches or small adapter modules over large edits to hot loops shared with Windows.
5. **Tests** — Run `pytest` on every PR (`backend/tests`). Add targeted tests when touching dispatch or kernel selection.
6. **Manual smoke (maintainer)** — Before merging large kernel or CUDA-path refactors, run a **short** CPU and (if available) GPU run on **Windows + NVIDIA** to confirm workers still claim runs and checkpoints look sane.

## Phased plan pointer

The long platform matrix and phased rollout (CPU baseline → optional accelerators) lives in [`HARDWARE_AND_KERNELS.md`](./HARDWARE_AND_KERNELS.md). This guardrail doc only states **non-negotiables** for the **current** Windows/AMD/NVIDIA baseline.

## macOS / Homebrew Python

Backend installs for Apple Silicon or Homebrew Python should use a **project virtualenv** (PEP 668). From the repo root:

```bash
bash scripts/setup-venv.sh
source .venv/bin/activate
```

See the root [`README.md`](../README.md) Quickstart for more detail.

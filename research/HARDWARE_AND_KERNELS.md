# Hardware targets and kernel portability

This note consolidates **where Collatz Lab should run** and **how we prepare kernels gradually**. It is planning documentation only: implementation follows in small steps (`collatz_lab/hardware/` package adapters, detection, then optional backends).

## Principles

1. **CPU-first everywhere** — All current verification kernels (`cpu-direct`, `cpu-accelerated`, `cpu-parallel`, `cpu-parallel-odd`, `cpu-sieve`, `cpu-barina`) must remain the portable baseline on every OS and ISA we care about.
2. **GPU is optional and explicit** — A worker advertises only kernels that are actually executable on that machine (`gpu-collatz-accelerated`, `gpu-sieve` today = Numba/CUDA only). No silent fallback that changes semantics.
3. **One logical run contract** — Same SQLite run shape, checkpoints, and validation story regardless of backend; only the executor changes.
4. **Gradual rollout** — Detect and classify hardware first; add accelerators one family at a time with tests and small validation ranges.

## CPU platforms (baseline)

| Class | Examples | Notes |
|-------|----------|--------|
| Desktop / laptop x86_64 | Intel Core, AMD Ryzen (Windows, Linux) | Primary dev target today; Numba CPU parallel works. |
| Apple Silicon (ARM64) | M1, M2, M3, M4, future unified SoCs (laptop / desktop) | Python + Numba **CPU** JIT generally works; treat as first-class for local lab users. |
| Windows on ARM64 | Snapdragon X Elite / Plus laptops, other WoA | Same as above: CPU kernels must run; GPU story TBD (often no CUDA). |
| Linux ARM64 | AWS Graviton, Ampere, Raspberry Pi 4/5 class | CPU baseline; Pi-scale machines may need smaller default batch sizes (policy later). |
| Older / embedded ARM | Various SBCs | Best-effort CPU-only; not a performance target. |

**Intel vs AMD** on x86_64: same ISA for our purposes; detection can still record vendor/model for reproducibility in run metadata.

## GPU and accelerator classes (optional paths)

| Class | Typical hardware | Candidate runtime (future) | Status |
|-------|------------------|-----------------------------|--------|
| NVIDIA discrete | RTX / Quadro / datacenter (T4, L4, A100, …) | **Numba CUDA** (current) | Implemented |
| NVIDIA mobile | Some laptop GPUs | CUDA if driver + toolkit align | Same stack when CUDA available |
| AMD discrete | Radeon RX / workstation | **ROCm / HIP** (Numba ROCm or alternate JIT) | Not implemented; research spike |
| Intel integrated / Arc | UHD, Iris, Arc | **oneAPI / SYCL / Level Zero**, or vendor samples | Not implemented; long-term option |
| Apple integrated | M-series GPU | **PyTorch MPS** (used by Collatz Lab for `gpu-collatz-accelerated`; sieve uses CPU Numba parity path) | Implemented behind optional `backend[mps]` (Torch) |
| Qualcomm Adreno | Phones, some WoA | OpenCL / Vulkan compute (theoretical) | Very long-term; low priority |
| “Phone-class” SoC | iPhone / iPad class (A-series, etc.) | Not a lab worker target | Out of scope unless we define a separate minimal client; compute and OS constraints differ |

**Integrated vs dedicated** only changes memory bandwidth and batch tuning; **detection** should record VRAM (or unified memory budget on Apple) and expose it in capability metadata so the scheduler can prefer CPU for tiny ranges on weak iGPUs if we add policy later.

## OS matrix (summary)

| OS | CPU path | GPU path (today) | GPU path (planned / research) |
|----|----------|------------------|-------------------------------|
| Windows x86_64 | Yes | NVIDIA CUDA | AMD ROCm, Intel (TBD) |
| Windows ARM64 | Yes | Rare CUDA if present | WoA GPU APIs TBD |
| Linux x86_64 | Yes | NVIDIA CUDA | AMD ROCm, Intel (TBD) |
| Linux ARM64 | Yes | CUDA if ARM64 build + driver | — |
| macOS (Apple Silicon) | Yes | — | Metal family (TBD implementation) |
| macOS (Intel, legacy) | Yes | NVIDIA CUDA if eGPU/driver | Same as Linux CUDA where applicable |

## Phased preparation (incremental)

**Phase A — Detection and adapters (no new kernels required)**  
- Phase A (done): `collatz_lab/hardware/` package with `constants`, `platform`, `windows_cuda`, `gpu`, `nvidia`, `discovery`, `selection`, `util`, `metrics`, `cpu_label`; public imports remain `from collatz_lab.hardware import …`.
- **Phase A2 (smart inventory):** `hardware/gpu_inventory.py` always merges **nvidia-smi** with OS probes in `hardware/adapters/display.py` (`system_profiler`, `lspci`, `Win32_VideoController`). Duplicate NVIDIA rows from PCI/WMI are dropped when SMI already returned devices; hybrid laptops keep Intel/AMD/Apple rows. Each capability includes `metadata.smart_detection` (schema version, probes used, `collatz_gpu_executable`, `next_backends_research`, optional `signals` / `raw_snippet` for dev spikes). Non-CUDA rows keep `supported_kernels=[]`. CUDA remains the only executable GPU path until Phase B.
- **CPU usage:** optional `psutil` (`backend[system]`); otherwise Windows performance counters, Linux `/proc/stat` delta, macOS without psutil leaves usage blank (browser Pressure API can still hint in the UI).
- Next: flesh out richer arch adapters and optional **executable** non-CUDA backends (Phase B).  
- Report: CPU vendor, model, core count, ISA; GPU vendor, model, driver, and whether **CUDA** (or future backends) probe as ready.  
- Dashboard / worker: show this in capabilities JSON so runs are reproducible.

**Phase B — First extra backend (choose one)**  
- Example: **Apple Metal** *or* **AMD ROCm** after a spike proves a single kernel can match CPU reference on a fixed interval.  
- Add new kernel **names** only if semantics differ; otherwise reuse `gpu-sieve` / `gpu-collatz-accelerated` with a `backend` or `device_kind` in metadata.

**Phase C — Policy and tuning**  
- Auto-select default hardware/kernel for *new* runs based on detected class (still overridable).  
- Batch sizes and thread counts per device class (integrated vs dedicated, unified memory vs VRAM).

**Phase D — Exotic / cloud**  
- Document-only for Vulkan/SYCL unless a maintainer commits to maintaining it.

## Related documents

- [`DEVELOPMENT_BACKLOG.md`](./DEVELOPMENT_BACKLOG.md) — priorities 0 and 8.  
- [`ROADMAP.md`](./ROADMAP.md) — Phase 2 compute.  
- [`WORKER_QUEUE.md`](./WORKER_QUEUE.md) — queue contract and hardware visibility.

## Changelog

- **2026-03-23** — Initial consolidation: CPU ISA matrix, integrated vs dedicated GPU, Apple Silicon / WoA / ARM Linux, NVIDIA/AMD/Intel/Qualcomm placeholders, phased rollout.
- **2026-03-23** — Phase A2: smart GPU inventory (`gpu_inventory` + `adapters/display`) + cross-platform CPU usage probes; documented `backend[system]` for psutil.
- **2026-03-23** — Phase A2b: `metadata.smart_detection` on CPU/GPU rows for automatic probe provenance and backend research hints; see [`SMART_DETECTION.md`](./SMART_DETECTION.md).

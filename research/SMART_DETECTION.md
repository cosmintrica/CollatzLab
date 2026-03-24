# Smart hardware detection metadata

Every `HardwareCapability` returned by `discover_hardware()` may include `metadata.smart_detection`: a **stable, JSON-safe** object describing how the row was produced and what to build next when Collatz cannot execute on that device.

## Goals

- **Automatic** — filled by the probe pipeline without manual edits.
- **Modular** — each adapter adds its own `probes` / `signals`; aggregation lives in `gpu_inventory` and `discovery`.
- **Correct** — `collatz_gpu_executable` / `collatz_cpu_executable` reflect whether **this repo’s current kernels** can run, not whether the silicon exists.

## Schema (`schema_version` = 1)

### CPU (`kind == "cpu"`)

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | int | Always `1` for this document. |
| `probes` | list[str] | Logical steps (e.g. `detect_cpu_label`, `platform_context`). |
| `collatz_cpu_executable` | bool | `true` for the baseline CPU kernel set. |
| `primary_signal` | str | e.g. `python_os_cpu`. |
| `vendor_class` | str | `cpu`. |
| `next_backends_research` | list[str] | Tunables (JIT, threading), not alternate ISAs. |
| `signals` | object? | e.g. `cpu_usage_probe`, `cpu_label_source`, `logical_cores`. |

### GPU (`kind == "gpu"`)

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | int | `1`. |
| `probes` | list[str] | e.g. `nvidia-smi_query_gpu`, `lspci_vga_class`, `system_profiler_SPDisplaysDataType`. |
| `collatz_gpu_executable` | bool | `true` only when this row advertises non-empty `supported_kernels` (CUDA path today). |
| `primary_signal` | str | Dominant data source for the row. |
| `vendor_class` | str | `nvidia` / `apple` / `amd` / `intel` / `unknown`. |
| `next_backends_research` | list[str] | Ordered hints (`cuda_numba`, `metal`, `rocm`, …). |
| `development_note` | str? | Human-readable gap description. |
| `signals` | object? | e.g. `driver_version`, `smi_index`, `probe_tool`, `raw_snippet` (truncated). |

## Consumers

- **Dashboard** — can surface diagnostics; full JSON is always available from `/api/workers/capabilities` and `/api/hardware`.
- **Developers** — export capabilities JSON when filing Metal/ROCm issues; attach `smart_detection.signals`.

## macOS gotchas

- **`system_profiler` and `sysctl` live under `/usr/sbin`**. If the API process inherits a minimal `PATH` (IDE, `launchd`, some CI agents), bare `system_profiler` fails with “file not found” and the inventory used to show **no GPU**. The code now tries **`/usr/sbin/system_profiler`**, then **`ioreg`**, then a **sysctl heuristic** (`hw.optional.gpu` + chip name) on Apple Silicon.
- **Docker / Linux VM on a Mac**: the Python process often reports `platform.system() == "Linux"`. Inventory then reflects the **container**, not the Mac GPU — run the API **natively on macOS** (or mount `lspci`/GPU — non-trivial) for host-like probes.

## Related code

- `collatz_lab/hardware/smart_detection.py` — builders and vendor → research map.
- `collatz_lab/hardware/gpu_inventory.py` — merge + dedupe NVIDIA vs display probes.
- `collatz_lab/hardware/adapters/display.py` — macOS / Linux / Windows display probes.

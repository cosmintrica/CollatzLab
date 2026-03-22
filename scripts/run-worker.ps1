param(
  [string]$Name = "managed-worker",
  [ValidateSet("auto", "cpu", "gpu")]
  [string]$Hardware = "auto",
  [int]$PollInterval = 5
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$env:COLLATZ_LAB_ROOT = $root.Path
$env:COLLATZ_LAB_WORKER_NAME = $Name
$env:COLLATZ_LAB_WORKER_HARDWARE = $Hardware
# Batch sizes control how much work the kernel does between checkpoints.
# Larger batches = higher hardware utilization (less checkpoint I/O overhead).
# RTX 4060 Ti can process ~500M seeds/checkpoint comfortably.
$env:COLLATZ_CPU_PARALLEL_BATCH_SIZE = "250000000"
$env:COLLATZ_CPU_PARALLEL_ODD_BATCH_SIZE = "500000000"
$env:COLLATZ_GPU_BATCH_SIZE = "500000000"
$env:COLLATZ_GPU_THREADS_PER_BLOCK = "256"
$env:NUMBA_NUM_THREADS = [string]([Environment]::ProcessorCount)

$backendSrc = Join-Path $root "backend\src"
if ($env:PYTHONPATH) {
  $env:PYTHONPATH = "$backendSrc;$($env:PYTHONPATH)"
} else {
  $env:PYTHONPATH = $backendSrc
}

python -m collatz_lab.cli worker start --name $Name --hardware $Hardware --poll-interval $PollInterval

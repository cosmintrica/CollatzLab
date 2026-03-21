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
$env:COLLATZ_CPU_PARALLEL_BATCH_SIZE = "2000000"
$env:COLLATZ_GPU_BATCH_SIZE = "10000000"
$env:COLLATZ_GPU_THREADS_PER_BLOCK = "256"

$backendSrc = Join-Path $root "backend\src"
if ($env:PYTHONPATH) {
  $env:PYTHONPATH = "$backendSrc;$($env:PYTHONPATH)"
} else {
  $env:PYTHONPATH = $backendSrc
}

python -m collatz_lab.cli worker start --name $Name --hardware $Hardware --poll-interval $PollInterval

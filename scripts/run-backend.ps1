$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$env:COLLATZ_LAB_ROOT = $root.Path
$backendSrc = Join-Path $root "backend\src"
if ($env:PYTHONPATH) {
  $env:PYTHONPATH = "$backendSrc;$($env:PYTHONPATH)"
} else {
  $env:PYTHONPATH = $backendSrc
}

python -m collatz_lab.main

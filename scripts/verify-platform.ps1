# Environment check before starting the stack (Windows / PowerShell).
# From repo root:
#   powershell -ExecutionPolicy Bypass -File .\scripts\verify-platform.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\verify-platform.ps1 -Full
param(
  [switch]$Full,
  [switch]$NoDashboardBuild,
  [switch]$CheckApi
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$env:COLLATZ_LAB_ROOT = $root.Path
$backendSrc = Join-Path $root "backend\src"
if ($env:PYTHONPATH) {
  $env:PYTHONPATH = "$backendSrc;$env:PYTHONPATH"
} else {
  $env:PYTHONPATH = $backendSrc
}

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Error "Missing .venv. Run: pip install -e .\backend[dev] in a venv at the project root."
}

Write-Host "== Collatz Lab — platform check (ROOT=$root) =="

Write-Host "[1/5] Import collatz_lab ..."
& $py -c "import collatz_lab; print('  OK')"

Write-Host "[2/5] collatz_lab.cli init ..."
& $py -m collatz_lab.cli init

Write-Host "[3/5] FastAPI app smoke ..."
& $py -c @"
from collatz_lab.config import Settings
from collatz_lab.api import create_app
app = create_app(Settings.from_env())
print('  OK routes:', len(app.routes))
"@

if (-not $NoDashboardBuild) {
  Write-Host "[4/5] Dashboard: npm run dashboard:build ..."
  npm run dashboard:build
} else {
  Write-Host "[4/5] Dashboard build skipped (-NoDashboardBuild)."
}

if ($Full) {
  Write-Host "[5/5] pytest backend/tests ..."
  & $py -m pytest backend/tests -q --tb=short
} else {
  Write-Host "[5/5] pytest skipped (use -Full for all tests)."
}

if ($CheckApi) {
  $health = if ($env:COLLATZ_LAB_HEALTH_URL) { $env:COLLATZ_LAB_HEALTH_URL } else { "http://127.0.0.1:8000/health" }
  try {
    $r = Invoke-WebRequest -Uri $health -TimeoutSec 5 -UseBasicParsing
    if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) {
      Write-Host "API: /health OK"
    }
  } catch {
    Write-Warning "API did not respond at $health (expected if the stack is not running)."
  }
}

Write-Host ""
Write-Host "Check succeeded. Start the stack: npm run stack:start:worker"

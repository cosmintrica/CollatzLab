param(
  [ValidateSet("static", "vite")]
  [string]$Mode = "static"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

if ($Mode -eq "vite") {
  npm run dev --prefix .\dashboard -- --host 0.0.0.0
  exit $LASTEXITCODE
}

npm run build --prefix .\dashboard
python -m http.server 5173 --bind 127.0.0.1 --directory dashboard/dist

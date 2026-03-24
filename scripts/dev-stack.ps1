param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("start", "stop", "restart", "status")]
  [string]$Action,
  [ValidateSet("static", "vite")]
  [string]$FrontendMode = "static",
  [switch]$WithWorker
)

# Optional (same as macOS): COLLATZ_STACK_CPU_WORKERS, COLLATZ_STACK_GPU_WORKERS (integers >=1).
# Each CPU worker gets NUMBA_NUM_THREADS ≈ ceil(ProcessorCount / CPU workers).

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$runtimeDir = Join-Path $root ".runtime"
$statePath = Join-Path $runtimeDir "dev-stack.json"
$backendOutLog = Join-Path $runtimeDir "backend.out.log"
$backendErrLog = Join-Path $runtimeDir "backend.err.log"
$dashboardOutLog = Join-Path $runtimeDir "dashboard.out.log"
$dashboardErrLog = Join-Path $runtimeDir "dashboard.err.log"
$workerOutLog = Join-Path $runtimeDir "worker.out.log"
$workerErrLog = Join-Path $runtimeDir "worker.err.log"
$backendUrl = "http://127.0.0.1:8000/health"
$dashboardUrl = "http://127.0.0.1:5173/"

function Ensure-RuntimeDir {
  if (-not (Test-Path $runtimeDir)) {
    New-Item -ItemType Directory -Path $runtimeDir | Out-Null
  }
}

function Load-State {
  if (-not (Test-Path $statePath)) {
    return $null
  }
  return Get-Content $statePath -Raw | ConvertFrom-Json
}

function Save-State($state) {
  Ensure-RuntimeDir
  $state | ConvertTo-Json -Depth 8 | Set-Content -Path $statePath -Encoding UTF8
}

function Remove-State {
  if (Test-Path $statePath) {
    Remove-Item $statePath -Force
  }
}

function Test-Pid([int]$ProcessId) {
  try {
    Get-Process -Id $ProcessId -ErrorAction Stop | Out-Null
    return $true
  } catch {
    return $false
  }
}

function Stop-Tree([int]$ProcessId) {
  if (-not $ProcessId) {
    return
  }
  if (-not (Test-Pid $ProcessId)) {
    return
  }
  & taskkill /PID $ProcessId /T /F | Out-Null
}

function Invoke-HealthCheck([string]$Uri, [int]$TimeoutSeconds = 60) {
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $response = Invoke-WebRequest -Uri $Uri -TimeoutSec 5
      if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
        return $true
      }
    } catch {
      Start-Sleep -Milliseconds 500
    }
  }
  return $false
}

function Wait-ForTcpPort([string]$Address, [int]$Port, [int]$TimeoutSeconds = 30) {
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
      $async = $client.BeginConnect($Address, $Port, $null, $null)
      if ($async.AsyncWaitHandle.WaitOne(1000)) {
        $client.EndConnect($async)
        return $true
      }
    } catch {
    } finally {
      $client.Close()
    }
    Start-Sleep -Milliseconds 300
  }
  return $false
}

function Start-ManagedWorker([string]$Name, [ValidateSet("auto", "cpu", "gpu")][string]$Hardware, [int]$PollInterval = 5) {
  $workerScript = Join-Path $root "scripts\run-worker.ps1"
  $workerScriptArg = '"' + $workerScript + '"'
  $outLog = Join-Path $runtimeDir ("{0}.out.log" -f $Name)
  $errLog = Join-Path $runtimeDir ("{0}.err.log" -f $Name)
  if (Test-Path $outLog) {
    Remove-Item $outLog -Force
  }
  if (Test-Path $errLog) {
    Remove-Item $errLog -Force
  }

  $process = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @(
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      $workerScriptArg,
      "-Name",
      $Name,
      "-Hardware",
      $Hardware,
      "-PollInterval",
      $PollInterval
    ) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog `
    -PassThru `
    -WindowStyle Hidden

  return [pscustomobject]@{
    name = $Name
    hardware = $Hardware
    pid = $process.Id
    logs = @($outLog, $errLog)
    required = $Hardware -eq "cpu"
  }
}

function Resolve-WorkerEntry($entry) {
  if ($null -eq $entry) {
    return $null
  }
  $running = $false
  if ($entry.PSObject.Properties.Name -contains "pid" -and $null -ne $entry.pid) {
    $running = Test-Pid ([int]$entry.pid)
  }
  return [pscustomobject]@{
    name = $entry.name
    hardware = $entry.hardware
    pid = if ($entry.PSObject.Properties.Name -contains "pid") { [int]$entry.pid } else { $null }
    running = $running
    required = if ($entry.PSObject.Properties.Name -contains "required") { [bool]$entry.required } else { $false }
    logs = if ($entry.PSObject.Properties.Name -contains "logs") { $entry.logs } else { @() }
  }
}

function Get-WorkerStateEntries($state) {
  if ($null -eq $state) {
    return @()
  }
  if ($state.PSObject.Properties.Name -contains "workers" -and $null -ne $state.workers) {
    return @($state.workers | ForEach-Object { Resolve-WorkerEntry $_ })
  }
  if ($state.PSObject.Properties.Name -contains "workerPid" -and $null -ne $state.workerPid) {
    return @(
      [pscustomobject]@{
        name = "managed-worker"
        hardware = if ($state.PSObject.Properties.Name -contains "workerHardware" -and $state.workerHardware) { $state.workerHardware } else { "auto" }
        pid = [int]$state.workerPid
        running = Test-Pid ([int]$state.workerPid)
        required = [bool]($state.workerEnabled -as [bool])
        logs = if ($state.PSObject.Properties.Name -contains "workerLogs") { $state.workerLogs } else { @() }
      }
    )
  }
  return @()
}

function Get-StatusObject {
  $state = Load-State
  if ($null -eq $state) {
    return [pscustomobject]@{
      running = $false
      backend = "stopped"
      dashboard = "stopped"
      worker = "disabled"
      cpuWorker = "disabled"
      gpuWorker = "disabled"
      workers = @()
      stateFile = $statePath
    }
  }

  $backendRunning = Test-Pid ([int]$state.backendPid)
  $dashboardRunning = Test-Pid ([int]$state.dashboardPid)
  $workers = Get-WorkerStateEntries $state
  $runningWorkers = @($workers | Where-Object { $_.running })
  $cpuWorker = $workers | Where-Object { $_.hardware -eq "cpu" } | Select-Object -First 1
  $gpuWorker = $workers | Where-Object { $_.hardware -eq "gpu" } | Select-Object -First 1

  return [pscustomobject]@{
    running = ($backendRunning -or $dashboardRunning -or $runningWorkers.Count -gt 0)
    backend = if ($backendRunning) { "running" } else { "stopped" }
    backendPid = [int]$state.backendPid
    backendUrl = $state.backendUrl
    backendLogs = $state.backendLogs
    dashboard = if ($dashboardRunning) { "running" } else { "stopped" }
    dashboardPid = [int]$state.dashboardPid
    dashboardUrl = $state.dashboardUrl
    dashboardLogs = $state.dashboardLogs
    worker = if ($runningWorkers.Count -gt 0) { "running" } elseif ($workers.Count -gt 0) { "stopped" } else { "disabled" }
    cpuWorker = if ($null -ne $cpuWorker) { if ($cpuWorker.running) { "running" } else { "stopped" } } else { "disabled" }
    gpuWorker = if ($null -ne $gpuWorker) { if ($gpuWorker.running) { "running" } else { "stopped" } } else { "disabled" }
    workers = $workers
    workerEnabled = if ($state.PSObject.Properties.Name -contains "workerEnabled") { [bool]$state.workerEnabled } else { $false }
    startedAt = $state.startedAt
    stateFile = $statePath
  }
}

function Stop-Stack {
  $state = Load-State
  if ($null -eq $state) {
    Write-Output "Stack already stopped."
    return
  }

  Stop-Tree ([int]$state.backendPid)
  Stop-Tree ([int]$state.dashboardPid)
  $workers = Get-WorkerStateEntries $state
  foreach ($worker in $workers) {
    Stop-Tree ([int]$worker.pid)
  }
  Remove-State
  Write-Output "Stack stopped."
}

function Start-Stack {
  Ensure-RuntimeDir
  $existing = Load-State
  if ($null -ne $existing) {
    $backendRunning = Test-Pid ([int]$existing.backendPid)
    $dashboardRunning = Test-Pid ([int]$existing.dashboardPid)
    $existingWorkers = Get-WorkerStateEntries $existing
    $workersRunning = @($existingWorkers | Where-Object { $_.running })
    if ($backendRunning -and $dashboardRunning -and (($WithWorker -and $workersRunning.Count -gt 0) -or (-not $WithWorker))) {
      Write-Output "Stack already running."
      Get-StatusObject | ConvertTo-Json -Depth 6
      return
    }
    Stop-Stack | Out-Null
  }

  foreach ($path in @($backendOutLog, $backendErrLog, $dashboardOutLog, $dashboardErrLog)) {
    if (Test-Path $path) {
      Remove-Item $path -Force
    }
  }

  $backendScript = Join-Path $root "scripts\run-backend.ps1"
  $dashboardScript = Join-Path $root "scripts\run-dashboard.ps1"
  $backendScriptArg = '"' + $backendScript + '"'
  $dashboardScriptArg = '"' + $dashboardScript + '"'

  $backendProcess = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $backendScriptArg) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $backendOutLog `
    -RedirectStandardError $backendErrLog `
    -PassThru `
    -WindowStyle Hidden

  $dashboardProcess = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $dashboardScriptArg, "-Mode", $FrontendMode) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $dashboardOutLog `
    -RedirectStandardError $dashboardErrLog `
    -PassThru `
    -WindowStyle Hidden

  $workers = @()
  if ($WithWorker) {
    $cpuCount = 1
    $gpuCount = 1
    if ($env:COLLATZ_STACK_CPU_WORKERS -match '^\d+$') { $cpuCount = [int]$env:COLLATZ_STACK_CPU_WORKERS }
    if ($env:COLLATZ_STACK_GPU_WORKERS -match '^\d+$') { $gpuCount = [int]$env:COLLATZ_STACK_GPU_WORKERS }
    if ($cpuCount -lt 1) { $cpuCount = 1 }
    if ($gpuCount -lt 1) { $gpuCount = 1 }
    if ($cpuCount -gt 16) { throw "COLLATZ_STACK_CPU_WORKERS must be between 1 and 16." }
    if ($gpuCount -gt 8) { throw "COLLATZ_STACK_GPU_WORKERS must be between 1 and 8." }

    $cores = [Environment]::ProcessorCount
    $threadsCpu = [int][Math]::Ceiling($cores / [double]$cpuCount)
    if ($threadsCpu -lt 1) { $threadsCpu = 1 }
    $threadsGpu = $cores
    if ($gpuCount -gt 1) {
      $threadsGpu = [int][Math]::Ceiling($cores / [double]$gpuCount)
    }

    for ($i = 1; $i -le $cpuCount; $i++) {
      $env:NUMBA_NUM_THREADS = "$threadsCpu"
      $cpuName = if ($cpuCount -eq 1) { "managed-worker-cpu" } else { "managed-worker-cpu-$i" }
      $workers += Start-ManagedWorker -Name $cpuName -Hardware "cpu" -PollInterval 5
    }
    for ($i = 1; $i -le $gpuCount; $i++) {
      $env:NUMBA_NUM_THREADS = "$threadsGpu"
      $gpuName = if ($gpuCount -eq 1) { "managed-worker-gpu" } else { "managed-worker-gpu-$i" }
      $workers += Start-ManagedWorker -Name $gpuName -Hardware "gpu" -PollInterval 5
    }
    Remove-Item Env:NUMBA_NUM_THREADS -ErrorAction SilentlyContinue
  }

  $state = [pscustomobject]@{
    backendPid = $backendProcess.Id
    dashboardPid = $dashboardProcess.Id
    backendUrl = $backendUrl
    dashboardUrl = $dashboardUrl
    frontendMode = $FrontendMode
    backendLogs = @($backendOutLog, $backendErrLog)
    dashboardLogs = @($dashboardOutLog, $dashboardErrLog)
    workers = $workers
    workerEnabled = [bool]$WithWorker
    startedAt = (Get-Date).ToString("o")
  }
  Save-State $state

  $backendHealthy = Wait-ForTcpPort -Address "127.0.0.1" -Port 8000
  $dashboardHealthy = Wait-ForTcpPort -Address "127.0.0.1" -Port 5173
  $workerHealthy = $true
  if ($WithWorker) {
    Start-Sleep -Seconds 2
    $cpuWorkers = @($workers | Where-Object { $_.hardware -eq "cpu" })
    $gpuWorkers = @($workers | Where-Object { $_.hardware -eq "gpu" })
    $cpuHealthy = $true
    foreach ($cw in $cpuWorkers) {
      if (-not (Test-Pid ([int]$cw.pid))) { $cpuHealthy = $false }
    }
    if (-not $cpuHealthy) {
      $workerHealthy = $false
    }
    $anyGpuDead = $false
    foreach ($gw in $gpuWorkers) {
      if (-not (Test-Pid ([int]$gw.pid))) { $anyGpuDead = $true }
    }
    if ($anyGpuDead) {
      Write-Warning "One or more GPU workers did not stay alive. CPU workers keep running."
    }
  }
  if (-not ($backendHealthy -and $dashboardHealthy -and $workerHealthy)) {
    Stop-Tree $backendProcess.Id
    Stop-Tree $dashboardProcess.Id
    foreach ($worker in $workers) {
      Stop-Tree ([int]$worker.pid)
    }
    Remove-State
    throw "Failed to start the full stack. Check $backendOutLog, $backendErrLog, $dashboardOutLog, $dashboardErrLog, $workerOutLog, and $workerErrLog."
  }

  Write-Output "Stack started."
  Get-StatusObject | ConvertTo-Json -Depth 6
}

function Show-Status {
  Get-StatusObject | ConvertTo-Json -Depth 6
}

switch ($Action) {
  "start" { Start-Stack }
  "stop" { Stop-Stack }
  "restart" {
    Stop-Stack
    Start-Stack
  }
  "status" { Show-Status }
}

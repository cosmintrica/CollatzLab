#!/usr/bin/env bash
# macOS dev stack: venv + deps (Apple Silicon: MPS), init, API + Vite + worker.
# Mirrors scripts/dev-stack.ps1 for Unix (no PowerShell required).
#
# Usage:
#   bash scripts/mac-dev-stack.sh start
#   bash scripts/mac-dev-stack.sh stop
#   bash scripts/mac-dev-stack.sh stop force   # also kill anything listening on :8000 and :5173
#   bash scripts/mac-dev-stack.sh restart
#   bash scripts/mac-dev-stack.sh status
#
# Optional environment:
#   COLLATZ_MAC_STACK_NO_WORKER=1  — do not start any workers (runs stay queued)
#   COLLATZ_MAC_STACK_SINGLE_WORKER=1  — one worker with --hardware auto (legacy). Default matches Windows: CPU + GPU workers.
#   COLLATZ_STACK_CPU_WORKERS=N  — managed CPU workers (default 1). NUMBA threads ≈ ceil(ncpu/N) each to limit oversubscription.
#   COLLATZ_STACK_GPU_WORKERS=M  — managed GPU workers (default 1). M>1 on a single Apple GPU is usually counterproductive.
#   COLLATZ_MAC_STACK_NO_SEED=1    — do not enqueue the tiny first-run demo job
#   COLLATZ_MAC_STACK_ALLOW_BUSY_API=1 — if /health responds but /api/directions is not Collatz-shaped,
#       try to spawn a second API anyway (usually fails to bind; expert / debugging only).
#   COLLATZ_SKIP_COMPUTE_THROTTLE=0 — worker honors DB compute profile throttling (default for this script: 1 = off).
#   COLLATZ_MPS_SIEVE_BATCH_SIZE — MPS gpu-sieve sub-chunk (default 1048576; from profile_mps_metal_sieve --quick winner).
#   COLLATZ_MPS_SYNC_EVERY — .any() sync cadence in MPS kernels (default 128; higher values regressed in profiling).
#   COLLATZ_MAC_EXTREME_THROUGHPUT=1 — if unset, sets 2M chunk + sync 128 (not 4M/256 — slower in MPS sweep).
#   COLLATZ_RANDOM_PROBES=0 — disable auto small randomized runs (owner collatz-random-probe) when the queue would stay empty.
#   COLLATZ_RANDOM_SEED — optional int/string seed for reproducible random enqueue in tests or debugging.
#   COLLATZ_MAC_STACK_BUILD_NATIVES=1 — on "start", build libsieve_descent_native (+ Metal on macOS) before workers (non-fatal if a step fails).
set -euo pipefail

ACTION="${1:-}"
STOP_FORCE_ARG="${2:-}"
if [[ -z "$ACTION" ]]; then
  echo "usage: $0 {start|stop [force]|restart|status|build-natives}" >&2
  exit 1
fi
if [[ "$ACTION" == "stop" && -n "$STOP_FORCE_ARG" && "$STOP_FORCE_ARG" != "force" ]]; then
  echo "usage: $0 stop [force]" >&2
  exit 1
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

RUNTIME_DIR="$root/.runtime"
STATE_PATH="$RUNTIME_DIR/mac-dev-stack.json"
BACKEND_OUT="$RUNTIME_DIR/backend.out.log"
BACKEND_ERR="$RUNTIME_DIR/backend.err.log"
DASH_OUT="$RUNTIME_DIR/dashboard.out.log"
DASH_ERR="$RUNTIME_DIR/dashboard.err.log"
WORKER_OUT="$RUNTIME_DIR/worker.out.log"
WORKER_ERR="$RUNTIME_DIR/worker.err.log"
WORKER_CPU_OUT="$RUNTIME_DIR/mac-managed-worker-cpu.out.log"
WORKER_CPU_ERR="$RUNTIME_DIR/mac-managed-worker-cpu.err.log"
WORKER_GPU_OUT="$RUNTIME_DIR/mac-managed-worker-gpu.out.log"
WORKER_GPU_ERR="$RUNTIME_DIR/mac-managed-worker-gpu.err.log"
STATE_HELPER="$root/scripts/collatz-mac-stack-state.py"

HEALTH_URL="${COLLATZ_LAB_HEALTH_URL:-http://127.0.0.1:8000/health}"
API_HOST_PORT="127.0.0.1:8000"

ensure_runtime_dir() {
  mkdir -p "$RUNTIME_DIR"
}

kill_tree() {
  local pid="$1"
  [[ -z "$pid" || "$pid" == "0" || ! "$pid" =~ ^[0-9]+$ ]] && return 0
  local children
  children=$(pgrep -P "$pid" 2>/dev/null || true)
  local c
  for c in $children; do
    kill_tree "$c"
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    sleep 0.3
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
}

wait_for_health() {
  local deadline=$((SECONDS + 90))
  while (( SECONDS < deadline )); do
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.4
  done
  echo "error: API did not become healthy at $HEALTH_URL (see $BACKEND_ERR)" >&2
  return 1
}

# e.g. http://127.0.0.1:8000/health -> http://127.0.0.1:8000
health_url_to_api_base() {
  local u="${1%/}"
  if [[ "$u" == */health ]]; then
    echo "${u%/health}"
  else
    echo "$u"
  fi
}

# True if this looks like our FastAPI app (same platform you already started on :8000).
collatz_lab_api_detected() {
  local base="${1%/}"
  local py="$root/.venv/bin/python"
  [[ -x "$py" ]] || return 1
  curl -sf "${base}/api/directions" 2>/dev/null | "$py" -c "
import json, sys
try:
    j = json.load(sys.stdin)
    if not isinstance(j, list) or len(j) < 1:
        raise SystemExit(1)
    for item in j[:8]:
        if not isinstance(item, dict) or 'slug' not in item:
            raise SystemExit(1)
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
" >/dev/null 2>&1
}

is_darwin() {
  [[ "$(uname -s)" == "Darwin" ]]
}

is_apple_silicon() {
  is_darwin && [[ "$(uname -m)" == "arm64" ]]
}

require_macos_for_start() {
  if ! is_darwin; then
    echo "error: mac-dev-stack.sh start is intended for macOS." >&2
    echo "  On Linux: use scripts/setup-venv.sh, then run-backend.sh / run-dashboard.sh / run-worker.sh in separate terminals." >&2
    exit 1
  fi
}

pick_python_for_venv() {
  if command -v python3.13 >/dev/null 2>&1; then
    echo "python3.13"
  elif command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12"
  elif command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    echo ""
  fi
}

ensure_venv_and_deps() {
  local py_creator
  py_creator="$(pick_python_for_venv)"
  if [[ -z "$py_creator" ]]; then
    echo "error: need Python 3.11+ (python3.13 recommended). Install from python.org or Homebrew." >&2
    exit 1
  fi

  if [[ ! -d "$root/.venv" ]]; then
    echo "Creating project .venv with $py_creator (avoids Homebrew PEP 668 'externally-managed-environment' errors) ..."
    "$py_creator" -m venv "$root/.venv"
  fi

  local PY="$root/.venv/bin/python"
  local PIP="$root/.venv/bin/pip"
  [[ -x "$PY" ]] || { echo "error: $PY missing" >&2; exit 1; }

  "$PY" -m pip install -U pip setuptools wheel

  local extras="dev,system"
  if is_apple_silicon; then
    extras="dev,system,mps"
    echo "Apple Silicon (arm64): installing backend[$extras] (PyTorch MPS for GPU kernels)."
  else
    echo "macOS (non-arm64): installing backend[$extras] (no MPS extra)."
  fi

  "$PIP" install -e "backend[$extras]"

  if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "error: Node.js (node + npm) is required for the dashboard. Install from https://nodejs.org/ or nvm." >&2
    exit 1
  fi

  if [[ ! -d "$root/dashboard/node_modules" ]]; then
    if [[ -f "$root/dashboard/package-lock.json" ]]; then
      echo "Installing dashboard dependencies (npm ci) ..."
      (cd "$root/dashboard" && npm ci)
    else
      echo "Installing dashboard dependencies (npm install) ..."
      (cd "$root/dashboard" && npm install)
    fi
  else
    echo "dashboard/node_modules present; skipping npm install. Remove it to force a clean install."
  fi
}

active_py() {
  if [[ -x "$root/.venv/bin/python" ]]; then
    echo "$root/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    echo ""
  fi
}

run_init() {
  local PY
  PY="$(active_py)"
  [[ -n "$PY" ]] || { echo "error: no python after venv setup" >&2; exit 1; }
  export COLLATZ_LAB_ROOT="$root"
  export PYTHONPATH="${root}/backend/src${PYTHONPATH:+:$PYTHONPATH}"
  "$PY" -m collatz_lab.cli init
}

count_runs() {
  local PY
  PY="$(active_py)"
  export COLLATZ_LAB_ROOT="$root"
  export PYTHONPATH="${root}/backend/src${PYTHONPATH:+:$PYTHONPATH}"
  "$PY" -c "
from collatz_lab.config import Settings
from collatz_lab.repository import LabRepository
print(len(LabRepository(Settings.from_env()).list_runs()))
"
}

maybe_seed_demo_run() {
  [[ "${COLLATZ_MAC_STACK_NO_SEED:-}" == "1" ]] && return 0
  local n
  n="$(count_runs)"
  if [[ "$n" != "0" ]]; then
    return 0
  fi
  local PY
  PY="$(active_py)"
  export COLLATZ_LAB_ROOT="$root"
  export PYTHONPATH="${root}/backend/src${PYTHONPATH:+:$PYTHONPATH}"
  echo "No runs in DB — enqueueing a tiny CPU demo run for the dashboard ledger."
  "$PY" -m collatz_lab.cli run start \
    --direction verification \
    --name mac-stack-welcome \
    --start 1 \
    --end 2000 \
    --kernel cpu-parallel \
    --hardware cpu \
    --enqueue-only
}

persist_stack_state() {
  local bp="$1" dp="$2" workers_json="$3"
  local PY="$root/.venv/bin/python"
  if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || true)"
  fi
  if [[ -z "$PY" ]]; then
    echo "error: need python to write stack state" >&2
    exit 1
  fi
  "$PY" "$STATE_HELPER" write "$STATE_PATH" "$HEALTH_URL" "$bp" "$dp" "$workers_json"
}

read_pids_from_state() {
  local PY="$root/.venv/bin/python"
  if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || true)"
  fi
  if [[ ! -f "$STATE_PATH" ]] || [[ -z "$PY" ]]; then
    echo ""
    return 1
  fi
  "$PY" "$STATE_HELPER" print-pids "$STATE_PATH"
}

pid_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

# Build native CPU .dylib/.so (+ metal_sieve_chunk on macOS). Safe to run anytime; logs warnings on failure.
cmd_build_natives() {
  echo "=== build-natives: CPU libsieve_descent_native ==="
  if command -v cc >/dev/null 2>&1; then
    (cd "$root/scripts/native_sieve_kit" && bash build_native_cpu_sieve_lib.sh) \
      || echo "warning: build_native_cpu_sieve_lib.sh failed (cpu-sieve stays on Numba)" >&2
  else
    echo "warning: no 'cc' on PATH; skip libsieve_descent_native" >&2
  fi
  if is_darwin; then
    echo "=== build-natives: Metal metal_sieve_chunk (requires Xcode CLT) ==="
    if command -v xcrun >/dev/null 2>&1; then
      (cd "$root/scripts/native_sieve_kit/metal" && bash build_metal_sieve_chunk.sh) \
        || echo "warning: build_metal_sieve_chunk.sh failed (gpu-sieve will use MPS if PyTorch is OK)" >&2
    else
      echo "warning: xcrun missing; skip metal_sieve_chunk" >&2
    fi
  else
    echo "=== build-natives: Metal skipped (macOS only) ==="
  fi
}

# Same env as Windows managed workers (dev-stack.ps1 Start-ManagedWorker).
launch_collatz_worker() {
  local name="$1" hw="$2" std_out="$3" std_err="$4" nb_override="${5:-}"
  (
    cd "$root"
    export COLLATZ_LAB_ROOT="$root"
    export PYTHONPATH="${root}/backend/src${PYTHONPATH:+:$PYTHONPATH}"
    export COLLATZ_LAB_WORKER_NAME="$name"
    export COLLATZ_LAB_WORKER_HARDWARE="$hw"
    export COLLATZ_CPU_SIEVE_BACKEND="${COLLATZ_CPU_SIEVE_BACKEND:-auto}"
    export COLLATZ_GPU_SIEVE_BACKEND="${COLLATZ_GPU_SIEVE_BACKEND:-auto}"
    export COLLATZ_SKIP_COMPUTE_THROTTLE="${COLLATZ_SKIP_COMPUTE_THROTTLE:-1}"
    export COLLATZ_CPU_PARALLEL_BATCH_SIZE="250000000"
    export COLLATZ_CPU_PARALLEL_ODD_BATCH_SIZE="500000000"
    export COLLATZ_GPU_BATCH_SIZE="500000000"
    export COLLATZ_GPU_THREADS_PER_BLOCK="256"
    export COLLATZ_CPU_SIEVE_BATCH_SIZE="${COLLATZ_CPU_SIEVE_BATCH_SIZE:-250000000}"
    if [[ "${COLLATZ_MAC_EXTREME_THROUGHPUT:-}" == "1" ]]; then
      : "${COLLATZ_MPS_SIEVE_BATCH_SIZE:=2097152}"
      : "${COLLATZ_MPS_SYNC_EVERY:=128}"
    fi
    export COLLATZ_MPS_SIEVE_BATCH_SIZE="${COLLATZ_MPS_SIEVE_BATCH_SIZE:-1048576}"
    export COLLATZ_MPS_SYNC_EVERY="${COLLATZ_MPS_SYNC_EVERY:-128}"
    if [[ -n "$nb_override" ]]; then
      export NUMBA_NUM_THREADS="$nb_override"
    else
      export NUMBA_NUM_THREADS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    fi
    # Native cpu-sieve (OpenMP in libsieve_descent_native) honors OMP_NUM_THREADS; align with Numba unless set explicitly.
    if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
      export OMP_NUM_THREADS="$NUMBA_NUM_THREADS"
    fi
    # Homebrew libomp (native .dylib) + PyTorch's bundled libomp in one process → OMP #15 abort without this.
    export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
    exec "$root/.venv/bin/python" -m collatz_lab.cli worker start \
      --name "$name" --hardware "$hw" --poll-interval 5
  ) >>"$std_out" 2>>"$std_err" &
  echo $!
}

cmd_start() {
  require_macos_for_start

  if [[ -f "$STATE_PATH" ]]; then
    local line p
    line="$(read_pids_from_state || true)"
    if [[ -n "$line" ]]; then
      IFS='|' read -ra ST <<<"$line"
      for p in "${ST[@]}"; do
        [[ "$p" == "0" ]] && continue
        if pid_alive "$p"; then
          echo "Stack appears already running (state: $STATE_PATH). Stop first: $0 stop" >&2
          exit 1
        fi
      done
    fi
  fi

  command -v curl >/dev/null 2>&1 || {
    echo "error: curl is required (macOS usually has it)." >&2
    exit 1
  }

  ensure_runtime_dir
  : >"$BACKEND_OUT"
  : >"$BACKEND_ERR"
  : >"$DASH_OUT"
  : >"$DASH_ERR"
  : >"$WORKER_OUT"
  : >"$WORKER_ERR"
  : >"$WORKER_CPU_OUT"
  : >"$WORKER_CPU_ERR"
  : >"$WORKER_GPU_OUT"
  : >"$WORKER_GPU_ERR"

  ensure_venv_and_deps

  if [[ "${COLLATZ_MAC_STACK_BUILD_NATIVES:-}" == "1" ]]; then
    echo "COLLATZ_MAC_STACK_BUILD_NATIVES=1 — building native libraries before workers ..."
    cmd_build_natives || true
  fi

  run_init

  export COLLATZ_LAB_ROOT="$root"
  export PYTHONPATH="${root}/backend/src${PYTHONPATH:+:$PYTHONPATH}"

  local api_base
  api_base="$(health_url_to_api_base "$HEALTH_URL")"
  local backend_pid=0
  local reuse_api=0

  if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
    if collatz_lab_api_detected "$api_base"; then
      echo "Collatz Lab API already running at $api_base — reusing it (this script will not start a second server)."
      reuse_api=1
      backend_pid=0
    elif [[ "${COLLATZ_MAC_STACK_ALLOW_BUSY_API:-}" == "1" ]]; then
      echo "warning: $HEALTH_URL responds but $api_base/api/directions is not clearly Collatz Lab; attempting to start another API (bind may fail)." >&2
    else
      echo "error: $HEALTH_URL responds, but $api_base/api/directions does not look like Collatz Lab." >&2
      echo "  Another application may be using the port. Stop it or change the API port in .env, then retry." >&2
      if command -v lsof >/dev/null 2>&1; then
        echo "  Check listener: lsof -nP -iTCP:8000 -sTCP:LISTEN" >&2
      fi
      echo "  Expert override: COLLATZ_MAC_STACK_ALLOW_BUSY_API=1 $0 start" >&2
      exit 1
    fi
  fi

  if [[ "$reuse_api" != "1" ]]; then
    echo "Starting API (logs: $BACKEND_OUT) ..."
    (
      cd "$root"
      export COLLATZ_LAB_ROOT="$root"
      export PYTHONPATH="${root}/backend/src${PYTHONPATH:+:$PYTHONPATH}"
      exec "$root/.venv/bin/python" -m collatz_lab.main
    ) >>"$BACKEND_OUT" 2>>"$BACKEND_ERR" &
    backend_pid=$!

    if ! wait_for_health; then
      kill_tree "$backend_pid"
      exit 1
    fi
    if ! pid_alive "$backend_pid"; then
      echo "error: API process exited while $HEALTH_URL still responded (another server on the port?)." >&2
      echo "  Our process is dead; see $BACKEND_ERR" >&2
      tail -30 "$BACKEND_ERR" >&2
      exit 1
    fi
    echo "API healthy (started by this script, pid $backend_pid)."
  else
    echo "API healthy (existing Collatz Lab process; not managed as a child of this script)."
  fi

  local workers_json="[]"
  if [[ "${COLLATZ_MAC_STACK_NO_WORKER:-}" != "1" ]]; then
    if [[ "${COLLATZ_MAC_STACK_SINGLE_WORKER:-}" == "1" ]]; then
      echo "Starting single worker --hardware auto (logs: $WORKER_OUT) ..."
      local wp_single
      wp_single="$(launch_collatz_worker mac-managed-worker auto "$WORKER_OUT" "$WORKER_ERR")"
      workers_json="[{\"name\":\"mac-managed-worker\",\"hardware\":\"auto\",\"pid\":${wp_single}}]"
    else
      local n_cpu="${COLLATZ_STACK_CPU_WORKERS:-1}"
      local n_gpu="${COLLATZ_STACK_GPU_WORKERS:-1}"
      [[ "$n_cpu" =~ ^[0-9]+$ ]] || n_cpu=1
      [[ "$n_gpu" =~ ^[0-9]+$ ]] || n_gpu=1
      [[ "$n_cpu" -lt 1 ]] && n_cpu=1
      [[ "$n_gpu" -lt 1 ]] && n_gpu=1
      [[ "$n_cpu" -gt 16 ]] && {
        echo "error: COLLATZ_STACK_CPU_WORKERS must be 1..16" >&2
        exit 1
      }
      [[ "$n_gpu" -gt 8 ]] && {
        echo "error: COLLATZ_STACK_GPU_WORKERS must be 1..8" >&2
        exit 1
      }
      local nproc
      nproc="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
      local threads_cpu=$(( (nproc + n_cpu - 1) / n_cpu ))
      local threads_gpu="$nproc"
      [[ "$n_gpu" -gt 1 ]] && threads_gpu=$(( (nproc + n_gpu - 1) / n_gpu ))
      local -a cpu_pids=()
      local -a gpu_pids=()
      local i wout werr pid
      for ((i = 1; i <= n_cpu; i++)); do
        local wname_cpu
        if [[ "$n_cpu" -eq 1 ]]; then
          wname_cpu="mac-managed-worker-cpu"
          wout="$RUNTIME_DIR/mac-managed-worker-cpu.out.log"
          werr="$RUNTIME_DIR/mac-managed-worker-cpu.err.log"
        else
          wname_cpu="mac-managed-worker-cpu-${i}"
          wout="$RUNTIME_DIR/mac-managed-worker-cpu-${i}.out.log"
          werr="$RUNTIME_DIR/mac-managed-worker-cpu-${i}.err.log"
        fi
        : >"$wout"
        : >"$werr"
        echo "Starting CPU worker $i/$n_cpu $wname_cpu (NUMBA_NUM_THREADS=$threads_cpu, logs: $wout) ..."
        pid="$(launch_collatz_worker "$wname_cpu" cpu "$wout" "$werr" "$threads_cpu")"
        cpu_pids+=("$pid")
      done
      for ((i = 1; i <= n_gpu; i++)); do
        local wname_gpu
        if [[ "$n_gpu" -eq 1 ]]; then
          wname_gpu="mac-managed-worker-gpu"
          wout="$RUNTIME_DIR/mac-managed-worker-gpu.out.log"
          werr="$RUNTIME_DIR/mac-managed-worker-gpu.err.log"
        else
          wname_gpu="mac-managed-worker-gpu-${i}"
          wout="$RUNTIME_DIR/mac-managed-worker-gpu-${i}.out.log"
          werr="$RUNTIME_DIR/mac-managed-worker-gpu-${i}.err.log"
        fi
        : >"$wout"
        : >"$werr"
        echo "Starting GPU worker $i/$n_gpu $wname_gpu (NUMBA_NUM_THREADS=$threads_gpu for CPU-side fallbacks, logs: $wout) ..."
        pid="$(launch_collatz_worker "$wname_gpu" gpu "$wout" "$werr" "$threads_gpu")"
        gpu_pids+=("$pid")
      done
      sleep 2
      for pid in "${cpu_pids[@]}"; do
        if [[ "$pid" != "0" ]] && ! pid_alive "$pid"; then
          echo "error: a CPU worker (pid $pid) exited immediately — see logs under $RUNTIME_DIR/mac-managed-worker-cpu-*.err.log" >&2
          for p in "${gpu_pids[@]}"; do
            [[ "$p" != "0" ]] && kill_tree "$p"
          done
          [[ "$backend_pid" != "0" ]] && kill_tree "$backend_pid"
          exit 1
        fi
      done
      for pid in "${gpu_pids[@]}"; do
        if [[ "$pid" != "0" ]] && ! pid_alive "$pid"; then
          echo "warning: a GPU worker (pid $pid) did not stay alive — see $RUNTIME_DIR/mac-managed-worker-gpu-*.err.log. CPU workers keep running." >&2
        fi
      done
      local PYJSON="$root/.venv/bin/python"
      [[ -x "$PYJSON" ]] || PYJSON="$(command -v python3)"
      local _oj_ifs="$IFS"
      IFS=,
      local _cpu_j="[${cpu_pids[*]}]"
      local _gpu_j="[${gpu_pids[*]}]"
      IFS="$_oj_ifs"
      workers_json="$("$PYJSON" -c "
import json, sys
cpu = json.loads(sys.argv[1])
gpu = json.loads(sys.argv[2])
nc, ng = len(cpu), len(gpu)
w = []
for i, p in enumerate(cpu):
    nm = 'mac-managed-worker-cpu' if nc == 1 else ('mac-managed-worker-cpu-%d' % (i + 1))
    w.append({'name': nm, 'hardware': 'cpu', 'pid': int(p)})
for i, p in enumerate(gpu):
    nm = 'mac-managed-worker-gpu' if ng == 1 else ('mac-managed-worker-gpu-%d' % (i + 1))
    w.append({'name': nm, 'hardware': 'gpu', 'pid': int(p)})
print(json.dumps(w))
" "$_cpu_j" "$_gpu_j")"
    fi
  else
    echo "Skipping workers (COLLATZ_MAC_STACK_NO_WORKER=1)."
  fi

  maybe_seed_demo_run

  echo "Starting Vite dashboard on http://127.0.0.1:5173/ (logs: $DASH_OUT) ..."
  (
    cd "$root/dashboard"
    exec npm run dev -- --host 127.0.0.1
  ) >>"$DASH_OUT" 2>>"$DASH_ERR" &
  local dashboard_pid=$!

  persist_stack_state "$backend_pid" "$dashboard_pid" "$workers_json"

  echo ""
  echo "Collatz Lab is up."
  local _api_base_for_hint
  _api_base_for_hint="$(health_url_to_api_base "$HEALTH_URL")"
  echo "  API:       ${_api_base_for_hint}/"
  echo "  Dashboard: http://127.0.0.1:5173/"
  if is_darwin; then
    echo "  Native stack (CPU .dylib + Metal): ${_api_base_for_hint}/api/workers/native-stack"
  fi
  if [[ "${COLLATZ_MAC_STACK_NO_WORKER:-}" == "1" ]]; then
    echo "  Workers:   (none — COLLATZ_MAC_STACK_NO_WORKER=1)"
  elif [[ "${COLLATZ_MAC_STACK_SINGLE_WORKER:-}" == "1" ]]; then
    echo "  Worker:    auto — logs $WORKER_OUT"
  else
    echo "  Workers:   COLLATZ_STACK_CPU_WORKERS=${COLLATZ_STACK_CPU_WORKERS:-1} CPU + COLLATZ_STACK_GPU_WORKERS=${COLLATZ_STACK_GPU_WORKERS:-1} GPU"
    echo "             logs: $RUNTIME_DIR/mac-managed-worker-cpu-*.out.log , mac-managed-worker-gpu-*.out.log"
  fi
  echo "  Logs:      $RUNTIME_DIR/*.log"
  echo "  Stop:      bash scripts/mac-dev-stack.sh stop"
}

report_dev_port_listeners() {
  local port any=0
  for port in 8000 5173; do
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      any=1
      echo "Still listening on TCP $port:"
      lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
    fi
  done
  if [[ "$any" -eq 1 ]]; then
    echo ""
    echo "If these are Collatz Lab (or stuck orphans), run:"
    echo "  bash scripts/mac-dev-stack.sh stop force"
    echo "(Kills every process listening on :8000 and :5173 — do not use if another app needs those ports.)"
  fi
}

free_dev_ports_force() {
  local port pids pid
  for port in 8000 5173; do
    pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    for pid in $pids; do
      echo "Force-stopping listener PID $pid on :$port ..."
      kill_tree "$pid"
    done
  done
  rm -f "$STATE_PATH"
}

cmd_stop() {
  local force_mode="${1:-}"
  if [[ "$force_mode" == "force" ]]; then
    echo "Force mode: stopping saved PIDs (if any), then freeing :8000 and :5173 ..."
  fi

  if [[ -f "$STATE_PATH" ]]; then
    local line bp wp dp
    line="$(read_pids_from_state)" || line=""
    if [[ -z "$line" ]]; then
      echo "warning: could not read PIDs from $STATE_PATH (remove it if stale)" >&2
      rm -f "$STATE_PATH"
    else
      IFS='|' read -ra ST <<<"$line"
      local bp="${ST[0]:-0}"
      local dp="${ST[1]:-0}"
      local idx wp
      echo "Stopping dashboard (pid $dp) ..."
      [[ "$dp" != "0" ]] && kill_tree "$dp"
      if [[ "${#ST[@]}" -gt 2 ]]; then
        local nw=$(( ${#ST[@]} - 2 ))
        echo "Stopping $nw worker PID(s) (reverse order) ..."
        for ((idx = ${#ST[@]} - 1; idx >= 2; idx--)); do
          wp="${ST[idx]}"
          [[ "$wp" != "0" ]] && kill_tree "$wp"
        done
      fi
      if [[ "$bp" != "0" ]]; then
        echo "Stopping API (pid $bp) ..."
        kill_tree "$bp"
      else
        echo "Leaving API running (was reused from outside this script; backend_pid=0 in state)."
      fi
      rm -f "$STATE_PATH"
    fi
    echo "Stopped (from state file)."
  else
    echo "No saved stack state ($STATE_PATH). Nothing to stop from PID file."
  fi

  if [[ "$force_mode" == "force" ]]; then
    free_dev_ports_force
    echo "Force stop complete."
  else
    report_dev_port_listeners
  fi
}

cmd_status() {
  if [[ ! -f "$STATE_PATH" ]]; then
    echo "No stack state file. Not started (or already stopped)."
    exit 0
  fi
  local PY="$root/.venv/bin/python"
  if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || true)"
  fi
  echo "State file: $STATE_PATH"
  if [[ -n "$PY" ]]; then
    "$PY" "$STATE_HELPER" print-status "$STATE_PATH"
  fi
  if command -v curl >/dev/null 2>&1; then
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
      echo "  API health: OK"
    else
      echo "  API health: unreachable"
    fi
  fi
}

case "$ACTION" in
  start) cmd_start ;;
  stop) cmd_stop "$STOP_FORCE_ARG" ;;
  restart)
    if [[ -f "$STATE_PATH" ]]; then
      cmd_stop "$STOP_FORCE_ARG" || true
    fi
    cmd_start
    ;;
  status) cmd_status ;;
  build-natives) cmd_build_natives ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 1
    ;;
esac

#!/usr/bin/env bash
# Build ``libsieve_descent_native.{dylib|so}`` for the lab cpu-sieve backend.
# Tries OpenMP first (Darwin: Homebrew libomp + rpath; Linux: -fopenmp), falls back to sequential.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
case "$(uname -s)" in
  Darwin) SUF=dylib ;;
  Linux) SUF=so ;;
  *)
    echo "error: unsupported OS for native cpu-sieve shared library (try numba backend)" >&2
    exit 1
    ;;
esac
OUT="libsieve_descent_native.${SUF}"

build_serial() {
  cc -std=c11 -O3 -Wall -Wextra -fPIC -shared \
    -DCOLLATZ_CPU_SIEVE_NATIVE_EXPORT -DCOLLATZ_NO_MAIN \
    -o "$OUT" sieve_descent.c
}

build_openmp_darwin() {
  local prefix="$1"
  cc -std=c11 -O3 -Wall -Wextra -fPIC -shared \
    -DCOLLATZ_CPU_SIEVE_NATIVE_EXPORT -DCOLLATZ_NO_MAIN -DCOLLATZ_CPU_SIEVE_OPENMP \
    -Xpreprocessor -fopenmp \
    -I"${prefix}/include" \
    -L"${prefix}/lib" -lomp \
    -Wl,-rpath,"${prefix}/lib" \
    -o "$OUT" sieve_descent.c
}

build_openmp_linux() {
  cc -std=c11 -O3 -Wall -Wextra -fPIC -shared \
    -DCOLLATZ_CPU_SIEVE_NATIVE_EXPORT -DCOLLATZ_NO_MAIN -DCOLLATZ_CPU_SIEVE_OPENMP \
    -fopenmp \
    -o "$OUT" sieve_descent.c
}

if [[ "$(uname -s)" == "Darwin" ]]; then
  if LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null)" && [[ -d "${LIBOMP_PREFIX}/lib" ]]; then
    if build_openmp_darwin "${LIBOMP_PREFIX}"; then
      echo "built (OpenMP/libomp) $ROOT/$OUT"
      exit 0
    fi
    echo "warning: OpenMP build failed; falling back to sequential" >&2
  fi
elif [[ "$(uname -s)" == "Linux" ]]; then
  if build_openmp_linux; then
    echo "built (OpenMP) $ROOT/$OUT"
    exit 0
  fi
  echo "warning: OpenMP build failed; falling back to sequential" >&2
fi

build_serial
echo "built (sequential, no OpenMP) $ROOT/$OUT"

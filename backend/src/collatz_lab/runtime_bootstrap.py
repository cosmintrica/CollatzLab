"""Early-process environment fixes before OpenMP / PyTorch / native libs load.

Call :func:`ensure_darwin_duplicate_openmp_ok` from CLI / API entrypoints **before**
importing submodules that pull in both Homebrew ``libomp`` (e.g. native cpu-sieve
``.dylib``) and PyTorch's bundled ``libomp``.
"""

from __future__ import annotations

import os
import sys


def ensure_darwin_duplicate_openmp_ok() -> None:
    """On macOS, tolerate two OpenMP runtimes in one process.

    Typical Collatz Lab stack: ``libsieve_descent_native.dylib`` links against
    Homebrew ``libomp.dylib``, while ``libtorch_python.dylib`` ships its own
    ``libomp``. The second runtime's init otherwise calls ``abort()`` with
    **OMP: Error #15** (duplicate libomp). LLVM documents ``KMP_DUPLICATE_LIB_OK``
    as an unsupported escape hatch; we set it only when unset so tests can override.
    """
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

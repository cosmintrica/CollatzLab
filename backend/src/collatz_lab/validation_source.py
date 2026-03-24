"""Platform-wide validation **source-of-truth** hooks for Collatz Lab.

This module is **not** macOS-specific. It documents how we establish correctness
across **all** operating systems and **all** fast compute backends (Numba,
native C / OpenMP, Metal, PyTorch MPS, CUDA): exact arithmetic and the
odd-only sieve reference are what optimized paths must match under the
documented contract.

**Principle:** no single fast backend is declared "the truth." Confidence comes
from cross-checks against reference logic and Python ``int`` (bigint) where the
contract requires exact arithmetic.

See ``docs/CORRECTNESS_AND_VALIDATION.md`` for the full protocol.
"""

from __future__ import annotations

from typing import Any

from .metrics_sot import NumberMetrics  # noqa: F401 — re-exported for callers

VALIDATION_PROTOCOL_PATH = "docs/CORRECTNESS_AND_VALIDATION.md"


def validation_contract_metadata() -> dict[str, Any]:
    """Static metadata for API, dashboards, and tooling (no heavy imports, no compute)."""
    return {
        "scope": "platform_wide",
        "applies_to": {
            "operating_systems": ["macOS", "Linux", "Windows"],
            "fast_backends": [
                "numba_cpu_sieve",
                "native_c_cpu_sieve_dylib",
                "metal_gpu_sieve",
                "mps_gpu_sieve",
                "cuda_gpu_sieve",
            ],
        },
        "protocol_documentation": VALIDATION_PROTOCOL_PATH,
        "exact_arithmetic": {
            "description": (
                "Python int descent metrics for overflow recovery and exact verification "
                "(sieve-style semantics: orbit first drops below seed)."
            ),
            "primary_function": "collatz_lab.services.metrics_descent_direct",
            "alias_function": "collatz_lab.validation_source.metrics_descent_exact",
        },
        "odd_only_sieve_reference": {
            "description": (
                "Pure Python mirror of the odd-only sieve kernel; use for parity on "
                "ranges without int64 overflow; see module docstring for limits."
            ),
            "module": "collatz_lab.sieve_reference",
            "functions": [
                "odd_sieve_descent_one",
                "odd_sieve_descent_linear_range",
            ],
        },
        "principle": (
            "Fast backends are validated against reference / bigint where applicable; "
            "none is absolute truth."
        ),
    }


def metrics_descent_exact(value: int) -> NumberMetrics:
    """Exact descent metrics using arbitrary-precision integers.

    Delegates to :func:`collatz_lab.metrics_sot.metrics_descent_direct`.
    Prefer this name in new code when the call site should read as "source-of-truth path."
    """
    from .metrics_sot import metrics_descent_direct

    return metrics_descent_direct(value)

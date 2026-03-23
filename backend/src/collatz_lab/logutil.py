"""Tame third-party log noise (Numba CUDA driver, etc.)."""

from __future__ import annotations

import logging

# Driver logs every deferred cuMemFree_v2 / pending dealloc at INFO
# (numba.cuda.cudadrv.driver._PendingDeallocs). Not application telemetry.
_NUMBA_CUDA_PARENT_LOGGERS = (
    "numba",
    "numba.cuda",
    "numba.cuda.cudadrv",
    "numba.cuda.dispatcher",
)
_NUMBA_CUDA_DRIVER = "numba.cuda.cudadrv.driver"


def silence_numba_cuda_info() -> None:
    """Quiet Numba CUDA: parents WARNING; driver ERROR so INFO dealloc spam never records.

    Call after logging.basicConfig and again before GPU work: make_logger() may set
    the driver logger to INFO when CUDA_LOG_LEVEL is set in the environment.
    """
    for name in _NUMBA_CUDA_PARENT_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger(_NUMBA_CUDA_DRIVER).setLevel(logging.ERROR)


def is_noise_log_entry(entry: dict) -> bool:
    """True for worker lines that are not useful in the dashboard (incl. pre-fix log files).

    When the user passes a search query, entries are not dropped so they can audit raw lines.
    """
    if entry.get("kind") != "worker":
        return False
    level = entry.get("level") or ""
    logger = entry.get("logger") or ""
    msg_l = (entry.get("msg") or "").lower()
    if level == "DEBUG" and (
        logger.startswith("numba") or logger.startswith("llvmlite")
    ):
        return True
    if level != "INFO":
        return False
    if _NUMBA_CUDA_DRIVER in logger or logger.endswith(".driver"):
        if "dealloc" in msg_l or "cumemfree" in msg_l.replace("_", ""):
            return True
    return False

"""Kernel names and hardware slugs (single source of truth for discovery and workers)."""

CPU_DIRECT_KERNEL = "cpu-direct"
CPU_ACCELERATED_KERNEL = "cpu-accelerated"
CPU_PARALLEL_KERNEL = "cpu-parallel"
CPU_PARALLEL_ODD_KERNEL = "cpu-parallel-odd"
CPU_SIEVE_KERNEL = "cpu-sieve"
CPU_BARINA_KERNEL = "cpu-barina"
CPU_KERNELS = [
    CPU_DIRECT_KERNEL,
    CPU_ACCELERATED_KERNEL,
    CPU_PARALLEL_KERNEL,
    CPU_PARALLEL_ODD_KERNEL,
    CPU_SIEVE_KERNEL,
    CPU_BARINA_KERNEL,
]
GPU_KERNEL = "gpu-collatz-accelerated"
GPU_SIEVE_KERNEL = "gpu-sieve"

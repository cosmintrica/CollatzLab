"""Platform-wide validation source-of-truth hooks."""

from __future__ import annotations

from dataclasses import asdict

from collatz_lab.validation_source import metrics_descent_exact, validation_contract_metadata


def test_validation_contract_metadata_shape():
    meta = validation_contract_metadata()
    assert meta["scope"] == "platform_wide"
    assert "macOS" in meta["applies_to"]["operating_systems"]
    assert "cuda_gpu_sieve" in meta["applies_to"]["fast_backends"]
    assert meta["odd_only_sieve_reference"]["module"] == "collatz_lab.sieve_reference"


def test_metrics_descent_exact_delegates_to_same_as_direct():
    from collatz_lab.services import metrics_descent_direct

    for n in (1, 3, 27):
        assert asdict(metrics_descent_exact(n)) == asdict(metrics_descent_direct(n))

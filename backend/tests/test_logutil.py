from collatz_lab.logutil import is_noise_log_entry


def test_noise_hides_cuda_dealloc_info():
    assert is_noise_log_entry(
        {
            "kind": "worker",
            "level": "INFO",
            "logger": "numba.cuda.cudadrv.driver",
            "msg": "add pending dealloc: cuMemFree_v2 524288 bytes",
        }
    )


def test_noise_keeps_cuda_errors():
    assert not is_noise_log_entry(
        {
            "kind": "worker",
            "level": "ERROR",
            "logger": "numba.cuda.cudadrv.driver",
            "msg": "something failed",
        }
    )


def test_noise_keeps_collatz_lab():
    assert not is_noise_log_entry(
        {
            "kind": "worker",
            "level": "INFO",
            "logger": "collatz_lab.worker",
            "msg": "poll",
        }
    )

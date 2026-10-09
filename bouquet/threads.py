"""One thread per process for every parallel path: TokaMaker (``nthreads``) and Python's numeric stack.

TokaMaker at ``nthreads>1`` reduces in a non-deterministic order (about 1 % jitter in li_1) and can hang
DLSODE on stiff slices, and BLAS/OpenMP pools sized to the node oversubscribe it once there is one solver
per core. Parallelism is across processes only.
"""
from __future__ import annotations

import contextlib
import os

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

_limits = None


def worker_env(env=None) -> dict:
    """The environment of a worker process: every thread pool at 1, and HDF5 file locking off (shared
    inputs are only read, and filesystems without flock, CFS among them, refuse the lock).  Updates
    ``env`` (default ``os.environ``) in place and returns it."""
    env = os.environ if env is None else env
    env.update({v: "1" for v in THREAD_VARS})
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    return env


@contextlib.contextmanager
def worker_environment():
    """:func:`worker_env` on ``os.environ`` for the duration (spawned children inherit it), then restored."""
    keys = THREAD_VARS + ("HDF5_USE_FILE_LOCKING",)
    saved = {k: os.environ.get(k) for k in keys}
    worker_env()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def pin_threads():
    """Hold this process to one thread: the environment for libraries loaded from now on, and
    threadpoolctl for the pools already loaded (the environment binds only at load time)."""
    global _limits
    from threadpoolctl import threadpool_limits
    worker_env()
    _limits = threadpool_limits(limits=1)


def thread_report() -> list:
    """``[{"library", "num_threads"}]`` for every thread pool loaded in this process."""
    from threadpoolctl import threadpool_info
    return [{"library": p.get("internal_api"), "num_threads": p.get("num_threads")}
            for p in threadpool_info()]


def require_single_thread(n, what):
    if int(n) != 1:
        raise ValueError(f"{what}={n}: parallel runs are one thread per process (TokaMaker at nthreads>1 is "
                         "not reproducible and can hang); use more workers instead")

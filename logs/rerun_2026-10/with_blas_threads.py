"""Run a script with the OpenBLAS pool of this process set to N threads (threadpoolctl), without changing its code.

In March 2026 WSL exposed the 20 logical CPUs of the host, so the OpenBLAS default was 20 threads; now it is 16, and
OPENBLAS_NUM_THREADS is capped at the visible CPUs, but openblas_set_num_threads (threadpool_limits) is not.
DCV: also export OMP/OPENBLAS/MKL_NUM_THREADS=4 so that joblib passes 4 to the workers (March: 20 // 5) while
this process (fold setup, final retrain) runs with N.

Usage (from the worktree root): python with_blas_threads.py N script.py [args...]
"""
import os
import runpy
import sys

import numpy  # noqa: F401  (loads OpenBLAS before setting the limit)
from threadpoolctl import threadpool_info, threadpool_limits

n_threads, script = int(sys.argv[1]), sys.argv[2]
_limit = threadpool_limits(n_threads)  # kept referenced for the whole run
print(f"[with_blas_threads] {[(i['internal_api'], i['num_threads']) for i in threadpool_info()]}", flush=True)
sys.argv = sys.argv[2:]
sys.path[0] = os.path.dirname(os.path.abspath(script))
runpy.run_path(script, run_name="__main__")

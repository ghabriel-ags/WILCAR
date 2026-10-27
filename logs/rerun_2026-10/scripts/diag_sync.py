"""Diagnosis of the WILCAR-C (M2) DCV on synchronous_machine, per fold and per n.

mode 'parallel': the production parallel path (_vcd_fold_worker per fold and step) executed in-process with joblib's
sequential backend; every worker call re-creates and re-seeds the model, so the result equals the multi-process run.
BLAS threads = $DIAG_THREADS (default 3, the limit loky sets in each of the 5 workers: 16 // 5; March: 4).
mode 'noparallel': the sequential path (--no-parallel).
mode 'march': parallel path without weight reuse (attribute removed after __init__), i.e. the March 2026 behaviour.
Usage (from the worktree root): python diag_sync.py <mode> <out_dir> [dataset]
"""
import os
import sys

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = os.environ.get("DIAG_THREADS", "3")

import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path[:0] = [str(Path.cwd() / "src"), str(Path.cwd())]
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import run_dynamic_cv as rdc  # noqa: E402
import wilcar.constrained as wc  # noqa: E402

mode, out = sys.argv[1], Path(sys.argv[2])
ds = sys.argv[3] if len(sys.argv) > 3 else "synchronous_machine"
out.mkdir(parents=True, exist_ok=True)
steps, calls, ctx = [], [], {"phase": "other"}


def h(param):
    if param is None:
        return None
    return hashlib.sha1(b"".join(np.ascontiguousarray(param[k]).tobytes() for k in ("W1", "b1", "W2", "b2"))).hexdigest()[:12]


orig_minimize = wc.minimize


def minimize_spy(fun, x0, *a, **k):
    res = orig_minimize(fun, x0, *a, **k)
    calls.append({**ctx, "n_params": len(x0), "success": bool(res.success), "status": int(res.status),
                  "message": str(res.message), "nit": int(res.nit), "fun": float(res.fun)})
    return res


wc.minimize = minimize_spy
orig_opt = wc.WILCARConstrainedNumpy._optimize_single_attempt


def opt_spy(self, x, y, n_neurons):
    ctx["phase"] = "step"
    ctx["attempt"] = ctx.get("attempt", 0) + 1
    try:
        return orig_opt(self, x, y, n_neurons)
    finally:
        ctx["phase"] = "other"


wc.WILCARConstrainedNumpy._optimize_single_attempt = opt_spy
orig_cs = wc.WILCARConstrainedNumpy.constructive_step


def cs_spy(self, n, x_train, y_train, x_test, y_test):
    fold = self.config.seed // 100
    prev = getattr(self, "_vcd_previous_param", None)
    ctx.update(fold=fold, n=n, attempt=0)
    res = orig_cs(self, n, x_train, y_train, x_test, y_test)
    mine = [c for c in calls if c.get("fold") == fold and c.get("n") == n and c["phase"] == "step"]
    last = mine[-1] if mine else {}
    steps.append({
        "fold": fold, "n": n, "success": res["success"], "n_attempts": res.get("n_attempts"),
        "r2_train": res.get("r2_train"), "mse_train": res.get("mse_train"),
        "r2_test": res.get("r2_test"), "mse_test": res.get("mse_test"), "scr": res.get("scr"),
        "prev_neurons": None if prev is None else prev["W1"].shape[0], "prev_hash": h(prev), "out_hash": h(self.param),
        "slsqp_calls": len(mine), "slsqp_success_count": sum(c["success"] for c in mine),
        "accepted_success": last.get("success"), "accepted_status": last.get("status"),
        "accepted_message": last.get("message"), "accepted_nit": last.get("nit"), "accepted_fun": last.get("fun"),
    })
    return res


wc.WILCARConstrainedNumpy.constructive_step = cs_spy
if mode == "rngcarry":  # variant (c): each fold carries its NumPy RNG state between steps
    sys.path.insert(0, str(Path(__file__).parent))
    import rngcarry
    rdc._vcd_fold_worker = rngcarry.vcd_fold_worker_rng
if mode == "march":
    orig_init = wc.WILCARConstrainedNumpy.__init__

    def init_without_state(self, *a, **k):
        orig_init(self, *a, **k)
        del self._vcd_previous_param

    wc.WILCARConstrainedNumpy.__init__ = init_without_state

inputs, targets = rdc.load_raw_data(ds, ".")
task = "classification" if ds in rdc.ALL_CLASSIFICATION_DATASETS else "regression"
cfg = rdc.VCDConfig(task=task, methods=[2], datasets=[ds])
with joblib.parallel_config(backend="sequential"):
    res = rdc.run_vcd_for_method(2, inputs, targets, rdc.EXPECTED_SIGNALS[ds], cfg, rdc.get_vcd_method_config(ds, 2),
                                 ds, out, parallel=(mode != "noparallel"), skip_retrain=True)
pd.DataFrame(steps).to_csv(out / f"steps_{mode}.csv", index=False)
pd.DataFrame(calls).to_csv(out / f"slsqp_calls_{mode}.csv", index=False)
json.dump({"n_star": res.optimal_neurons, "J_T": [h_["J_T"] for h_ in res.history],
           "n": [h_["n"] for h_ in res.history]}, open(out / f"history_{mode}.json", "w"))
print(f"DONE {mode} n*={res.optimal_neurons} steps={len(res.history)}", file=sys.__stdout__, flush=True)

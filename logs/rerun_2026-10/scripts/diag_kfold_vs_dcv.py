"""Read-only comparison: fold 0 of the WILCAR-C K-fold (cv_20260310_140954) vs fold 0 of the WILCAR-C DCV on
synchronous_machine. Checks data, partition, normalization, seed, linearization initial weights, the n = 1 solution
(and its SLSQP status / attempts) and the K-fold trajectory R2(n). Default BLAS threads (as the March K-fold run).
Usage (from the worktree root): python diag_kfold_vs_dcv.py"""
import csv
import sys
from pathlib import Path

sys.path[:0] = [str(Path.cwd() / "src"), str(Path.cwd())]
import numpy as np  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
import run_cross_validation as rcv  # noqa: E402
import run_dynamic_cv as rdc  # noqa: E402
import wilcar.constrained as wc  # noqa: E402

ds = "synchronous_machine"
P = lambda *a: print(*a, file=sys.__stdout__, flush=True)  # noqa: E731

# --- data, partition, normalization
Xc, yc = rcv.load_raw_data(ds, ".")
Xd, yd = rdc.load_raw_data(ds, ".")
P("raw data identical:", np.array_equal(Xc, Xd) and np.array_equal(yc, yd))
tr_c, te_c = next(iter(KFold(n_splits=5, shuffle=True, random_state=rcv.CVConfig().seed).split(Xc)))
tr_d, te_d = next(iter(KFold(n_splits=5, shuffle=True, random_state=rdc.VCDConfig().seed).split(Xd)))
P("fold-0 partition identical:", np.array_equal(tr_c, tr_d) and np.array_equal(te_c, te_d))
nc = rcv.normalize_data(Xc[tr_c], Xc[te_c], yc[tr_c], yc[te_c], task="regression")
nd = rdc.normalize_data(Xd[tr_d], Xd[te_d], yd[tr_d], yd[te_d], task="regression")
P("fold-0 normalized data identical:", all(np.array_equal(a, b) for a, b in zip(nc, nd)))
Xtr, Xte, ytr, yte = nc

# --- configs and seeds
cv_cfg = rcv.CVConfig(task="regression")
kf_tc = rcv.create_training_config(cv_cfg, rcv.get_method_config(ds, 2))
kf_tc.seed = cv_cfg.seed + 0 * 100 + 2
dcv_cfg = rdc.VCDConfig(task="regression")
dcv_tc = rdc.create_training_config(dcv_cfg, rdc.get_vcd_method_config(ds, 2))
dcv_tc.seed = dcv_cfg.seed + 0 * 100
P(f"seed: K-fold {kf_tc.seed}, DCV {dcv_tc.seed}")
same_cfg = {k: (getattr(kf_tc, k), getattr(dcv_tc, k)) for k in ("max_neurons", "patience_constructive",
            "delta_perturbation", "tolerance", "task") if getattr(kf_tc, k) != getattr(dcv_tc, k)}
P("other config differences:", same_cfg or "none")

# --- SLSQP spy and attempt counts
calls, attempts = [], {}
orig_min = wc.minimize


def min_spy(fun, x0, *a, **k):
    r = orig_min(fun, x0, *a, **k)
    calls.append((len(x0), bool(r.success), int(r.status), str(r.message), int(r.nit), float(r.fun)))
    return r


wc.minimize = min_spy
orig_tsm = wc.WILCARConstrainedNumpy._train_single_model


def tsm_spy(self, x_train, y_train, x_test, n_neurons, previous_param):
    out = orig_tsm(self, x_train, y_train, x_test, n_neurons, previous_param)
    attempts.setdefault(id(self), {})[n_neurons] = (out[0], out[2])
    return out


wc.WILCARConstrainedNumpy._train_single_model = tsm_spy


def model(seed):
    tc = rcv.create_training_config(cv_cfg, rcv.get_method_config(ds, 2))
    tc.seed = seed
    import io, contextlib  # noqa: E401
    with contextlib.redirect_stdout(io.StringIO()):
        return wc.WILCARConstrainedNumpy(Xtr, ytr, Xte, yte, rcv.EXPECTED_SIGNALS[ds], tc, ds)


def n1(m):
    """Train only n = 1 as both paths do (_train_single_model with previous_param=None)."""
    x, y = m.train_inputs.T, m.train_targets.reshape(1, -1)
    calls.clear()
    ok, _, att = m._train_single_model(x, y, m.test_inputs.T, 1, None)
    from utils.metrics import calculate_all_metrics
    r2_tr = calculate_all_metrics(y.flatten(), m.predict(x).flatten())[2]
    r2_te = calculate_all_metrics(m.test_targets, m.predict(m.test_inputs.T).flatten())[2]
    return ok, att, r2_tr, r2_te, calls[-1]


m_kf, m_dcv = model(2), model(0)
P("linearization initial weights identical (seed 2 vs 0):", np.array_equal(m_kf.initial_weights, m_dcv.initial_weights))
for name, seed in (("K-fold path (seed 2)", 2), ("DCV path (seed 0)", 0)):
    ok, att, r2tr, r2te, last = n1(model(seed))
    P(f"n=1 {name}: success={ok} attempts={att} R2_train={r2tr:.4f} R2_test={r2te:.4f} | accepted SLSQP: "
      f"success={last[1]} status={last[2]} nit={last[4]} fun={last[5]:.6f} msg='{last[3]}'")

# --- full K-fold fold 0 (fold 1 in the CSV) trajectory
m = model(2)
import io, contextlib  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    res = m.train()
acc = attempts[id(m)]
ok_n = [n for n in sorted(acc) if acc[n][0]]
P("K-fold fold 0 trajectory (accepted n: R2_train / R2_test):",
  " ".join(f"{n}:{a:.3f}/{b:.3f}" for n, a, b in zip(ok_n, res.r2_train, res.r2_test)))
P(f"K-fold fold 0: n*={res.best_neurons} R2_test*={res.best_test_r2:.5f}; skipped sizes: "
  f"{[n for n in sorted(acc) if not acc[n][0]]}")
march = next(r for r in csv.DictReader(open("/home/ghabr/projects/thesis/results/cross_validation/cv_20260310_140954/"
                                            "synchronous_machine/fold_results.csv")) if r["method_id"] == "2" and r["fold"] == "1")
P(f"March K-fold fold 1 (=fold 0): n*={march['best_neurons']} R2_test={float(march['test_r2']):.5f}")

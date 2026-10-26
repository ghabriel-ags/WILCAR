"""DCV of WILCAR-C (M2): weight reuse between constructive steps and the rule for discarded steps.

In the parallel path (the default, used in March 2026) the worker recreates the model at every step n and
restored _vcd_previous_param only if the fresh model already had that attribute, which it never had, so M2
re-initialized the whole network at every n. Expected behaviour:
  * from n = 2 on, _train_single_model receives the parameters of the last accepted step;
  * fold states advance only when step n is accepted (all K folds reach 100% SCR); after a discarded step
    every fold keeps the state of the last accepted n, in the parallel and in the sequential path.
joblib is switched to the threading backend so the spy sees the worker calls.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_dynamic_cv as rdc  # noqa: E402
from wilcar.constrained import WILCARConstrainedNumpy  # noqa: E402


def synthetic():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(80, 3))
    y = x[:, 0] + 0.5 * x[:, 1] + 0.05 * rng.normal(size=80)
    return x, y, [+1, +1, 0]


def run_dcv(monkeypatch, tmp_path, parallel, max_n=3, fail=None):
    """Run the M2 DCV on synthetic data; record (fold, n, neurons of previous_param) per _train_single_model call.
    fail = (n, fold): that fold reports no 100% SCR at size n (after training, like a real failure)."""
    monkeypatch.setattr(sys, "stdout", sys.stdout)  # workers redirect stdout from threads; restore it afterwards
    calls = []
    orig = WILCARConstrainedNumpy._train_single_model

    def spy(self, x_train, y_train, x_test, n_neurons, previous_param):
        fold = self.config.seed // 100 if parallel else self._fold_for_test
        calls.append((fold, n_neurons, None if previous_param is None else previous_param["W1"].shape[0]))
        out = orig(self, x_train, y_train, x_test, n_neurons, previous_param)
        if fail == (n_neurons, fold):
            return False, out[1], out[2]
        return out

    monkeypatch.setattr(WILCARConstrainedNumpy, "_train_single_model", spy)
    if not parallel:  # sequential path: tag the K models in creation order
        created = []
        orig_init = WILCARConstrainedNumpy.__init__

        def init(self, *a, **k):
            orig_init(self, *a, **k)
            self._fold_for_test = len(created)
            created.append(self)

        monkeypatch.setattr(WILCARConstrainedNumpy, "__init__", init)

    x, y, signals = synthetic()
    cfg = rdc.VCDConfig(task="regression", methods=[2], datasets=["synthetic"])
    mc = rdc.VCDMethodConfig(max_neurons=max_n, patience=10)
    with joblib.parallel_config(backend="threading"):
        res = rdc.run_vcd_for_method(2, x, y, signals, cfg, mc, "synthetic", tmp_path,
                                     parallel=parallel, skip_retrain=True)
    return calls, res


def previous_sizes(calls, n):
    return {fold: prev for fold, size, prev in calls if size == n}


def test_parallel_worker_reuses_weights(monkeypatch, tmp_path):
    calls, res = run_dcv(monkeypatch, tmp_path, parallel=True)
    assert [h["n"] for h in res.history] == [1, 2, 3]
    assert previous_sizes(calls, 1) == {k: None for k in range(5)}
    assert previous_sizes(calls, 2) == {k: 1 for k in range(5)}
    assert previous_sizes(calls, 3) == {k: 2 for k in range(5)}


@pytest.mark.parametrize("parallel", [True, False])
def test_discarded_step_keeps_last_accepted_state(monkeypatch, tmp_path, parallel):
    calls, res = run_dcv(monkeypatch, tmp_path, parallel=parallel, max_n=4, fail=(2, 1))
    assert [h["n"] for h in res.history] == [1, 3, 4]  # n = 2 discarded
    assert set(previous_sizes(calls, 3).values()) == {1}  # every fold grows from the last accepted n = 1
    assert set(previous_sizes(calls, 4).values()) == {3}
    assert len(previous_sizes(calls, 3)) == 5

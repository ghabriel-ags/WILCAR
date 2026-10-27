"""Run one (dataset, method, fold) exactly as run_cross_validation.run_cv_for_dataset does, without try/except.
Usage (from the worktree root): python run_one_fold.py <dataset> <method_id> <fold 1-based> [task]
Prints the fold result as JSON on the last line."""
import json, os, sys, time
if os.environ.get("BLOCK_TORCH") == "1":
    sys.modules["torch"] = None  # reproduce an environment without torch (March 2026)
from pathlib import Path
sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "src")]
import numpy as np
from sklearn.model_selection import KFold
import run_cross_validation as rcv
if os.environ.get("BLAS_THREADS"):  # e.g. 20 = March K-fold default (WSL then exposed 20 CPUs); the env var
    from threadpoolctl import threadpool_limits  # OPENBLAS_NUM_THREADS is capped at the 16 visible CPUs
    _blas_limit = threadpool_limits(int(os.environ["BLAS_THREADS"]))
from threadpoolctl import threadpool_info  # noqa: E402
print("THREADPOOL " + json.dumps(threadpool_info()), flush=True)

ds, mid, fold = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
task = sys.argv[4] if len(sys.argv) > 4 else "classification"
cv_config = rcv.CVConfig(task=task, methods=[mid], datasets=[ds])
inputs, targets = rcv.load_raw_data(ds, ".")
kfold = KFold(n_splits=cv_config.n_folds, shuffle=cv_config.shuffle, random_state=cv_config.seed)
train_idx, test_idx = list(kfold.split(inputs))[fold - 1]
Xtr, Xte, ytr, yte = rcv.normalize_data(inputs[train_idx], inputs[test_idx], targets[train_idx], targets[test_idx], task=task)
mc = rcv.get_method_config(ds, mid)
tc = rcv.create_training_config(cv_config, mc)
tc.seed = cv_config.seed + (fold - 1) * 100 + mid
factory = rcv.get_method_factories()[mid]
if os.environ.get("UNFIXED") == "1":  # restore _expand_network as of commit 0e54bfe (before the fix)
    import copy
    from wilcar.constrained import WILCARConstrainedNumpy
    def _expand_network_march(self, previous_param, n_neurons):
        new_param = copy.deepcopy(previous_param)
        new_w1 = np.random.randn(1, self.n_inputs) * np.sqrt(1. / self.n_inputs)
        new_param['W1'] = np.vstack([new_param['W1'], new_w1])
        new_param['b1'] = np.vstack([new_param['b1'], np.zeros((1, 1))])
        new_w2 = np.random.randn(1, 1) * np.sqrt(1. / n_neurons)
        new_param['W2'] = np.hstack([new_param['W2'], new_w2])
        return new_param
    WILCARConstrainedNumpy._expand_network = _expand_network_march
model = factory(train_inputs=Xtr, train_targets=ytr, test_inputs=Xte, test_targets=yte,
                expected_signals=rcv.EXPECTED_SIGNALS[ds], config=tc, dataset_name=ds)
t0 = time.time()
res = model.train()
out = {"fold": fold, "best_neurons": res.best_neurons, "test_r2": res.best_test_r2, "test_mse": res.best_test_mse,
       "train_r2": res.best_train_r2, "scr": res.best_conformity, "time": time.time() - t0}
if task == "classification":
    out.update(test_accuracy=res.best_test_accuracy, test_f1=res.best_test_f1, test_mcc=res.best_test_mcc)
print("RESULT " + json.dumps(out))

"""Run one (dataset, method, fold) of the K-fold as run_cross_validation.run_cv_for_dataset does and save its record.

Same setup as run_one_fold.py (KFold split, normalization, method config, seed + 100 * fold_idx + method), but the
model is trained by run_cross_validation.run_method itself and the per-fold record is completed with the same fields
as in run_cv_for_dataset (fold, dataset, max_neurons_config, patience_config). The record is saved as a pickle, to be
aggregated with the other folds by compose_kfold_run.py, and printed as JSON on the last line.

Usage (from the worktree root, BLAS threads set by the caller, e.g. with_blas_threads.py 20):
    python run_fold_record.py <dataset> <method_id> <fold 1-based> <out.pkl> [task]
"""
import json
import pickle
import sys
from pathlib import Path

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "src")]
from sklearn.model_selection import KFold  # noqa: E402
from threadpoolctl import threadpool_info  # noqa: E402

import run_cross_validation as rcv  # noqa: E402

print("THREADPOOL " + json.dumps(threadpool_info()), flush=True)

ds, method_id, fold, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
task = sys.argv[5] if len(sys.argv) > 5 else "classification"
cv_config = rcv.CVConfig(task=task, methods=[method_id], datasets=[ds])
inputs, targets = rcv.load_raw_data(ds, ".")
kfold = KFold(n_splits=cv_config.n_folds, shuffle=cv_config.shuffle, random_state=cv_config.seed)
fold_idx = fold - 1
train_idx, test_idx = list(kfold.split(inputs))[fold_idx]
X_train_norm, X_test_norm, y_train_norm, y_test_norm = rcv.normalize_data(
    inputs[train_idx], inputs[test_idx], targets[train_idx], targets[test_idx], task=cv_config.task)

method_config = rcv.get_method_config(ds, method_id)
print(f"Running {rcv.METHOD_NAMES[method_id]} fold {fold} (n≤{method_config.max_neurons}, p={method_config.patience})",
      flush=True)
training_config = rcv.create_training_config(cv_config, method_config)
training_config.seed = cv_config.seed + fold_idx * 100 + method_id
result = rcv.run_method(method_id=method_id, train_inputs=X_train_norm, train_targets=y_train_norm,
                        test_inputs=X_test_norm, test_targets=y_test_norm,
                        expected_signals=rcv.EXPECTED_SIGNALS[ds], config=training_config, dataset_name=ds)
result['fold'] = fold_idx + 1
result['dataset'] = ds
result['max_neurons_config'] = method_config.max_neurons
result['patience_config'] = method_config.patience

out.write_bytes(pickle.dumps(result))
print("RESULT " + json.dumps(result, default=str))

"""Compose a K-fold run from per-fold records and write its outputs with run_cross_validation's own code.

The records come from the fold_results_partial.csv of a run (the incremental save of run_cv_for_dataset: one row per
method and fold, conformity_details included) and from run_fold_record.py pickles. run_cross_validation.main() is then
executed with run_method replaced by a lookup of these records, selected by the per-fold seed
(seed + 100 * fold_idx + method), so that fold_results.csv, conformity_details.csv, summary_stats.csv,
detailed_results.json, consolidated_results.csv, latex_tables.tex and cv_config.json come from the runner's code.

Usage (from a worktree root; the run folder is created there and then renamed to OUT_DIR):
    python compose_kfold_run.py OUT_DIR [--partial CSV:1,2,3 ...] [--pickle FILE.pkl ...] -- <run_cross_validation args>
"""
import argparse
import ast
import copy
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "src")]
import run_cross_validation as rcv  # noqa: E402

# Fields that run_cv_for_dataset adds to the run_method record
ADDED = ("fold", "dataset", "max_neurons_config", "patience_config")
INT_FIELDS = ("method_id", "best_neurons")

ap = argparse.ArgumentParser()
ap.add_argument("out_dir", type=Path)
ap.add_argument("--partial", action="append", default=[], help="fold_results_partial.csv:<folds, comma separated>")
ap.add_argument("--pickle", action="append", default=[], help="record saved by run_fold_record.py")
args, runner_args = ap.parse_known_args()
runner_args = [a for a in runner_args if a != "--"]
assert not args.out_dir.exists(), f"{args.out_dir} exists"
assert "--seed" not in runner_args and "--resume" not in runner_args and "--skip-existing" not in runner_args

records, added = {}, {}
for spec in args.partial:
    path, folds = spec.rsplit(":", 1)
    df = pd.read_csv(path, float_precision="round_trip")
    assert "error" not in df.columns, f"{path} has failed folds"
    for row in df[df.fold.isin([int(f) for f in folds.split(",")])].to_dict("records"):
        rec = {}
        for k, v in row.items():
            if k == "conformity_details":
                v = ast.literal_eval(v)
            elif k in INT_FIELDS or k in ("fold", "max_neurons_config", "patience_config"):
                v = int(v)
            elif isinstance(v, float):
                v = float(v)
            rec[k] = v
        key = (rec["method_id"], rec["fold"])
        assert key not in records, f"duplicate record {key}"
        added[key] = {k: rec.pop(k) for k in ADDED}
        records[key] = rec
for path in args.pickle:
    rec = pickle.loads(Path(path).read_bytes())
    key = (rec["method_id"], rec["fold"])
    assert key not in records, f"duplicate record {key}"
    added[key] = {k: rec.pop(k) for k in ADDED}
    records[key] = rec
print(f"[compose] records (method, fold): {sorted(records)}", flush=True)

SEED = 0  # run_cross_validation default (--seed not allowed above)
used = []


def run_method_from_records(method_id, train_inputs, train_targets, test_inputs, test_targets, expected_signals,
                            config, dataset_name):
    fold = (config.seed - SEED - method_id) // 100 + 1
    used.append((method_id, fold))
    return copy.deepcopy(records[(method_id, fold)])


rcv.run_method = run_method_from_records
before = set((Path.cwd() / "results" / "cross_validation").glob("cv_*"))
sys.argv = ["run_cross_validation.py", *runner_args]
rcv.main()
created = sorted(set((Path.cwd() / "results" / "cross_validation").glob("cv_*")) - before)
assert len(created) == 1, created
assert sorted(used) == sorted(records), (used, sorted(records))

# The fields added by run_cv_for_dataset must equal those of the source records
for ds_dir in created[0].iterdir():
    if ds_dir.is_dir():
        fr = pd.read_csv(ds_dir / "fold_results.csv")
        for r in fr.to_dict("records"):
            src = added[(r["method_id"], r["fold"])]
            assert all(r[k] == src[k] for k in ADDED), (r, src)
created[0].rename(args.out_dir)
print(f"[compose] written to {args.out_dir}")

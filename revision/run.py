"""
Nested-CV campaign runner (resumable).

Examples
  python -m revision.run --datasets pima --methods WILCAR-C LR-C --reps 1 --jobs 2
  python -m revision.run --all --reps 5 --jobs 18            # full campaign (20-core machine)
Results: one JSON line per (dataset, method, selection, rep, fold) in results/revision/results.jsonl
"""
import argparse, json, os, time, traceback
from dataclasses import asdict
from pathlib import Path
from joblib import Parallel, delayed

from .data import DATASETS, load
from .methods import METHODS, Cfg
from .protocol import run_fold, BASELINES

OUT = Path(__file__).resolve().parents[1] / "results" / "revision"


def task(ds, method, sel, rep, fold, cfg, out):
    X, y, s = load(ds)
    try:
        row = run_fold(X, y, s, method, sel, rep, fold, cfg=cfg)
        row["dataset"] = ds
    except Exception as e:  # keep the campaign going; the error is logged
        row = {"dataset": ds, "method": method, "selection": sel, "rep": rep, "fold": fold,
               "error": repr(e), "trace": traceback.format_exc()[-2000:]}
    with open(out, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--methods", nargs="*", default=list(METHODS) + list(BASELINES))
    ap.add_argument("--dcv", action="store_true", help="also run DCV for WILCAR / WILCAR-C")
    ap.add_argument("--all", action="store_true", help="all datasets, methods and DCV")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--mode", default="mean", choices=["mean", "multi", "global"])
    ap.add_argument("--zero-out", action="store_true")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--fast", action="store_true", help="short configuration for smoke tests")
    a = ap.parse_args()
    if a.all:
        a.dcv = True
    cfg = Cfg(constraint_mode=a.mode, zero_out=a.zero_out)
    if a.fast:
        cfg = Cfg(constraint_mode=a.mode, zero_out=a.zero_out, max_neurons=40, patience=10,
                  max_neurons_c=10, patience_c=4, epochs=600, es_patience=100, max_reinit=10)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"results_{a.tag}.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            if "error" not in r:
                done.add((r["dataset"], r["method"], r["selection"], r["rep"], r["fold"]))
    jobs = []
    for ds in a.datasets:
        for m in a.methods:
            sels = ["holdout"] + (["dcv"] if a.dcv and m.startswith("WILCAR") else [])
            for sel in sels:
                for rep in range(a.reps):
                    for fold in range(a.folds):
                        if (ds, m, sel, rep, fold) not in done:
                            jobs.append((ds, m, sel, rep, fold))
    (OUT / f"config_{a.tag}.json").write_text(json.dumps({"cfg": asdict(cfg), "args": vars(a)}, indent=1))
    print(f"{len(jobs)} tasks -> {out}  ({a.jobs} workers)")
    t0 = time.time()
    Parallel(n_jobs=a.jobs, verbose=10)(delayed(task)(*j, cfg, out) for j in jobs)
    print(f"done in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()

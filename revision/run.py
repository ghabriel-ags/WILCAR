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

from .data import ORIGINAL, ALL, BENCHMARKS, SYN, SYNTHETIC, load, true_prob
from .methods import METHODS, Cfg
from .protocol import run_fold, evaluate, BASELINES

OUT = Path(__file__).resolve().parents[1] / "results" / "revision"


def task(ds, method, sel, rep, fold, cfg, out):
    try:
        if fold < 0:                     # official train/test split of a benchmark, rep = seed
            (Xtr, ytr), (Xte, yte), s = load(ds, split=True)
            row = {**evaluate(Xtr, ytr, Xte, yte, s, method, sel, seed=rep, cfg=cfg), "rep": rep, "fold": fold}
        else:
            X, y, s = load(ds)
            row = run_fold(X, y, s, method, sel, rep, fold, cfg=cfg,
                           ptrue=true_prob(ds) if ds in SYNTHETIC else None)
        row["dataset"] = ds
    except Exception as e:  # keep the campaign going; the error is logged
        row = {"dataset": ds, "method": method, "selection": sel, "rep": rep, "fold": fold,
               "error": repr(e), "trace": traceback.format_exc()[-2000:]}
    with open(out, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(ORIGINAL), choices=list(ALL) + list(SYN))
    ap.add_argument("--methods", nargs="*", default=list(METHODS) + list(BASELINES))
    ap.add_argument("--dcv", action="store_true", help="also run DCV for WILCAR / WILCAR-C")
    ap.add_argument("--dcv-methods", nargs="*", default=["WILCAR", "WILCAR-C"],
                    help="methods that also get DCV selection (e.g. RIXM RIXM-C to test the weight-reuse claim)")
    ap.add_argument("--all", action="store_true", help="all datasets (incl. benchmarks), methods and DCV")
    ap.add_argument("--syn", action="store_true", help="synthetic ground-truth datasets only")
    ap.add_argument("--lit", action="store_true", help="benchmarks only, official train/test split, --reps seeds")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--mode", default="mean", choices=["mean", "multi", "global", "cegis"])
    ap.add_argument("--time-budget", type=float, default=3600.0, help="seconds per constructive selection loop")
    ap.add_argument("--zero-out", action="store_true")
    ap.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "gd"], help="unconstrained training")
    ap.add_argument("--l2", type=float, default=0.3, help="prior precision lambda0 (penalty lambda0/N on the mean BCE), WILCAR/RIXM")
    ap.add_argument("--elm-no-bias", action="store_true", help="dissertation ELM (zero hidden biases, no output bias)")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--fast", action="store_true", help="short configuration for smoke tests")
    a = ap.parse_args()
    if a.all:
        a.dcv = True
        if a.datasets == list(ORIGINAL):
            a.datasets = list(ALL)
    if a.syn:
        a.datasets = list(SYN)
    if a.lit:
        a.datasets = [d for d in a.datasets if d in BENCHMARKS] or list(BENCHMARKS)
    for d in a.datasets:                 # fail fast (e.g. benchmarks not downloaded)
        load(d)
    base = dict(constraint_mode=a.mode, zero_out=a.zero_out, optimizer=a.optimizer, elm_bias=not a.elm_no_bias, l2=a.l2,
                time_budget=a.time_budget)
    cfg = Cfg(**base)
    if a.fast:
        cfg = Cfg(**base, max_neurons=40, patience=10, max_neurons_c=10, patience_c=4, epochs=600,
                  es_patience=100, max_reinit=10, lbfgs_maxiter=300, bb_budget=50_000, cegis_rounds=5)
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
            sels = ["holdout"] + (["dcv"] if a.dcv and m in a.dcv_methods else [])
            for sel in sels:
                for rep in range(a.reps):
                    for fold in ([-1] if a.lit else range(a.folds)):
                        if (ds, m, sel, rep, fold) not in done:
                            jobs.append((ds, m, sel, rep, fold))
    (OUT / f"config_{a.tag}.json").write_text(json.dumps({"cfg": asdict(cfg), "args": vars(a)}, indent=1))
    print(f"{len(jobs)} tasks -> {out}  ({a.jobs} workers)")
    t0 = time.time()
    Parallel(n_jobs=a.jobs, verbose=10)(delayed(task)(*j, cfg, out) for j in jobs)
    print(f"done in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()

"""
Ablation of the certificate (Algorithm 2 of the paper): which bounding components decide how many networks.

    python -m revision.cert_bench            # -> results/revision/cert_bench.csv (+ summary printed)

Population: SLFNs trained on the real datasets at several sizes, (i) without constraints, (ii) with the mean-gain
constraint and (iii) in the cegis mode -- the three kinds of networks the certificate meets in the experiments.
Each network is checked by four variants of the branch and bound with the same node budget:
  interval : natural interval extension only
  +mv      : + mean-value form with gradient enclosure
  +mono    : + monotonicity test (full input-space search)
  full     : + pre-activation-space search when n < p (the configuration used in the experiments)
Soundness is cross-checked by dense sampling: no certified network may show a sampled violation.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .data import load, ORIGINAL
from .methods import Model, Cfg
from . import certify

OUT = Path(__file__).resolve().parents[1] / "results" / "revision"
VARIANTS = {"interval": dict(mv=False, mono=False, zspace=False), "+mv": dict(mv=True, mono=False, zspace=False),
            "+mono": dict(mv=True, mono=True, zspace=False), "full": dict(mv=True, mono=True, zspace=True)}


def population(datasets, sizes, seeds, rng_eval):
    for ds in datasets:
        X, y, s = load(ds)
        X = MinMaxScaler().fit_transform(X)
        if len(X) > 2000:
            idx = np.random.default_rng(0).choice(len(X), 2000, replace=False); X, y = X[idx], y[idx]
        for kind, name, mode in [("free", "RIXM", "mean"), ("mean", "RIXM-C", "mean"), ("cegis", "RIXM-C", "cegis")]:
            for n in sizes:
                for seed in seeds:
                    cfg = Cfg(constraint_mode=mode, max_reinit=3, cegis_rounds=5, max_anchors_mean=300)
                    m = Model(name, s, cfg, seed=1000 * seed + n)
                    An = m.anchors(X) if m.constrained else None
                    t0 = time.time()
                    th, info = m.step(n, X, y, None, An)
                    yield dict(dataset=ds, kind=kind, n=n, seed=seed, p=X.shape[1], train_s=time.time() - t0), th, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(ORIGINAL) + ["heart_disease", "compas"])
    ap.add_argument("--sizes", nargs="*", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--budget", type=int, default=200_000)
    a = ap.parse_args()
    rows, out = [], OUT / "cert_bench.jsonl"
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l); done.add((r["dataset"], r["kind"], r["n"], r["seed"], r["variant"]))
    rng = np.random.default_rng(0)
    for meta, th, s in population(a.datasets, a.sizes, a.seeds, rng):
        if all((meta["dataset"], meta["kind"], meta["n"], meta["seed"], v) in done for v in VARIANTS):
            continue
        n, p = meta["n"], meta["p"]
        U = rng.random((50_000, p))
        W1, b1, idx, C = certify.coefficients(th, n, p, s)
        sampled_viol = bool(len(idx)) and bool(((certify._dsig(U @ W1.T + b1) @ C.T) < 0).any())
        for v, kw in VARIANTS.items():
            t0 = time.time()
            st, _, nodes = certify.certify(th, n, p, s, budget=a.budget, **kw)
            r = {**meta, "variant": v, "status": st, "nodes": int(nodes), "time_s": time.time() - t0,
                 "sampled_violation": sampled_viol}
            assert not (st == "certified" and sampled_viol), r         # soundness cross-check
            with open(out, "a") as fh:
                fh.write(json.dumps(r) + "\n")
        print(meta["dataset"], meta["kind"], n, flush=True)
    df = pd.DataFrame([json.loads(l) for l in open(out)])
    df.to_csv(OUT / "cert_bench.csv", index=False)
    summ = df.groupby(["kind", "variant"]).agg(
        nets=("status", "size"), certified=("status", lambda x: 100 * np.mean(x == "certified")),
        counterexample=("status", lambda x: 100 * np.mean(x == "counterexample")),
        unknown=("status", lambda x: 100 * np.mean(x == "unknown")),
        median_s=("time_s", "median"), p90_s=("time_s", lambda x: np.percentile(x, 90)), median_nodes=("nodes", "median"))
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()

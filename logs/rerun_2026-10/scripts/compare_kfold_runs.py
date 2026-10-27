"""compare_kfold_runs.py OFFICIAL_RUN NEW_RUN: per dataset, method and fold, are the fold_results.csv rows identical
(all columns except training_time)?"""
import sys
from pathlib import Path
import pandas as pd
off, new = Path(sys.argv[1]), Path(sys.argv[2])
tag = new.parent.parent.parent.name
for f in sorted(off.glob("*/fold_results.csv")):
    n = new / f.parent.name / "fold_results.csv"
    if not n.exists():
        print(tag, f.parent.name, "MISSING"); continue
    a, b = (pd.read_csv(p, float_precision="round_trip").drop(columns=["training_time"]).set_index(["method_id", "fold"])
            for p in (f, n))
    for key in a.index:
        if key not in b.index:
            print(tag, f.parent.name, key, "MISSING"); continue
        ra, rb = a.loc[key], b.loc[key]
        diff = [c for c in a.columns if not (ra[c] == rb[c] or (pd.isna(ra[c]) and pd.isna(rb[c])))]
        print(tag, f.parent.name, f"M{key[0]} fold {key[1]}:", "IDENTICAL" if not diff else
              "DIFFERENT " + "; ".join(f"{c}: {ra[c]} -> {rb[c]}" for c in diff[:4]))

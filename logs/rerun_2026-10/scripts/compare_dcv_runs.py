"""compare.py OFFICIAL_RUN NEW_RUN: per dataset/method, are vcd_history.csv, fold_details_at_optimal.csv and
vcd_summary.json identical except timing columns?"""
import json, sys
from pathlib import Path
import pandas as pd
off, new = Path(sys.argv[1]), Path(sys.argv[2])
for h in sorted(off.glob("*/method_*/vcd_history.csv")):
    rel = h.parent.relative_to(off); n = new / rel
    tag = f"{new.parent.parent.parent.name}:{rel}"
    if not (n / "vcd_history.csv").exists():
        print(tag, "MISSING"); continue
    diffs = []
    for f in ("vcd_history.csv", "fold_details_at_optimal.csv"):
        a = pd.read_csv(h.parent / f, float_precision="round_trip"); b = pd.read_csv(n / f, float_precision="round_trip")
        a = a.drop(columns=[c for c in a if "time" in c]); b = b.drop(columns=[c for c in b if "time" in c])
        if a.shape != b.shape: diffs.append(f"{f}: shape {a.shape} vs {b.shape}"); continue
        if not a.equals(b):
            col = next(c for c in a if not a[c].equals(b[c])); row = int((a[col] != b[col]).idxmax())
            diffs.append(f"{f}: first diff col={col} row={row} ({a[col][row]!r} vs {b[col][row]!r})")
    ja, jb = (json.loads((p / "vcd_summary.json").read_text()) for p in (h.parent, n))
    ks = [k for k in ja if "time" not in k and ja[k] != jb.get(k)]
    if ks: diffs.append("summary: " + ", ".join(f"{k} {ja[k]} vs {jb.get(k)}" for k in ks))
    print(tag, "IDENTICAL" if not diffs else "DIFFERENT | " + " | ".join(diffs))

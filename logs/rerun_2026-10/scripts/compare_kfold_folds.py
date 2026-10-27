"""Compare per-fold WILCAR-C results of the verification runs with the March fold_results.csv (exact equality)."""
import csv, glob, json, re, sys
from pathlib import Path
V = Path(sys.argv[1]); R = Path("/home/ghabr/projects/thesis/results/cross_validation")
MARCH = {"ch": (R / "cv_20260310_140954/computer_hardware/fold_results.csv"),
         "br": (R / "cv_20260321_015941/breast__binary_/fold_results.csv")}
FIELDS = ["best_neurons", "test_r2", "test_mse", "train_r2", "scr", "test_accuracy", "test_f1", "test_mcc"]
ok_all = True
for log in sorted(V.glob("*_f*.log")):
    ds, variant, fold = re.match(r"(\w+?)_(\w+)_f(\d)\.log", log.name).groups()
    m = re.search(r"^RESULT (.*)$", log.read_text(), re.M)
    if not m:
        print(f"{log.name}: no RESULT yet"); ok_all = False; continue
    new = json.loads(m.group(1))
    old = next(r for r in csv.DictReader(open(MARCH[ds])) if r["method_id"] == "2" and int(r["fold"]) == int(fold))
    diffs = []
    for k in FIELDS:
        if k in new and k in old and old[k] != "":
            a, b = float(old[k]), float(new[k])
            if a != b:
                diffs.append(f"{k}: {a!r} -> {b!r}")
    ok_all &= not diffs
    print(f"{log.name}: {'IDENTICAL' if not diffs else 'DIFFERENT ' + '; '.join(diffs)} (n*={new['best_neurons']}, t={new['time']:.0f}s)")
print("ALL IDENTICAL" if ok_all else "NOT ALL IDENTICAL")

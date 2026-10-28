"""compare_fold_record.py PARTIAL_CSV RECORD.pkl: is the run_fold_record.py record identical to the row of the same
method and fold in a run's fold_results_partial.csv (all fields except training_time, conformity_details included)?"""
import ast
import pickle
import sys
from pathlib import Path

import pandas as pd

rec = pickle.loads(Path(sys.argv[2]).read_bytes())
df = pd.read_csv(sys.argv[1], float_precision="round_trip")
rows = df[(df.method_id == rec["method_id"]) & (df.fold == rec["fold"])].to_dict("records")
assert len(rows) == 1, f"{len(rows)} rows for method {rec['method_id']} fold {rec['fold']}"
row = rows[0]
row["conformity_details"] = ast.literal_eval(row["conformity_details"])
assert list(row) == list(rec), (list(row), list(rec))
diff = [k for k in rec if k != "training_time" and row[k] != rec[k]]
print(f"M{rec['method_id']} fold {rec['fold']}:", "IDENTICAL" if not diff else
      "DIFFERENT " + "; ".join(f"{k}: {row[k]} -> {rec[k]}" for k in diff))
print(f"training_time: run {row['training_time']:.1f} s, record {rec['training_time']:.1f} s")
sys.exit(1 if diff else 0)

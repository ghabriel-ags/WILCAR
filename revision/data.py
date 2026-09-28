"""Binary classification datasets of Appendix B (raw CSVs in data/raw)."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASETS = {
    # key: (file, has_header, expected gain signs)  -- signals as in src/utils/data.py / Appendix B
    "algerian": ("Algerian_forest_fires__binary_.csv", False, [+1, -1, 0, -1, +1, +1, +1, +1, +1, +1]),
    "breast_original": ("breast__binary_.csv", False, [+1] * 9),
    "breast_diagnostic": ("Diagnostic Breast Cancer__binary_.csv", False, [+1] * 9 + [0] + [+1] * 9 + [0] + [+1] * 9 + [0]),
    "heart_failure": ("heart_failure_clinical_records_dataset__binary_.csv", False, [+1, +1, 0, 0, -1, +1, 0, +1, -1, 0, 0, -1]),
    "pima": ("pima__binary_.csv", True, [+1, +1, 0, +1, 0, +1, +1, +1]),
}


def load(key):
    fname, header, signals = DATASETS[key]
    df = pd.read_csv(RAW / fname, sep=";", header=0 if header else None).dropna()
    v = df.values.astype(float)
    X, y = v[:, :-1], v[:, -1].astype(int)
    assert X.shape[1] == len(signals), (key, X.shape, len(signals))
    return X, y, signals

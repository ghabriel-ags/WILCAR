"""
Binary classification datasets.

* Appendix B datasets of the dissertation (raw CSVs in data/raw).
* Monotonicity benchmarks of the monotone-NN literature (Liu et al., 2020; Sivaraman et al., 2020;
  Nolte et al., 2023; Runje & Shankaranarayana, 2023): COMPAS, Heart Disease and Loan Defaulter, with the
  same preprocessing, train/test split and monotone features as the public Zenodo release
  (record 7968969) used by CMNN. Download them once with `python -m revision.fetch_benchmarks`.
  For cross-validation the official train and test files are concatenated; `load(key, split=True)`
  returns the official split instead (literature-comparable protocol).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data"
RAW = ROOT / "raw"
BENCH = ROOT / "benchmarks"

DATASETS = {
    # key: (file, has_header, expected gain signs)  -- signals as in src/utils/data.py / Appendix B
    "algerian": ("Algerian_forest_fires__binary_.csv", False, [+1, -1, 0, -1, +1, +1, +1, +1, +1, +1]),
    "breast_original": ("breast__binary_.csv", False, [+1] * 9),
    "breast_diagnostic": ("Diagnostic Breast Cancer__binary_.csv", False, [+1] * 9 + [0] + [+1] * 9 + [0] + [+1] * 9 + [0]),
    "heart_failure": ("heart_failure_clinical_records_dataset__binary_.csv", False, [+1, +1, 0, 0, -1, +1, 0, +1, -1, 0, 0, -1]),
    "pima": ("pima__binary_.csv", True, [+1, +1, 0, +1, 0, +1, +1, +1]),
}

# key: (zenodo name, signs by column name or by position, max samples used for CV (None = all))
BENCHMARKS = {
    "compas": ("compas", {"priors_count": 1, "juv_fel_count": 1, "juv_misd_count": 1, "juv_other_count": 1}, None),
    "heart_disease": ("heart", {"trestbps": 1, "chol": 1}, None),
    "loan": ("loan", [-1, 1, -1, -1, 1] + [0] * 23, 20_000),
}

ORIGINAL = tuple(DATASETS)
ALL = ORIGINAL + tuple(BENCHMARKS)


def _bench_frames(name):
    fr = []
    for prefix in ("train", "test"):
        f = BENCH / f"{prefix}_{name}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found - run `python -m revision.fetch_benchmarks` first")
        df = pd.read_csv(f)
        df.columns = [c.replace(" ", "_") for c in df.columns]
        fr.append(df.dropna())
    return fr


def _bench_xy(df, signs):
    cols = [c for c in df.columns if c != "ground_truth"]
    if isinstance(signs, dict):
        s = [int(signs.get(c, 0)) for c in cols]
        missing = set(signs) - set(cols)
        assert not missing, f"monotone columns not found: {missing}"
    else:
        s = list(signs); assert len(s) == len(cols), (len(s), len(cols))
    return df[cols].values.astype(float), df["ground_truth"].values.astype(int), s


def load(key, split=False, seed=0):
    if key in DATASETS:
        fname, header, signals = DATASETS[key]
        df = pd.read_csv(RAW / fname, sep=";", header=0 if header else None).dropna()
        v = df.values.astype(float)
        X, y = v[:, :-1], v[:, -1].astype(int)
        assert X.shape[1] == len(signals), (key, X.shape, len(signals))
        assert not split, "official split only exists for the benchmarks"
        return X, y, signals
    name, signs, max_n = BENCHMARKS[key]
    tr, te = _bench_frames(name)
    if split:
        Xtr, ytr, s = _bench_xy(tr, signs)
        Xte, yte, _ = _bench_xy(te, signs)
        if max_n and len(Xtr) > max_n:
            Xtr, ytr = _subsample(Xtr, ytr, max_n, seed)
        return (Xtr, ytr), (Xte, yte), s
    X, y, s = _bench_xy(pd.concat([tr, te], ignore_index=True), signs)
    if max_n and len(X) > max_n:
        X, y = _subsample(X, y, max_n, seed)
    return X, y, s


def _subsample(X, y, n, seed):
    from sklearn.model_selection import train_test_split
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return X[idx], y[idx]

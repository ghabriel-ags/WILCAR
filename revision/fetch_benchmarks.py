"""
Download the monotonicity benchmarks (COMPAS, Heart Disease, Loan Defaulter) released on Zenodo
(record 7968969, used by Runje & Shankaranarayana, ICML 2023) into data/benchmarks/.

  python -m revision.fetch_benchmarks
"""
import urllib.request
from .data import BENCH, BENCHMARKS

URL = "https://zenodo.org/record/7968969/files/{}"


def main():
    BENCH.mkdir(parents=True, exist_ok=True)
    for name, *_ in BENCHMARKS.values():
        for prefix in ("train", "test"):
            fn = f"{prefix}_{name}.csv"
            dst = BENCH / fn
            if dst.exists():
                print("exists  ", dst); continue
            print("download", fn)
            urllib.request.urlretrieve(URL.format(fn), dst)
    from .data import load
    for key in BENCHMARKS:
        X, y, s = load(key)
        print(f"{key:14s} X={X.shape}  positives={y.mean():.3f}  constrained={sum(1 for v in s if v)}")


if __name__ == "__main__":
    main()

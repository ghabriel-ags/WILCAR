"""
Remove finished tasks of some methods from a results file, so that the next run of the same command recomputes them
(e.g. after a change in the constrained training).

    python -m revision.purge --tag main --methods WILCAR-C RIXM-C ELM-C
A backup results_<tag>.jsonl.bak is written first.
"""
import argparse, json, shutil
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "revision"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    a = ap.parse_args()
    f = OUT / f"results_{a.tag}.jsonl"
    shutil.copy(f, f.with_suffix(".jsonl.bak"))
    rows = [l for l in open(f)]
    keep = [l for l in rows if json.loads(l).get("method") not in set(a.methods)]
    f.write_text("".join(keep))
    print(f"{f.name}: removed {len(rows) - len(keep)} of {len(rows)} lines (backup: {f.name}.bak)")


if __name__ == "__main__":
    main()

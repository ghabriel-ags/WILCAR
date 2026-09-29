"""
Statistics and tables for the Neural Networks revision.

  python -m revision.analyze --tag main            # reads results/revision/results_main.jsonl

Outputs (results/revision/<tag>/):
  summary.csv          mean ± std over the 25 outer folds, per dataset × method(selection)
  ranks.csv            average MCC rank per method (Friedman over datasets)
  friedman.txt         Friedman chi² and Iman–Davenport F, Holm-corrected pairwise tests vs. the best method
  wilcoxon_pairs.csv   constrained vs. unconstrained counterpart, paired over folds, per dataset: Wilcoxon and the
                       corrected resampled t-test of Nadeau & Bengio (2003), both Holm-corrected
  cd_diagram.png       critical-difference diagram (Nemenyi, alpha = 0.05)
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1] / "results" / "revision"
K_OUT = 5
PAIRS = [("WILCAR", "WILCAR-C"), ("RIXM", "RIXM-C"), ("ELM", "ELM-C"), ("LR", "LR-C"),
         ("XGB", "XGB-C"), ("LGBM", "LGBM-C"), ("MLP", "MINMAX"), ("MLP", "CMNN"), ("MLP", "LMN"),
         ("WILCAR-C", "CMNN"), ("WILCAR-C", "LMN")]
# Nemenyi critical values q_0.05 (Demšar, 2006), k = 2..20
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def load(tag):
    rows = [json.loads(l) for l in open(ROOT / f"results_{tag}.jsonl")]
    bad = [r for r in rows if "error" in r]
    if bad:
        print(f"{len(bad)} failed tasks (see 'error' field)")
    df = pd.DataFrame([r for r in rows if "error" not in r])
    df["label"] = np.where(df["selection"] == "dcv", df["method"] + " (DCV)", df["method"])
    return df


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main"); ap.add_argument("--metric", default="mcc")
    a = ap.parse_args()
    out = ROOT / a.tag; out.mkdir(parents=True, exist_ok=True)
    df = load(a.tag)
    cols = [c for c in ["mcc", "acc", "f1", "auc", "brier", "viol_test", "viol_domain", "vars_violated_domain",
                        "certified", "scr_legacy", "n_star", "time_s"] if c in df]
    g = df.groupby(["dataset", "label"])[cols]
    summ = g.mean().round(4).join(g.std().round(4), rsuffix="_std")
    summ.to_csv(out / "summary.csv")
    print(summ[[c for c in ["mcc", "mcc_std", "viol_domain", "certified", "n_star", "time_s"] if c in summ]].to_string())
    if "cert_status" in df:
        df.groupby(["dataset", "label"])["cert_status"].value_counts().unstack(fill_value=0).to_csv(out / "certification.csv")

    # ---- Friedman over datasets on mean metric
    piv = df.groupby(["dataset", "label"])[a.metric].mean().unstack()
    piv = piv.dropna(axis=1)
    ranks = piv.rank(axis=1, ascending=False) if a.metric != "brier" else piv.rank(axis=1, ascending=True)
    R = ranks.mean().sort_values(); R.to_csv(out / "ranks.csv")
    N, k = piv.shape
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    Ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) != chi2 else np.inf
    p_ff = 1 - stats.f.cdf(Ff, k - 1, (k - 1) * (N - 1))
    se = np.sqrt(k * (k + 1) / (6 * N))
    best = R.index[0]
    z = (R - R[best]) / se
    pv = pd.Series(2 * (1 - stats.norm.cdf(np.abs(z.drop(best).values))), index=z.drop(best).index)
    with open(out / "friedman.txt", "w") as fh:
        fh.write(f"datasets N={N}, methods k={k}\nFriedman chi2={chi2:.3f}; Iman-Davenport F={Ff:.3f}, p={p_ff:.4g}\n")
        fh.write(f"Nemenyi CD (alpha=0.05) = {Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * N)):.3f}\n")
        fh.write("Holm-corrected tests vs. best (" + best + "):\n")
        for m, p_, pa in zip(pv.index, pv.values, holm(pv.values)):
            fh.write(f"  {m:20s} rank={R[m]:.2f}  p={p_:.4f}  p_holm={pa:.4f}\n")
    print(open(out / "friedman.txt").read())

    # ---- paired Wilcoxon, constrained vs unconstrained, per dataset
    recs = []
    for ds, d in df[df.selection == "holdout"].groupby("dataset"):
        for u, c in PAIRS:
            a_ = d[d.method == u].set_index(["rep", "fold"])[a.metric]
            b_ = d[d.method == c].set_index(["rep", "fold"])[a.metric]
            j = a_.index.intersection(b_.index)
            if len(j) < 5:
                continue
            diff = (b_[j] - a_[j]).values
            try:
                p = stats.wilcoxon(diff, zero_method="zsplit").pvalue
            except ValueError:
                p = 1.0
            # corrected resampled t-test (Nadeau & Bengio, 2003): variance inflated by n_test/n_train = 1/(k-1)
            J, sd = len(diff), diff.std(ddof=1)
            t = diff.mean() / np.sqrt((1 / J + 1 / (K_OUT - 1)) * sd ** 2) if sd > 0 else 0.0
            p_nb = 2 * (1 - stats.t.cdf(abs(t), J - 1)) if sd > 0 else 1.0
            recs.append({"dataset": ds, "pair": f"{c} vs {u}", "mean_diff": diff.mean(), "median_diff": np.median(diff),
                         "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "p": p, "p_nb": p_nb})
    w = pd.DataFrame(recs)
    if len(w):
        w["p_holm"] = holm(w["p"].values)
        w["p_nb_holm"] = holm(w["p_nb"].values)
        w.round(4).to_csv(out / "wilcoxon_pairs.csv", index=False); print(w.round(4).to_string())

    # ---- CD diagram
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cd = Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6 * N))
        fig, ax = plt.subplots(figsize=(8, 0.35 * k + 1.2))
        for i, (m, r) in enumerate(R.items()):
            ax.plot([r, r], [0, -(i + 1)], color="0.6", lw=0.8); ax.plot(r, 0, "ko", ms=3)
            ax.text(r, -(i + 1), f" {m} ({r:.2f})", va="center", fontsize=8)
        ax.plot([1, 1 + cd], [0.6, 0.6], "k-", lw=2); ax.text(1 + cd / 2, 0.8, f"CD = {cd:.2f}", ha="center", fontsize=8)
        ax.set_xlim(0.8, k + 0.5); ax.set_ylim(-(k + 1), 1.2); ax.set_yticks([]); ax.set_xlabel(f"average rank ({a.metric.upper()})")
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(out / "cd_diagram.png", dpi=200)
    except Exception as e:
        print("CD diagram skipped:", e)


if __name__ == "__main__":
    main()

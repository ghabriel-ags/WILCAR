"""
Generate the LaTeX tables and number macros of the manuscript from the revision campaign.

    python paper/make_tables.py                       # reads results/revision/results_<tag>.jsonl
    python paper/make_tables.py --map main=smoke3     # use another tag for a role (testing)

Every table degrades gracefully: a missing tag yields "--" cells and a "pending" note, so the manuscript
always compiles. Outputs: paper/tables/*.tex and paper/tables/numbers.tex.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "revision"
OUT = HERE / "tables"
sys.path.insert(0, str(ROOT))

DS = [("algerian", "ALG"), ("breast_original", "BCO"), ("breast_diagnostic", "BCD"), ("heart_failure", "HF"),
      ("pima", "PIM"), ("heart_disease", "HD"), ("compas", "COM"), ("loan", "LOAN")]
ROLES = ["main", "cegis", "global", "multi", "lit", "l2zero", "gd", "zeroout", "dcvall"]
STAR = r"$^{\star}$"

# (role, method, selection, display) -- rows of the main comparison
MAIN_ROWS = [
    ("cegis", "WILCAR-C", "dcv", "WILCAR-C" + STAR + " (DCV)"),
    ("cegis", "WILCAR-C", "holdout", "WILCAR-C" + STAR),
    ("cegis", "RIXM-C", "holdout", "RIXM-C" + STAR),
    ("main", "WILCAR-C", "holdout", "WILCAR-C (mean)"),
    ("main", "WILCAR", "dcv", "WILCAR (DCV)"),
    ("main", "WILCAR", "holdout", "WILCAR"),
    ("main", "RIXM", "holdout", "RIXM"),
    ("main", "ELM", "holdout", "ELM"),
    ("main", "MLP", "holdout", "MLP"),
    ("main", "LR-C", "holdout", "LR-C"),
    ("main", "XGB-C", "holdout", "XGB-C"),
    ("main", "LGBM-C", "holdout", "LGBM-C"),
    ("main", "MINMAX", "holdout", "Min-Max"),
    ("main", "CMNN", "holdout", "CMNN"),
    ("main", "LMN", "holdout", "LMN"),
]

# published test accuracies (%) on the official splits
PUBLISHED = {  # method: (COMPAS, Heart Disease, Loan) ; source key
    "XGBoost": (("68.5", "0.1"), None, ("63.7", "0.1"), "Runje2023"),
    "Min-Max Net": (("67.8", "0.1"), ("75", "4"), ("64.9", "0.1"), "Runje2023"),
    "DLN": (("67.9", "0.3"), ("86", "2"), ("65.1", "0.2"), "Runje2023"),
    "Certified MNN": (("68.8", "0.2"), None, ("65.2", "0.1"), "Liu2020"),
    "COMET": (None, ("86", "3"), None, "Sivaraman2020"),
    "LMN": (("69.3", "0.1"), ("89.6", "1.9"), ("65.44", "0.03"), "Nolte2023"),
    "CMNN": (("69.2", "0.2"), ("89", "0"), ("65.3", "0.01"), "Runje2023"),
}


def load(tag):
    f = RES / f"results_{tag}.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in open(f)]
    df = pd.DataFrame([r for r in rows if "error" not in r])
    return df if len(df) else None


def fmt(m, s=None, d=3):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    return f"{m:.{d}f}" + (rf"\,{{\scriptsize({s:.{d - 1}f})}}" if s is not None and not np.isnan(s) else "")


def cell_stats(df, method, sel, ds, col):
    if df is None:
        return None, None
    d = df[(df.method == method) & (df.selection == sel) & (df.dataset == ds)]
    if not len(d):
        return None, None
    return float(d[col].mean()), float(d[col].std(ddof=1)) if len(d) > 1 else float("nan")


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text)
    print("wrote", OUT / name)


# ------------------------------------------------------------------------------------------ tables
def table_datasets():
    from revision.data import DATASETS, BENCHMARKS, load as dload
    info = {"algerian": ("Algerian forest fires", "Fire ecology", r"\citet{Abid2020}"),
            "breast_original": ("Breast cancer (original)", "Oncology", r"\citet{Wolberg1990}"),
            "breast_diagnostic": ("Breast cancer (diagnostic)", "Oncology", r"\citet{Street1993}"),
            "heart_failure": ("Heart failure records", "Cardiology", r"\citet{Ahmad2017}"),
            "pima": ("Pima Indians diabetes", "Endocrinology", r"\citet{Smith1988}"),
            "heart_disease": ("Heart disease (Cleveland)", "Cardiology", r"\citet{Detrano1989}"),
            "compas": ("COMPAS recidivism", "Criminal justice", r"\citet{Angwin2016}"),
            "loan": ("Loan defaulter", "Credit risk", r"\citet{LendingClub}")}
    lines = []
    for key, short in DS:
        name, dom, cite = info[key]
        try:
            X, y, s = dload(key)
            n, p, c, pos = len(y), X.shape[1], sum(1 for v in s if v), 100 * y.mean()
            extra = r"$^{a}$" if key == "loan" else ""
            lines.append(f"{short} & {name} {cite} & {dom} & {n}{extra} & {p} & {c} ({100 * c / p:.0f}\\%) & {pos:.1f} \\\\")
        except Exception:
            lines.append(f"{short} & {name} {cite} & {dom} & -- & -- & -- & -- \\\\")
    write("datasets.tex", "\n".join([
        r"\begin{table}[t]\centering\small",
        r"\caption{Datasets. $n$: samples used for cross-validation; $p$: inputs; constrained: inputs with a prescribed gain sign $s_i\neq0$ (constraint ratio); positive: share of the positive class. The last three are the monotonicity benchmarks of \citet{Liu2020,Sivaraman2020,Runje2023}, with their preprocessing and monotone features.}",
        r"\label{tab:datasets}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{llllrrr}\toprule",
        r"ID & Dataset & Domain & $n$ & $p$ & Constrained & Positive (\%) \\\midrule",
        *lines,
        r"\bottomrule\end{tabular}}",
        r"\par\smallskip\footnotesize $^{a}$Stratified subsample of the 20\,000 rows used for cross-validation (the full official split is used in Section~\ref{sec:lit}).",
        r"\end{table}"]))


def table_main(dfs, metric="mcc", name="main_mcc.tex", caption_metric="MCC"):
    avail = [r for r in MAIN_ROWS]
    M = pd.DataFrame(index=[r[3] for r in avail], columns=[k for k, _ in DS], dtype=float)
    S = M.copy()
    for role, meth, sel, disp in avail:
        for key, _ in DS:
            m, s = cell_stats(dfs.get(role), meth, sel, key, metric)
            M.loc[disp, key] = np.nan if m is None else m
            S.loc[disp, key] = np.nan if s is None else s
    complete = M.dropna(axis=1, how="any")
    ranks = complete.rank(axis=0, ascending=(metric == "brier")).mean(axis=1) if complete.shape[1] else None
    lines = []
    for disp in M.index:
        cells = []
        for key, _ in DS:
            m, s = M.loc[disp, key], S.loc[disp, key]
            col = M[key].dropna()
            best = len(col) and ((m == col.max()) if metric != "brier" else (m == col.min()))
            t = fmt(None if np.isnan(m) else m, s)
            cells.append(rf"\textbf{{{t}}}" if best and not np.isnan(m) else t)
        r = "--" if ranks is None or np.isnan(ranks.get(disp, np.nan)) else f"{ranks[disp]:.2f}"
        lines.append(disp + " & " + " & ".join(cells) + f" & {r} \\\\")
        if disp.startswith("RIXM-C" + STAR) or disp == "RIXM" or disp == "ELM":
            lines.append(r"\midrule")
    pending = [role for role in {r[0] for r in avail} if dfs.get(role) is None]
    note = (r"\par\smallskip\footnotesize\textit{Pending runs: " + ", ".join(sorted(pending)) + "}.") if pending else ""
    write(name, "\n".join([
        r"\begin{table*}[t]\centering",
        rf"\caption{{Test {caption_metric} (mean and standard deviation over the outer folds; $5\times5$-fold nested cross-validation, "
        r"$5\times2$ for COM and LOAN). $\star$: counterexample-guided constraints with branch-and-bound certificate (Section~\ref{sec:cegis}). "
        r"Best value per dataset in bold; last column: average rank over the datasets with complete results (Friedman ranks).}",
        rf"\label{{tab:{name.split('.')[0]}}}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{l" + "c" * (len(DS) + 1) + r"}\toprule",
        "Method & " + " & ".join(s for _, s in DS) + r" & Rank \\\midrule",
        *(lines[:-1] if lines and lines[-1] == r"\midrule" else lines),
        r"\bottomrule\end{tabular}}", note, r"\end{table*}"]))
    return M, ranks


CERT_ROWS = [("main", "mean"), ("multi", "multi"), ("global", "global"), ("cegis", "cegis")]


def table_cert(dfs):
    lines = []
    for role, mode in CERT_ROWS:
        df = dfs.get(role)
        for meth, sel, disp in [("WILCAR-C", "dcv", "WILCAR-C (DCV)"), ("WILCAR-C", "holdout", "WILCAR-C"),
                                ("RIXM-C", "holdout", "RIXM-C"), ("ELM-C", "holdout", "ELM-C")]:
            if df is None:
                lines.append(f"{mode} & {disp} & -- & -- & -- & -- & -- & -- \\\\"); continue
            d = df[(df.method == meth) & (df.selection == sel)]
            if not len(d):
                continue
            mcc = d.groupby("dataset").mcc.mean().mean()
            st = d.get("cert_status", pd.Series(["none"] * len(d)))
            cert = 100 * np.mean(st.isin(["certified", "by_construction"]))
            unk = 100 * np.mean(st == "unknown")
            lines.append(f"{mode} & {disp} & {mcc:.3f} & {100 * d.viol_test.mean():.2f} & {100 * d.viol_domain.mean():.2f} & "
                         f"{cert:.0f} & {unk:.0f} & {d.n_star.mean():.1f} \\\\")
        lines.append(r"\midrule")
    df = dfs.get("main")
    for meth, sel, disp in [("WILCAR", "holdout", "WILCAR"), ("RIXM", "holdout", "RIXM"), ("MLP", "holdout", "MLP"),
                            ("XGB", "holdout", "XGB"), ("LGBM", "holdout", "LGBM")]:
        if df is None:
            lines.append(f"free & {disp} & -- & -- & -- & -- & -- & -- \\\\"); continue
        d = df[(df.method == meth) & (df.selection == sel)]
        if not len(d):
            continue
        st = d.get("cert_status", pd.Series(["none"] * len(d)))
        cert = "--" if (st == "none").all() else f"{100 * np.mean(st == 'certified'):.0f}"
        ns = "--" if d.n_star.isna().all() else f"{d.n_star.mean():.1f}"
        lines.append(f"free & {disp} & {d.groupby('dataset').mcc.mean().mean():.3f} & {100 * d.viol_test.mean():.2f} & "
                     f"{100 * d.viol_domain.mean():.2f} & {cert} & -- & {ns} \\\\")
    write("certification.tex", "\n".join([
        r"\begin{table}[t]\centering\small",
        r"\caption{Monotonicity by constraint mode, pooled over datasets and outer folds. MCC: mean over datasets; "
        r"viol.: share of (point, constrained input) pairs with the wrong gain sign on test points and on $10^4$ uniform points of $[0,1]^p$; "
        r"cert.: share of final models certified monotone on $[0,1]^p$ by Algorithm~\ref{alg:bb} (ELM-C: by construction in the global and cegis modes); "
        r"unk.: budget exhausted; $n^\ast$: selected hidden units.}",
        r"\label{tab:cert}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrr}\toprule",
        r"Mode & Method & MCC & viol.\ test (\%) & viol.\ domain (\%) & cert.\ (\%) & unk.\ (\%) & $n^\ast$ \\\midrule",
        *(lines[:-1] if lines[-1] == r"\midrule" else lines),
        r"\bottomrule\end{tabular}}\end{table}"]))


def table_lit(dfs):
    df = dfs.get("lit")
    keys = [("compas", "COMPAS"), ("heart_disease", "Heart Disease"), ("loan", "Loan Defaulter")]
    lines = []
    for name, (c, h, l, src) in PUBLISHED.items():
        cells = [("--" if v is None else f"{v[0]} ({v[1]})") for v in (c, h, l)]
        lines.append(f"{name} \\citep{{{src}}} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    ours = [("WILCAR-C", "dcv", "WILCAR-C" + STAR + " (DCV)"), ("WILCAR-C", "holdout", "WILCAR-C" + STAR),
            ("RIXM-C", "holdout", "RIXM-C" + STAR), ("WILCAR", "holdout", "WILCAR"), ("RIXM", "holdout", "RIXM"),
            ("MLP", "holdout", "MLP"), ("XGB-C", "holdout", "XGB-C"), ("LGBM-C", "holdout", "LGBM-C"),
            ("MINMAX", "holdout", "Min-Max (ours)"), ("CMNN", "holdout", "CMNN (ours)"), ("LMN", "holdout", "LMN (ours)")]
    for meth, sel, disp in ours:
        cells = []
        for k, _ in keys:
            m, s = cell_stats(df, meth, sel, k, "acc")
            cells.append("--" if m is None else f"{100 * m:.1f} ({100 * s:.1f})" if not np.isnan(s) else f"{100 * m:.1f}")
        lines.append(f"{disp} & " + " & ".join(cells) + r" \\")
    write("literature.tex", "\n".join([
        r"\begin{table}[t]\centering\small",
        r"\caption{Test accuracy (\%) on the official train/test splits of the monotonicity benchmarks. Upper block: values reported in the cited papers "
        r"(mean and standard deviation of the best five of ten runs for CMNN). Lower block: this work, five seeds, hyper-parameters selected on a validation split "
        r"of the official training set; ``(ours)'' marks our runs of the published architectures under the same protocol.}",
        r"\label{tab:lit}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{lccc}\toprule",
        r"Method & " + " & ".join(n for _, n in keys) + r" \\\midrule",
        *lines, r"\bottomrule\end{tabular}}",
        "" if df is not None else r"\par\smallskip\footnotesize\textit{Pending run: lit}.",
        r"\end{table}"]))


def table_ablation(dfs):
    orig = [k for k, _ in DS[:5]]
    rows = [("main", "L-BFGS, $\\lambda_0=0.3$ (default)", {}), ("l2zero", "L-BFGS, $\\lambda_0=0$", {}),
            ("gd", "gradient descent, $\\lambda_0=0$ (dissertation)", {}), ("zeroout", "new unit starts at $v_{n}=0$", {}),
            ("dcvall", "DCV without weight reuse", {})]
    lines = []
    for role, disp, _ in rows:
        df = dfs.get(role)
        cells = []
        for meth, sel in [("WILCAR", "holdout"), ("WILCAR", "dcv"), ("RIXM", "holdout"), ("RIXM", "dcv"),
                          ("WILCAR-C", "holdout"), ("RIXM-C", "holdout")]:
            if df is None:
                cells.append("--"); continue
            d = df[(df.method == meth) & (df.selection == sel) & (df.dataset.isin(orig))]
            cells.append("--" if not len(d) else f"{d.groupby('dataset').mcc.mean().mean():.3f}")
        lines.append(disp + " & " + " & ".join(cells) + r" \\")
    write("ablation.tex", "\n".join([
        r"\begin{table}[t]\centering\small",
        r"\caption{Ablations on the five original datasets: test MCC averaged over datasets (constrained variants in the dissertation's mean-gain mode). "
        r"H: hold-out selection; DCV: dynamic cross-validation.}",
        r"\label{tab:ablation}",
        r"\resizebox{\textwidth}{!}{\begin{tabular}{lcccccc}\toprule",
        r"Setting & WILCAR (H) & WILCAR (DCV) & RIXM (H) & RIXM (DCV) & WILCAR-C (H) & RIXM-C (H) \\\midrule",
        *lines, r"\bottomrule\end{tabular}}\end{table}"]))


def numbers(dfs, M, ranks):
    def mac(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}"
    out = []
    ce = dfs.get("cegis")
    if ce is not None and "cert_status" in ce:
        d = ce[ce.method.isin(["WILCAR-C", "RIXM-C"])]
        out.append(mac("CegisCertified", f"{100 * np.mean(d.cert_status == 'certified'):.1f}\\%"))
        out.append(mac("CegisUnknown", f"{100 * np.mean(d.cert_status == 'unknown'):.1f}\\%"))
    mn = dfs.get("main")
    if mn is not None and "cert_status" in mn:
        d = mn[mn.method.isin(["WILCAR-C", "RIXM-C"])]
        out.append(mac("MeanCertified", f"{100 * np.mean(d.cert_status == 'certified'):.1f}\\%"))
        out.append(mac("MeanViolDomain", f"{100 * d.viol_domain.mean():.2f}\\%"))
        d = mn[mn.method.isin(["WILCAR", "RIXM"])]
        out.append(mac("FreeViolDomain", f"{100 * d.viol_domain.mean():.1f}\\%"))
    if ranks is not None and len(ranks):
        out.append(mac("BestRankMethod", ranks.idxmin()))
    write("numbers.tex", "\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", nargs="*", default=[], help="role=tag overrides")
    a = ap.parse_args()
    tags = {r: r for r in ROLES}
    tags.update(dict(m.split("=") for m in a.map))
    dfs = {r: load(t) for r, t in tags.items()}
    print({r: (None if d is None else len(d)) for r, d in dfs.items()})
    table_datasets()
    M, ranks = table_main(dfs, "mcc", "main_mcc.tex", "MCC")
    table_main(dfs, "auc", "main_auc.tex", "AUC")
    table_main(dfs, "brier", "main_brier.tex", "Brier score")
    table_cert(dfs)
    table_lit(dfs)
    table_ablation(dfs)
    numbers(dfs, M, ranks)


if __name__ == "__main__":
    main()

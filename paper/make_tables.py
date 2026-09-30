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
ROLES = ["main", "cegis", "global", "multi", "lit", "l2zero", "gd", "zeroout", "dcvall", "syn", "syn_cegis", "syn_global"]
STAR = r"$^{\star}$"

SELNAME = {"holdout": "hold-out", "dcv": "DCV", "dcv1se": "DCV-1SE"}
MODEL = {"MINMAX": "Min-Max", "LR-C": "LR", "XGB-C": "XGBoost", "LGBM-C": "LightGBM", "XGB": "XGBoost", "LGBM": "LightGBM",
         "WILCAR-C": "WILCAR", "RIXM-C": "RIXM", "ELM-C": "ELM"}


def lab(method, constraint, sel):
    """Row label as three table cells: model & constraint & selection."""
    return f"{MODEL.get(method, method)} & {constraint} & {SELNAME.get(sel, sel)}"


# full comparison (Supplementary Material): (title, [(role, method, selection, constraint)])
MAIN_GROUPS = [
    ("Certified SLFNs, counterexample-guided (this work)", [("cegis", "WILCAR-C", "dcv", "cegis"), ("cegis", "WILCAR-C", "holdout", "cegis"),
                                                            ("cegis", "RIXM-C", "holdout", "cegis")]),
    ("SLFNs with sign constraints", [("global", "WILCAR-C", "dcv", "sign"), ("global", "WILCAR-C", "holdout", "sign"),
                                     ("global", "RIXM-C", "holdout", "sign")]),
    ("SLFNs with anchor constraints", [("multi", "WILCAR-C", "dcv", "anchor"), ("multi", "WILCAR-C", "holdout", "anchor"),
                                       ("multi", "RIXM-C", "holdout", "anchor")]),
    ("SLFNs with the average-gain constraint of \\citet{Sa2026}",
     [("main", "WILCAR-C", "dcv", "mean"), ("main", "WILCAR-C", "holdout", "mean"), ("main", "RIXM-C", "holdout", "mean")]),
    ("Unconstrained SLFNs", [("main", "WILCAR", "dcv", "none"), ("main", "WILCAR", "holdout", "none"),
                             ("main", "RIXM", "holdout", "none"), ("main", "ELM", "holdout", "none"), ("main", "MLP", "holdout", "none")]),
    ("Monotone baselines", [("main", "LR-C", "holdout", "sign"), ("main", "XGB-C", "holdout", "monotone"),
                            ("main", "LGBM-C", "holdout", "monotone"), ("main", "MINMAX", "holdout", "structural"),
                            ("main", "CMNN", "holdout", "structural"), ("main", "LMN", "holdout", "structural")]),
    ("Unconstrained baselines", [("main", "LR", "holdout", "none"), ("main", "XGB", "holdout", "none"),
                                 ("main", "LGBM", "holdout", "none")]),
]
MAIN_ROWS = [(r, m, se, lab(m, c, se)) for _, rows in MAIN_GROUPS for (r, m, se, c) in rows]

# compact comparison of the article: one or two representatives per family
COMPACT_GROUPS = [
    ("Certified SLFNs (this work)", [("cegis", "WILCAR-C", "dcv", "cegis"), ("cegis", "RIXM-C", "holdout", "cegis")]),
    ("Other SLFNs", [("global", "WILCAR-C", "dcv", "sign"), ("main", "WILCAR-C", "dcv", "mean"), ("main", "WILCAR", "dcv", "none"),
                     ("main", "RIXM", "holdout", "none"), ("main", "MLP", "holdout", "none")]),
    ("Monotone baselines", [("main", "LR-C", "holdout", "sign"), ("main", "XGB-C", "holdout", "monotone"),
                            ("main", "LGBM-C", "holdout", "monotone"), ("main", "MINMAX", "holdout", "structural"),
                            ("main", "CMNN", "holdout", "structural"), ("main", "LMN", "holdout", "structural")]),
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
    "Activation-switch MNN": (("69.5", "0.1"), ("94", "1"), ("65.4", "0.1"), "Sartor2025"),
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
            "compas": ("COMPAS recidivism", "Criminal justice", r"\citet{Dressel2018}"),
            "loan": ("Loan defaulter", "Credit risk", r"\citet{Liu2020}")}
    lines = []
    for key, short in DS:
        name, dom, cite = info[key]
        try:
            X, y, s = dload(key)
            n, p, c, pos = len(y), X.shape[1], sum(1 for v in s if v), 100 * y.mean()
            extra = r"\textsuperscript{a}" if key == "loan" else ""
            lines.append(f"{short} & {name} {cite} & {dom} & {n}{extra} & {p} & {c} ({100 * c / p:.0f}\\%) & {pos:.1f} \\\\")
        except Exception:
            known = {"loan": ("20000\\textsuperscript{a}", 28, 5)}          # declared in data.py; positive rate needs the Zenodo file
            if key in known:
                n, p, c = known[key]
                lines.append(f"{short} & {name} {cite} & {dom} & {n} & {p} & {c} ({100 * c / p:.0f}\\%) & \\pend \\\\")
            else:
                lines.append(f"{short} & {name} {cite} & {dom} & -- & -- & -- & -- \\\\")
    write("datasets.tex", "\n".join([
        r"\begin{table*}[!tbp]\centering\small",
        r"\caption{Datasets. $n$: samples used for cross-validation; $p$: inputs; constrained: inputs with a prescribed gain sign $s_i\neq0$ (constraint ratio); positive: share of the positive class. The last three are the monotonicity benchmarks of \citet{Liu2020,Sivaraman2020,Runje2023}, with their preprocessing and monotone features.}",
        r"\label{tab:datasets}",
        r"\resizebox{\linewidth}{!}{\begin{tabular}{llllrrr}\toprule",
        r"ID & Dataset & Domain & $n$ & $p$ & Constrained & Positive (\%) \\\midrule",
        *lines,
        r"\bottomrule\end{tabular}}",
        r"\par\smallskip\footnotesize \textsuperscript{a}Stratified subsample of 20\,000 rows, used for cross-validation because of the cost of the constructive SLSQP fits (Section~\ref{sec:lit} keeps the official test set whole).",
        r"\end{table*}"]))


def _ranks(M, lower_better):
    """Average Friedman ranks over the datasets that at least half of the rows have; rows incomplete there get none."""
    cols = [c for c in M.columns if M[c].notna().sum() >= max(2, len(M) // 2)]
    if not cols:
        return None, []
    sub = M[cols].dropna(axis=0, how="any")
    if len(sub) < 2:
        return None, cols
    return sub.rank(axis=0, ascending=lower_better).mean(axis=1), cols


def table_main(dfs, metric="mcc", name="main_mcc.tex", caption_metric="MCC", supp=True, groups=None):
    groups = groups or MAIN_GROUPS
    rows_all = [(r, m, se, lab(m, c, se)) for _, rows in groups for (r, m, se, c) in rows]
    M = pd.DataFrame(index=[r[3] for r in rows_all], columns=[k for k, _ in DS], dtype=float)
    S = M.copy()
    for role, meth, sel, disp in rows_all:
        for key, _ in DS:
            m, sd = cell_stats(dfs.get(role), meth, sel, key, metric)
            M.loc[disp, key] = np.nan if m is None else m
            S.loc[disp, key] = np.nan if sd is None else sd
    low = metric == "brier"
    ranks, rcols = _ranks(M, low)
    ncol = 3 + len(DS) + 1
    lines = []
    for title, rows in groups:
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{title}}}}} \\")
        for (role, meth, sel, c) in rows:
            disp = lab(meth, c, sel)
            cells = []
            for key, _ in DS:
                m, sd = M.loc[disp, key], S.loc[disp, key]
                col = M[key].dropna()
                best = key in rcols and len(col) and not np.isnan(m) and (m == (col.min() if low else col.max()))
                t = fmt(None if np.isnan(m) else m, sd if supp else None)
                cells.append(rf"\textbf{{{t}}}" if best else t)
            r = "--" if ranks is None or disp not in ranks.index else f"{ranks[disp]:.1f}"
            lines.append(disp + " & " + " & ".join(cells) + f" & {r} \\\\")
        lines.append(r"\midrule")
    if supp:
        cap = (rf"\caption{{Test {caption_metric} on the real datasets, all methods: mean and, in parentheses, standard deviation over "
               r"the outer folds. Conventions as in Table~\ref{M-tab:main_mcc} of the article; best value per dataset in bold; "
               r"rank: average rank over the eight datasets.}")
    else:
        cap = (rf"\caption{{Test {caption_metric} on the real datasets, mean over the outer folds ($5\times5$-fold nested cross-validation; "
               r"$5\times2$ for LOAN). Constraints as in \cref{sec:gains}; monotone: monotone splits of the tree ensembles; structural: "
               r"monotone by architecture. Selection of the number of hidden units by DCV or on a hold-out split. Best value per dataset "
               r"in bold; rank: average rank over the eight datasets. All methods and standard deviations: Supplementary "
               r"\cref{S-tab:main_mcc_full}.}")
    write(name, "\n".join([
        r"\begin{table*}[!tbp]\centering",
        cap,
        rf"\label{{tab:{name.split('.')[0]}}}",
        r"\resizebox{\linewidth}{!}{\begin{tabular}{lll" + "c" * len(DS) + r"c}\toprule",
        "Model & Constraint & Selection & " + " & ".join(sn for _, sn in DS) + r" & Rank \\\midrule",
        *(lines[:-1] if lines and lines[-1] == r"\midrule" else lines),
        r"\bottomrule\end{tabular}}",
        r"\end{table*}"]))
    return M, ranks


MODES = [("none", "main", "WILCAR", "--- ", "none"),
         ("mean", "main", "WILCAR-C", r"average gain $\ge\varepsilon_m$ on the training inputs, \eqref{eq:mean}", "sign of the average effect"),
         ("anchor", "multi", "WILCAR-C", r"gain $\ge\varepsilon_p$ at $k$-means anchors, \eqref{eq:point}", "none between anchors"),
         ("sign", "global", "WILCAR-C", r"$s_iv_jW_{ji}\ge0$, \eqref{eq:sign}", r"monotone (\cref{prop:sign})"),
         ("cegis", "cegis", "WILCAR-C", r"gain $\ge\varepsilon_p$ at anchors grown by counterexamples + certificate", r"monotone (\cref{prop:always})")]


def table_modes(dfs):
    """Constraint families: what they impose and guarantee, and what they deliver on the real data (WILCAR, DCV)."""
    lines = []
    for mode, role, meth, cond, guar in MODES:
        df = dfs.get(role)
        d = None if df is None else df[(df.method == meth) & (df.selection == "dcv")]
        if d is None or not len(d):
            lines.append(f"{mode} & {cond} & {guar} & -- & -- & -- & -- \\\\"); continue
        mcc = d.groupby("dataset").mcc.mean().mean()
        if mode == "sign":
            cert = "100"                                    # by construction (Proposition 1)
        else:
            cert = f"{100 * np.mean(d.cert_status.isin(['certified', 'by_construction'])):.0f}"
        viol = 0.0 if mode in ("sign", "cegis") else 100 * d.viol_domain.mean()
        lines.append(f"{mode} & {cond} & {guar} & {mcc:.3f} & {viol:.1f} & {cert} & {d.time_s.median():.0f} \\\\")
    write("modes.tex", "\n".join([
        r"\begin{table*}[!tbp]\centering\small",
        r"\caption{Constraint families and what they deliver on the real data (WILCAR with DCV, 185 outer folds over eight datasets). "
        r"Guarantee: what holds on the whole domain $\B$. MCC: mean over datasets; viol.: share of (point, constrained input) pairs "
        r"with a wrong-sign gain on $10^4$ uniform points of $\B$; cert.: networks proved monotone by the certificate or by "
        r"construction; time: median per outer fold (selection, refit and certificate).}",
        r"\label{tab:modes}",
        r"\resizebox{\linewidth}{!}{\begin{tabular}{lllrrrr}\toprule",
        r"Mode & Condition imposed & Guarantee on $\B$ & MCC & viol.\ (\%) & cert.\ (\%) & time (s) \\\midrule",
        *lines, r"\bottomrule\end{tabular}}\end{table*}"]))


def table_lit(dfs):
    df = dfs.get("lit")
    keys = [("compas", "COMPAS"), ("heart_disease", "Heart Disease"), ("loan", "Loan Defaulter")]
    lines = []
    for name, (c, h, l, src) in PUBLISHED.items():
        cells = [("--" if v is None else f"{v[0]} ({v[1]})") for v in (c, h, l)]
        lines.append(f"{name} \\citep{{{src}}} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    ours = [("WILCAR-C", "dcv", "WILCAR, cegis, DCV"), ("RIXM-C", "holdout", "RIXM, cegis, hold-out"),
            ("WILCAR", "dcv", "WILCAR, unconstrained, DCV"), ("LR-C", "holdout", "LR, sign"),
            ("XGB-C", "holdout", "XGBoost, monotone"), ("LGBM-C", "holdout", "LightGBM, monotone"),
            ("MINMAX", "holdout", "Min-Max (our runs)"), ("CMNN", "holdout", "CMNN (our runs)"), ("LMN", "holdout", "LMN (our runs)")]
    for meth, sel, disp in ours:
        cells = []
        for k, _ in keys:
            m, s = cell_stats(df, meth, sel, k, "acc")
            cells.append("--" if m is None else f"{100 * m:.1f} ({100 * s:.1f})" if not np.isnan(s) else f"{100 * m:.1f}")
        lines.append(f"{disp} & " + " & ".join(cells) + r" \\")
    write("literature.tex", "\n".join([
        r"\begin{table*}[!tbp]\centering\small",
        r"\caption{Test accuracy (\%) on the official train/test splits of the benchmarks, mean (standard deviation). Upper block: "
        r"values reported in the cited papers, each under its own protocol (e.g.\ best five of ten runs for CMNN). Lower block: "
        r"this work, five seeds, one common protocol (hyper-parameters selected on a validation split of the official training set); "
        r"``our runs'': the published architectures trained under that protocol.}",
        r"\label{tab:lit}",
        r"\resizebox{\linewidth}{!}{\begin{tabular}{lccc}\toprule",
        r"Method & " + " & ".join(n for _, n in keys) + r" \\\midrule",
        *lines, r"\bottomrule\end{tabular}}",
        "" if df is not None else r"\par\smallskip\footnotesize\textit{Pending run: lit}.",
        r"\end{table*}"]))


SYN_ROWS = [("syn_cegis", "WILCAR-C", "dcv", lab("WILCAR-C", "cegis", "dcv")),
            ("syn_global", "WILCAR-C", "dcv", lab("WILCAR-C", "sign", "dcv")),
            ("syn", "WILCAR-C", "dcv", lab("WILCAR-C", "mean", "dcv")), ("syn", "WILCAR", "dcv", lab("WILCAR", "none", "dcv")),
            ("syn", "LR-C", "holdout", lab("LR-C", "sign", "holdout")), ("syn", "XGB-C", "holdout", lab("XGB-C", "monotone", "holdout")),
            ("syn", "LGBM-C", "holdout", lab("LGBM-C", "monotone", "holdout")), ("syn", "MINMAX", "holdout", lab("MINMAX", "structural", "holdout")),
            ("syn", "CMNN", "holdout", lab("CMNN", "structural", "holdout")), ("syn", "LMN", "holdout", lab("LMN", "structural", "holdout"))]
SYN_DS = [("syn_and", "and"), ("syn_or", "or"), ("syn_prod", "prod"), ("syn_and3", "and3"),
          ("syn_andfree", "andfree"), ("syn_andneg", "andneg"), ("syn_wave", "wave")]
SYN_BREAKS = (lab("WILCAR", "none", "dcv"),)


def table_syn(dfs):
    M = pd.DataFrame(index=[r[3] for r in SYN_ROWS], columns=[k for k, _ in SYN_DS], dtype=float)
    cert = {}
    for role, meth, sel, disp in SYN_ROWS:
        df = dfs.get(role)
        ok = df is not None and "mae_true" in df
        for key, _ in SYN_DS:
            m, _ = cell_stats(df, meth, sel, key, "mae_true") if ok else (None, None)
            M.loc[disp, key] = np.nan if m is None else m
        if ok and "cert_status" in df:
            d = df[(df.method == meth) & (df.selection == sel) & df.dataset.isin([k for k, _ in SYN_DS])]
            if len(d) and (d.cert_status != "none").any():
                cert[disp] = f"{100 * np.mean(d.cert_status.isin(['certified', 'by_construction'])):.0f}"
    lines = []
    for disp in M.index:
        cells = []
        for key, _ in SYN_DS:
            v = M.loc[disp, key]; col = M[key].dropna()
            t = "--" if np.isnan(v) else f"{v:.3f}"
            cells.append(rf"\textbf{{{t}}}" if len(col) and not np.isnan(v) and v == col.min() else t)
        av = M.loc[disp].mean() if M.loc[disp].notna().all() else np.nan
        colav = M.mean(axis=1).dropna()
        ta = "--" if np.isnan(av) else f"{av:.3f}"
        cells.append(rf"\textbf{{{ta}}}" if len(colav) and not np.isnan(av) and av == colav.min() else ta)
        lines.append(disp + " & " + " & ".join(cells) + f" & {cert.get(disp, '--')} \\\\")
        if disp in SYN_BREAKS:
            lines.append(r"\midrule")
    pending = [r for r in ("syn", "syn_cegis", "syn_global") if dfs.get(r) is None]
    note = (r"\par\smallskip\footnotesize\textit{Pending runs: " + ", ".join(pending).replace("_", r"\_") + "}.") if pending else ""
    write("synthetic.tex", "\n".join([
        r"\begin{table*}[!tbp]\centering\small",
        r"\caption{Synthetic targets (Supplementary \cref{S-tab:synthdef}): mean absolute error between the fitted and the true "
        r"$P(Y=1\mid x)$ on $2\times10^4$ uniform points, averaged over the 25 outer folds; mean: over the seven targets; cert.: share of final models "
        r"that are monotone on $[0,1]^p$ (certified by the certificate of \cref{sec:certificate} or by construction). Constraints as in \cref{sec:gains}; "
        r"every SLFN uses the best of three initialisations per constructive step. Best value per target in bold.}",
        r"\label{tab:synth}",
        r"\resizebox{\linewidth}{!}{\begin{tabular}{lll" + "c" * (len(SYN_DS) + 2) + r"}\toprule",
        "Model & Constraint & Selection & " + " & ".join(n for _, n in SYN_DS) + r" & mean & cert.\ (\%) \\\midrule",
        *lines, r"\bottomrule\end{tabular}}", note, r"\end{table*}"]))


def table_cert_bench():
    f = RES / "cert_bench.jsonl"
    if not f.exists():
        return
    df = pd.DataFrame([json.loads(l) for l in open(f)])
    kinds = [("free", "unconstrained"), ("mean", "mean constraint"), ("cegis", "cegis step")]
    variants = [("interval", "natural interval extension"), ("+mv", "+ mean-value form"),
                ("+mono", "+ monotonicity test"), ("full", "+ pre-activation space")]
    lines = []
    for k, kd in kinds:
        d0 = df[df.kind == k]
        for v, vd in variants:
            d = d0[d0.variant == v]
            if not len(d):
                continue
            lines.append(f"{kd if v == 'interval' else ''} & {vd} & {100 * np.mean(d.status == 'certified'):.1f} & "
                         f"{100 * np.mean(d.status == 'counterexample'):.1f} & {100 * np.mean(d.status == 'unknown'):.1f} & "
                         f"{1000 * d.time_s.median():.1f} & {1000 * np.percentile(d.time_s, 90):.0f} \\\\")
        lines.append(r"\midrule")
    nn = len(df) // len(variants)
    per = nn // len(kinds)
    viol = {k: 100 * df[(df.kind == k) & (df.variant == "full")].sampled_violation.mean() for k, _ in kinds}
    write("cert_bench.tex", "\n".join([
        r"\begin{table*}[!tbp]\centering\small",
        rf"\caption{{Ablation of the certificate (\cref{{alg:bb}}) on {nn} SLFNs ({per} of each kind: seven real datasets, "
        r"$n\in\{2,4,8,16\}$, two seeds), with the same budget of $2\times10^5$ boxes. Outcome shares (\%) and time per network "
        r"(median and 90th percentile, ms). Dense sampling ($5\times10^4$ points) finds violations in "
        rf"{viol['free']:.0f}\%, {viol['mean']:.0f}\% and {viol['cegis']:.0f}\% of the unconstrained, mean-constrained and cegis networks; "
        r"no certified network has a sampled violation.}",
        r"\label{tab:certbench}",
        r"\resizebox{\linewidth}{!}{\begin{tabular}{llrrrrr}\toprule",
        r"Networks & Bounds & certified & counterex. & unknown & median & p90 \\\midrule",
        *(lines[:-1] if lines and lines[-1] == r"\midrule" else lines),
        r"\bottomrule\end{tabular}}\end{table*}"]))


def table_size(dfs):
    """Selected size n* and time per outer fold (mean), by dataset: hold-out vs DCV vs DCV-1SE."""
    rows = [("main", "WILCAR", "dcv", lab("WILCAR", "none", "dcv")), ("main", "WILCAR", "holdout", lab("WILCAR", "none", "holdout")),
            ("main", "RIXM", "holdout", lab("RIXM", "none", "holdout")),
            ("main", "WILCAR-C", "dcv", lab("WILCAR-C", "mean", "dcv")), ("global", "WILCAR-C", "dcv", lab("WILCAR-C", "sign", "dcv")),
            ("multi", "WILCAR-C", "dcv", lab("WILCAR-C", "anchor", "dcv")),
            ("cegis", "WILCAR-C", "dcv", lab("WILCAR-C", "cegis", "dcv")), ("cegis", "WILCAR-C", "holdout", lab("WILCAR-C", "cegis", "holdout")),
            ("cegis", "RIXM-C", "holdout", lab("RIXM-C", "cegis", "holdout"))]
    lines = []
    for role, meth, sel, disp in rows:
        cells = []
        for key, _ in DS:
            m, _s = cell_stats(dfs.get(role), meth, sel, key, "n_star")
            t, _t = cell_stats(dfs.get(role), meth, sel, key, "time_s")
            cells.append("--" if m is None else f"{m:.1f}\\,{{\\scriptsize({t:.0f}\\,s)}}")
        lines.append(disp + " & " + " & ".join(cells) + r" \\")
    write("size.tex", "\n".join([
        r"\begin{table*}[!tbp]\centering",
        r"\caption{Selected number of hidden units $n^\ast$ and, in parentheses, time per outer fold (selection, refit and certificate), "
        r"means over the outer folds; selection loops stopped by the one-hour budget are included as they are.}",
        r"\label{tab:size}",
        r"\resizebox{\linewidth}{!}{\begin{tabular}{lll" + "c" * len(DS) + r"}\toprule",
        "Model & Constraint & Selection & " + " & ".join(sn for _, sn in DS) + r" \\\midrule",
        *lines, r"\bottomrule\end{tabular}}\end{table*}"]))


# one representative per family for the critical-difference diagram (Nemenyi needs few methods with 8 datasets)
CD_ROWS = [lab("WILCAR-C", "cegis", "dcv"), lab("RIXM-C", "cegis", "holdout"), lab("WILCAR-C", "sign", "dcv"),
           lab("WILCAR-C", "mean", "dcv"), lab("WILCAR", "none", "dcv"), lab("RIXM", "none", "holdout"),
           lab("MLP", "none", "holdout"), lab("LR-C", "sign", "holdout"),
           lab("XGB-C", "monotone", "holdout"), lab("LGBM-C", "monotone", "holdout"), lab("MINMAX", "structural", "holdout"),
           lab("CMNN", "structural", "holdout"), lab("LMN", "structural", "holdout")]
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
       13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426}


def figure_cd(M):
    """Critical-difference diagram (Demsar, 2006) over the datasets complete for every CD row; written only when at least
    six datasets are complete, so that no stale or partial figure reaches the manuscript."""
    fig_dir = HERE / "figures"
    for ext in ("pdf", "png"):
        (fig_dir / f"fig_cd.{ext}").unlink(missing_ok=True)
    rows = [r for r in CD_ROWS if r in M.index and M.loc[r].notna().sum() > 0]
    sub = M.loc[rows].dropna(axis=1, how="any")
    if sub.shape[1] < 6 or len(rows) < 3:
        print("CD diagram skipped: complete datasets", sub.shape[1], "rows", len(rows)); return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = sub.rank(axis=0, ascending=False).mean(axis=1).sort_values()
    k, N = len(R), sub.shape[1]
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
    def name_(r):
        m_, c_, s_ = (x.strip() for x in r.split("&"))
        return " ".join([m_, "free" if c_ == "none" else c_] + ([] if s_ == "hold-out" else [s_]))
    name = {r: name_(r) for r in R.index}
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * k + 1.6))
    ax.set_xlim(1, k); ax.set_ylim(-(k // 2 + 2), 1.5); ax.axis("off")
    ax.plot([1, k], [0, 0], "k-", lw=1)
    for t in range(1, k + 1):
        ax.plot([t, t], [0, 0.12], "k-", lw=1); ax.text(t, 0.25, str(t), ha="center", fontsize=8)
    ax.plot([1, 1 + cd], [1.1, 1.1], "k-", lw=2); ax.text(1 + cd / 2, 1.25, f"CD = {cd:.2f}", ha="center", fontsize=8)
    half = (k + 1) // 2
    for i, (r, v) in enumerate(R.items()):
        left = i < half
        y = -(i + 1) if left else -(k - i)
        x_end = 0.8 if left else k + 0.2
        ax.plot([v, v, x_end], [0, y, y], "k-", lw=0.7)
        ax.text(x_end + (-0.05 if left else 0.05), y, f"{name[r]} ({v:.2f})", ha="right" if left else "left",
                va="center", fontsize=8)
    # cliques: maximal groups of consecutive methods whose rank range is below CD
    vals, cl = R.values, []
    for a_ in range(k):
        b_ = max(j for j in range(a_, k) if vals[j] - vals[a_] < cd)
        if b_ > a_ and not any(c0 <= a_ and b_ <= c1 for c0, c1 in cl):
            cl.append((a_, b_))
    for q, (a_, b_) in enumerate(cl):
        yy = -0.35 - 0.22 * q
        ax.plot([vals[a_] - 0.03, vals[b_] + 0.03], [yy, yy], "k-", lw=2.5)
    fig.tight_layout()
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "fig_cd.pdf"); fig.savefig(fig_dir / "fig_cd.png", dpi=200)
    print("wrote", fig_dir / "fig_cd.pdf", "k =", k, "N =", N)


def numbers(dfs, M, ranks):
    def mac(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}"
    out = []
    ce = dfs.get("cegis")
    if ce is not None and "cert_status" in ce:
        d = ce[ce.method.isin(["WILCAR-C", "RIXM-C"])]
        fb = (d["cert_fallback"].fillna("") != "") if "cert_fallback" in d else pd.Series(False, index=d.index)
        out.append(mac("CegisBB", f"{100 * np.mean((d.cert_status == 'certified') & ~fb):.1f}\\%"))
        out.append(mac("CegisFallback", f"{100 * np.mean(fb):.1f}\\%"))
    mn = dfs.get("main")
    if mn is not None and "cert_status" in mn:
        d = mn[mn.method.isin(["WILCAR-C", "RIXM-C"])]
        out.append(mac("MeanCertified", f"{100 * np.mean(d.cert_status == 'certified'):.1f}\\%"))
        out.append(mac("MeanViolDomain", f"{100 * d.viol_domain.mean():.2f}\\%"))
        d = mn[mn.method.isin(["WILCAR", "RIXM"])]
        out.append(mac("FreeViolDomain", f"{100 * d.viol_domain.mean():.1f}\\%"))
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
    M, ranks = table_main(dfs, "mcc", "main_mcc_full.tex", "MCC", supp=True)
    table_main(dfs, "mcc", "main_mcc.tex", "MCC", supp=False, groups=COMPACT_GROUPS)
    table_main(dfs, "auc", "main_auc.tex", "AUC", supp=True)
    table_main(dfs, "brier", "main_brier.tex", "Brier score", supp=True)
    table_modes(dfs)
    table_lit(dfs)
    table_syn(dfs)
    table_size(dfs)
    table_cert_bench()
    numbers(dfs, M, ranks)
    figure_cd(M)

if __name__ == "__main__":
    main()

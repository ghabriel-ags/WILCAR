"""Consolidated report of the October 2026 re-run (Heart Failure and Pima after the CSV fix).

Run from the worktree root:  python rerun_report/make_report.py [--old-root ../thesis/results]
Reads the new run folders from logs/rerun_2026-10/manifest.json and the March folders listed in OLD_RUNS,
and writes summary_kfold.csv, summary_dcv.csv, summary_algerian_wilcarc.csv, correlations.csv and comparison.md.
"""
import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HF = "heart_failure_clinical_records_dataset__binary_"
PI = "pima__binary_"
ALG = "Algerian_forest_fires__binary_"
DATASETS = {HF: "Heart Failure", PI: "Pima"}
METHODS = {1: "WILCAR", 2: "WILCAR-C", 3: "RIXM", 4: "RIXM-C", 5: "ELM", 6: "ELM-C"}
CONSTRAINED = (2, 4, 6)

OLD_RUNS = {
    "cv": ["cv_20260322_092703", "cv_20260329_233245", "cv_20260330_090253", "cv_20260323_194416",
           "cv_20260322_112130", "cv_20260329_095115", "cv_20260323_195252"],
    "dcv": ["vcd_20260328_200812", "vcd_20260329_001412"],
}
# March Algerian run: WILCAR-C (M2) fold 1 failed, so its metrics cover folds 2-5 only
OLD_ALG_RUN = "cv_20260320_225457"
ALG_FIX_NOTE = (Path(__file__).parent / "algerian_fix_note.md")
# March run of HF RIXM-C used --max-neurons 5 --patience 5; the re-run does the same.
CONFIG_NOTE = {(HF, 4): "max_neurons=5, patience=5 (override, as in March)"}

HF_COLUMNS = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
              "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking",
              "time", "DEATH_EVENT"]
EXPECTED_SIGNALS = {
    HF: [+1, +1, 0, 0, -1, +1, 0, +1, -1, 0, 0, -1],
    PI: [+1, +1, 0, +1, 0, +1, +1, +1],
}


def pm(mean, sd, nd):
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"


# ----------------------------------------------------------------------------- loading
def load_cv(dirs, datasets=tuple(DATASETS)):
    """(dataset, method_id) -> {'summary': row, 'folds': DataFrame, 'run': dir name}."""
    out = {}
    for d in dirs:
        d = Path(d)
        for ds in datasets:
            sub = d / ds
            if not (sub / "summary_stats.csv").exists():
                continue
            summ = pd.read_csv(sub / "summary_stats.csv", float_precision="round_trip")
            folds = pd.read_csv(sub / "fold_results.csv", float_precision="round_trip")
            for _, row in summ.iterrows():
                key = (ds, int(row["method_id"]))
                assert key not in out, f"duplicate {key} in {d.name} and {out[key]['run']}"
                out[key] = {"summary": row, "folds": folds[folds["method_id"] == key[1]], "run": d.name}
    return out


def load_dcv(dirs):
    """(dataset, method_id) -> {'consolidated': row, 'summary': json, 'folds': DataFrame, 'run': dir name}."""
    out = {}
    for d in dirs:
        d = Path(d)
        cons = pd.read_csv(d / "consolidated_vcd_results.csv", float_precision="round_trip")
        for _, row in cons.iterrows():
            key = (row["dataset"], int(row["method_id"]))
            mdir = d / row["dataset"] / f"method_{key[1]}"
            assert key not in out, f"duplicate {key}"
            out[key] = {
                "consolidated": row,
                "summary": json.loads((mdir / "vcd_summary.json").read_text()),
                "folds": pd.read_csv(mdir / "fold_details_at_optimal.csv", float_precision="round_trip"),
                "run": d.name,
            }
    return out


# ----------------------------------------------------------------------------- tables
def kfold_table(cv):
    rows = []
    for ds in DATASETS:
        for m in METHODS:
            if (ds, m) not in cv:
                continue
            s = cv[(ds, m)]["summary"]
            rows.append({
                "dataset": DATASETS[ds], "method": f"M{m} {METHODS[m]}",
                "accuracy": pm(s.test_accuracy_mean, s.test_accuracy_std, 3),
                "mcc": pm(s.test_mcc_mean, s.test_mcc_std, 3),
                "f1": pm(s.test_f1_mean, s.test_f1_std, 3),
                "n_star": pm(s.best_neurons_mean, s.best_neurons_std, 1),
                "scr_pct": f"{100 * s.scr_mean:.2f}",
                "time_per_fold_s": f"{s.training_time_mean:.0f}",
                "config": CONFIG_NOTE.get((ds, m), "code defaults"),
                "run": cv[(ds, m)]["run"],
            })
    return pd.DataFrame(rows)


def dcv_table(dcv):
    rows = []
    for ds in DATASETS:
        for m in (1, 2):
            if (ds, m) not in dcv:
                continue
            c, s = dcv[(ds, m)]["consolidated"], dcv[(ds, m)]["summary"]
            rows.append({
                "dataset": DATASETS[ds], "method": f"M{m} {METHODS[m]}",
                "n_star": int(c["n*"]),
                "accuracy": pm(c["cv_Acc_mean"], c["cv_Acc_std"], 3),
                "f1_mean": f"{c['cv_F1_mean']:.3f}",
                "mcc_mean": f"{c['cv_MCC_mean']:.3f}",
                "scr_pct": f"{100 * c['cv_SCR']:.2f}",
                "total_time_min": f"{c['total_time_min']:.1f}",
                "final_n": int(c["final_n"]),
                "final_scr_pct": f"{100 * c['final_SCR']:.2f}",
                "J_T_metric": s.get("J_T_metric"),
                "run": dcv[(ds, m)]["run"],
            })
    return pd.DataFrame(rows)


def read_dataset(path, ds):
    if ds == PI:
        return pd.read_csv(path, sep=";")
    df = pd.read_csv(path, sep=";", header=None)
    df.columns = HF_COLUMNS
    return df


def correlations(old_csv_text):
    rows = []
    for ds, fname in ((HF, f"data/raw/{HF}.csv"), (PI, f"data/raw/{PI}.csv")):
        new = read_dataset(fname, ds).astype(float)
        old = read_dataset(io.StringIO(old_csv_text[ds]), ds).astype(float)
        y_new, y_old = new.iloc[:, -1], old.iloc[:, -1]
        counts = y_new.value_counts().to_dict()
        for j, col in enumerate(new.columns[:-1]):
            r_new = new[col].corr(y_new)
            r_old = old[col].corr(y_old)
            rows.append({
                "dataset": DATASETS[ds], "variable": col, "expected_signal": EXPECTED_SIGNALS[ds][j],
                "r_pb": f"{r_new:.3f}", "r_pb_before_fix": f"{r_old:.3f}",
                "changed": "yes" if round(r_new, 3) != round(r_old, 3) else "",
                "n": len(new), "n_class0": int(counts.get(0.0, 0)), "n_class1": int(counts.get(1.0, 0)),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Algerian WILCAR-C
ALG_FOLD_FIELDS = [("best_neurons", "n*", 0), ("test_accuracy", "Acc", 3), ("test_mcc", "MCC", 3),
                   ("test_f1", "F1", 3), ("scr", "SCR", 2)]


def algerian_table(old, new):
    rows = []
    for label, f in (("março (4 folds: 2-5)", old), ("outubro (5 folds)", new)):
        s = f["summary"]
        rows.append({
            "run": f["run"], "versao": label, "folds_ok": int(f["folds"]["test_mcc"].notna().sum()),
            "accuracy": pm(s.test_accuracy_mean, s.test_accuracy_std, 3),
            "mcc": pm(s.test_mcc_mean, s.test_mcc_std, 3),
            "f1": pm(s.test_f1_mean, s.test_f1_std, 3),
            "n_star": pm(s.best_neurons_mean, s.best_neurons_std, 2),
            "scr_pct": f"{100 * s.scr_mean:.2f}",
            "time_per_fold_s": f"{s.training_time_mean:.0f}",
        })
    return pd.DataFrame(rows)


def algerian_md(old, new, table):
    md = ["## Algerian — WILCAR-C (5 folds)", "",
          "Re-run só do M2 com o código corrigido; o WILCAR (M1) não teve falha e não foi re-rodado.", "",
          "| Versão | folds | Acc | MCC | F1 | n* | SCR % | t/fold (s) |", "|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        md.append(f"| {r.versao} | {r.folds_ok} | {r.accuracy} | {r.mcc} | {r.f1} | {r.n_star} | {r.scr_pct} | {r.time_per_fold_s} |")
    md += ["", "Por fold (março → outubro):", "",
           "| Fold | " + " | ".join(f"{h} mar | {h} out" for _, h, _ in ALG_FOLD_FIELDS) + " | idêntico? |",
           "|---|" + "---|" * (2 * len(ALG_FOLD_FIELDS) + 1)]
    fo = old["folds"].set_index("fold"); fn = new["folds"].set_index("fold")
    for k in range(1, 6):
        o = fo.loc[k] if k in fo.index else None
        n = fn.loc[k]
        failed = o is None or pd.isna(o["test_mcc"])
        cells = [str(k)]
        for key, _, nd in ALG_FOLD_FIELDS:
            cells += ["erro" if failed else f"{o[key]:.{nd}f}", f"{n[key]:.{nd}f}"]
        same = "—" if failed else ("sim" if all(o[key] == n[key] for key in
                                               ("best_neurons", "test_r2", "test_mse", "test_mcc", "scr")) else "**não**")
        cells.append(same)
        md.append("| " + " | ".join(cells) + " |")
    if "error" in fo:
        errs = fo["error"].dropna()
        if len(errs):
            md += ["", f"Erro de março no fold {errs.index[0]}: `{errs.iloc[0]}`"]
    md += ["", ALG_FIX_NOTE.read_text().strip() if ALG_FIX_NOTE.exists() else "[nota sobre a causa ausente]", ""]
    return "\n".join(md)


# ----------------------------------------------------------------------------- checks
def sanity_checks(cv, dcv):
    lines, failures = [], 0

    def check(ok, text):
        nonlocal failures
        failures += not ok
        lines.append(f"- {'✅' if ok else '❌ **FALHA**'} {text}")

    for ds in DATASETS:
        for m in CONSTRAINED:
            f = cv.get((ds, m))
            if f is None:
                check(False, f"{DATASETS[ds]} M{m} {METHODS[m]}: run ausente")
                continue
            scr = f["folds"]["scr"].to_numpy()
            check(len(scr) == 5 and np.all(scr == 1.0),
                  f"{DATASETS[ds]} M{m} {METHODS[m]} (k-fold): SCR por fold = {[f'{v:.0%}' for v in scr]}")
        f = dcv.get((ds, 2))
        if f is None:
            check(False, f"{DATASETS[ds]} DCV M2 WILCAR-C: run ausente")
        else:
            scr = f["folds"]["scr"].to_numpy()
            fin = f["consolidated"]["final_SCR"]
            check(len(scr) == 5 and np.all(scr == 1.0) and fin == 1.0,
                  f"{DATASETS[ds]} DCV M2 WILCAR-C: SCR por fold = {[f'{v:.0%}' for v in scr]}, final_SCR = {fin:.0%}")
        for m in (1, 2):
            f = dcv.get((ds, m))
            if f is not None:
                c = f["consolidated"]
                check(c["final_n"] <= c["n*"],
                      f"{DATASETS[ds]} DCV M{m} {METHODS[m]}: final_n = {int(c['final_n'])} ≤ n* = {int(c['n*'])}")
    f = cv.get((ALG, 2))
    if f is not None:
        scr = f["folds"]["scr"].to_numpy()
        err = f["folds"]["error"].notna().any() if "error" in f["folds"] else False
        check(len(scr) == 5 and not err and np.all(scr == 1.0),
              f"Algerian M2 WILCAR-C (k-fold, re-run): 5 folds sem erro, SCR por fold = {[f'{v:.0%}' for v in scr]}")
    nan_hits = []
    for name, store in (("k-fold", cv), ("DCV", dcv)):
        for key, f in store.items():
            for part, obj in f.items():
                if isinstance(obj, pd.DataFrame):
                    num = obj.select_dtypes("number")
                elif isinstance(obj, pd.Series):
                    num = pd.to_numeric(obj, errors="coerce")[obj.map(lambda v: isinstance(v, (int, float)))]
                else:
                    continue
                if num.isna().to_numpy().any():
                    nan_hits.append(f"{name} {key[0]} M{key[1]} ({part})")
    check(not nan_hits, "nenhum NaN nas tabelas de resultado" + (f": {nan_hits}" if nan_hits else ""))
    return lines, failures


# ----------------------------------------------------------------------------- comparison
def ranking(cv, ds, metric):
    vals = {m: cv[(ds, m)]["summary"][f"test_{metric}_mean"] for m in METHODS if (ds, m) in cv}
    return sorted(vals, key=lambda m: -vals[m])


def fmt_rank(order):
    return " > ".join(METHODS[m] for m in order)


def comparison_md(cv_old, cv_new, dcv_old, dcv_new, corr, checks, failures, header):
    md = ["# Re-run Heart Failure / Pima (out/2026) vs. março/2026", "", header, "",
          "## Checagens de sanidade", "",
          f"**{'Todas as checagens passaram.' if failures == 0 else f'{failures} checagem(ns) FALHARAM — ver abaixo.'}**",
          "", *checks, ""]

    md += ["## K-fold (5 folds): março → outubro", "",
           "Δ = novo − antigo. Valores: média ± desvio sobre os 5 folds.", ""]
    for ds in DATASETS:
        md += [f"### {DATASETS[ds]}", "",
               "| Método | Acc mar | Acc out | Δ | MCC mar | MCC out | Δ | F1 mar | F1 out | Δ | n* mar | n* out | SCR % mar | SCR % out | t/fold mar (s) | t/fold out (s) |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            if (ds, m) not in cv_old or (ds, m) not in cv_new:
                md.append(f"| M{m} {METHODS[m]} | (ausente) |")
                continue
            o, n = cv_old[(ds, m)]["summary"], cv_new[(ds, m)]["summary"]
            cells = [f"M{m} {METHODS[m]}" + (" ¹" if (ds, m) in CONFIG_NOTE else "")]
            for k in ("accuracy", "mcc", "f1"):
                a, b, sa, sb = o[f"test_{k}_mean"], n[f"test_{k}_mean"], o[f"test_{k}_std"], n[f"test_{k}_std"]
                cells += [pm(a, sa, 3), pm(b, sb, 3), f"{b - a:+.3f}"]
            cells += [pm(o.best_neurons_mean, o.best_neurons_std, 1), pm(n.best_neurons_mean, n.best_neurons_std, 1),
                      f"{100 * o.scr_mean:.2f}", f"{100 * n.scr_mean:.2f}",
                      f"{o.training_time_mean:.0f}", f"{n.training_time_mean:.0f}"]
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
        for metric, label in (("mcc", "MCC"), ("accuracy", "Accuracy"), ("f1", "F1")):
            ro, rn = ranking(cv_old, ds, metric), ranking(cv_new, ds, metric)
            flag = "**mudou**" if ro != rn else "igual"
            md += [f"- Ordenação por {label} ({flag}):", f"  - março:   {fmt_rank(ro)}", f"  - outubro: {fmt_rank(rn)}"]
        md.append("")
    if CONFIG_NOTE:
        md += ["¹ " + "; ".join(f"{DATASETS[d]} M{m}: {t}" for (d, m), t in CONFIG_NOTE.items()), ""]

    md += ["## DCV (WILCAR / WILCAR-C, J_T = soma da BCE de teste)", "",
           "| Dataset | Método | n* mar | n* out | Acc mar | Acc out | Δ | MCC mar | MCC out | Δ | F1 mar | F1 out | SCR % mar | SCR % out | final_n mar/out | tempo mar/out (min) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        for m in (1, 2):
            if (ds, m) not in dcv_old or (ds, m) not in dcv_new:
                md.append(f"| {DATASETS[ds]} | M{m} {METHODS[m]} | (ausente) |")
                continue
            o, n = dcv_old[(ds, m)]["consolidated"], dcv_new[(ds, m)]["consolidated"]
            md.append("| " + " | ".join([
                DATASETS[ds], f"M{m} {METHODS[m]}", str(int(o["n*"])), str(int(n["n*"])),
                pm(o.cv_Acc_mean, o.cv_Acc_std, 3), pm(n.cv_Acc_mean, n.cv_Acc_std, 3),
                f"{n.cv_Acc_mean - o.cv_Acc_mean:+.3f}",
                f"{o.cv_MCC_mean:.3f}", f"{n.cv_MCC_mean:.3f}", f"{n.cv_MCC_mean - o.cv_MCC_mean:+.3f}",
                f"{o.cv_F1_mean:.3f}", f"{n.cv_F1_mean:.3f}",
                f"{100 * o.cv_SCR:.2f}", f"{100 * n.cv_SCR:.2f}",
                f"{int(o.final_n)}/{int(n.final_n)}", f"{o.total_time_min:.1f}/{n.total_time_min:.1f}",
            ]) + " |")
    md.append("")
    for ds in DATASETS:
        if all((ds, m) in dcv_old and (ds, m) in dcv_new for m in (1, 2)):
            def order(store):
                return fmt_rank(sorted((1, 2), key=lambda m: -store[(ds, m)]["consolidated"]["cv_MCC_mean"]))
            ro, rn = order(dcv_old), order(dcv_new)
            md.append(f"- {DATASETS[ds]}, DCV por MCC: março {ro}; outubro {rn} ({'**mudou**' if ro != rn else 'igual'})")
    md.append("")

    md += ["## Correlação ponto-bisserial com o alvo (antes → depois da correção)", "",
           "| Dataset | Variável | sinal esperado | r antes | r depois |", "|---|---|---|---|---|"]
    for _, r in corr[corr["changed"] == "yes"].iterrows():
        md.append(f"| {r.dataset} | {r.variable} | {r.expected_signal:+d} | {r.r_pb_before_fix} | {r.r_pb} |")
    md += ["", "As demais variáveis não mudam (tabela completa em `correlations.csv`).", ""]
    for ds in DATASETS:
        c = corr[corr["dataset"] == DATASETS[ds]].iloc[0]
        md.append(f"- {DATASETS[ds]}: n = {c.n}, classe 0 = {c.n_class0}, classe 1 = {c.n_class1}")
    md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-root", default="../thesis/results", help="results/ folder holding the March runs")
    ap.add_argument("--out", default="rerun_report", help="output folder")
    ap.add_argument("--new-dirs", nargs="*", default=None,
                    help="override the new run folders (default: results_dir entries of manifest.json)")
    args = ap.parse_args()

    old_root = Path(args.old_root)
    cv_old = load_cv([old_root / "cross_validation" / d for d in OLD_RUNS["cv"]])
    dcv_old = load_dcv([old_root / "dynamic_cv" / d for d in OLD_RUNS["dcv"]])

    if args.new_dirs is None:
        manifest = json.loads(Path("logs/rerun_2026-10/manifest.json").read_text())
        # finished runs that are not superseded (in-progress and stopped jobs are left out)
        current = [j for j in manifest["jobs"]
                   if j["results_dir"] and j["exit_code"] == 0 and not j.get("superseded_by")]
        new_dirs = [j["results_dir"] for j in current]
        header_commit = manifest["commit"]
        threads_note = "\n\nThreads do BLAS por run (política: K-fold com 20, DCV do WILCAR-C com 4 por worker):  \n" + \
            "  \n".join(f"- `{Path(j['results_dir']).name}` ({j['label']}): {j['blas_threads']}" for j in current)
    else:
        new_dirs, header_commit, threads_note = args.new_dirs, "(teste)", ""
    cv_new = load_cv([d for d in new_dirs if "cross_validation" in d], datasets=(*DATASETS, ALG))
    alg_old = load_cv([old_root / "cross_validation" / OLD_ALG_RUN], datasets=(ALG,)).get((ALG, 2))
    dcv_new = load_dcv([d for d in new_dirs if "dynamic_cv" in d])

    OUT = Path(args.out)
    OUT.mkdir(exist_ok=True)
    kfold_table(cv_new).to_csv(OUT / "summary_kfold.csv", index=False)
    dcv_table(dcv_new).to_csv(OUT / "summary_dcv.csv", index=False)

    old_csv = {ds: subprocess.run(["git", "show", f"0e54bfe:data/raw/{ds}.csv"], capture_output=True,
                                  text=True, check=True).stdout for ds in DATASETS}
    corr = correlations(old_csv)
    corr.to_csv(OUT / "correlations.csv", index=False)

    checks, failures = sanity_checks(cv_new, dcv_new)
    header = (f"Código: branch `dissertation/rerun-hf-pima`, commit `{header_commit}`. "
              f"Única mudança de entrada vs. março: os dois CSVs corrigidos (HF idade, 2 células; Pima DPF, 47 células).\n\n"
              f"Runs de março: {', '.join(OLD_RUNS['cv'] + OLD_RUNS['dcv'])}.  \n"
              f"Runs novas: {', '.join(Path(d).name for d in new_dirs)}." + threads_note)
    md = comparison_md(cv_old, cv_new, dcv_old, dcv_new, corr, checks, failures, header)
    if (ALG, 2) in cv_new:
        table = algerian_table(alg_old, cv_new[(ALG, 2)])
        table.to_csv(OUT / "summary_algerian_wilcarc.csv", index=False)
        md += "\n" + algerian_md(alg_old, cv_new[(ALG, 2)], table)
    else:
        md += ("\n## Algerian — WILCAR-C (5 folds)\n\nPendente: o K-fold ainda está em execução; entra num commit "
               "posterior. Causa e correção do erro de março em `algerian_fix_note.md`.\n")
    (OUT / "comparison.md").write_text(md)
    print(f"sanity failures: {failures}")


if __name__ == "__main__":
    main()

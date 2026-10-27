"""DCV of WILCAR-C (M2): March 2026 (no weight reuse, parallel worker bug) vs. October 2026 (weight reuse restored).

Run from the worktree root:  python rerun_report/make_dcv_reuse_report.py [--old-root ../thesis/results]
New runs: the results_dir of the dcvm2_* jobs in logs/rerun_2026-10/manifest.json (or --new-dirs).
Writes dcv_wilcarc_reuse.csv, dcv_wilcarc_reuse.md, jt_curves.csv and jt_curves.png to rerun_report/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("rerun_report")
REGRESSION = ["airfoil_self_noise", "computer_hardware", "energy_heating", "energy_cooling", "lavender_friction",
              "optical_network", "aquatic_toxicity", "fish_toxicity", "real_estate", "synchronous_machine",
              "yacht_hydrodynamics"]
CLASSIFICATION = {"Algerian_forest_fires__binary_": "vcd_20260323_210207",
                  "breast__binary_": "vcd_20260324_091505",
                  "Diagnostic_Breast_Cancer__binary_": "vcd_20260328_200827",
                  "heart_failure_clinical_records_dataset__binary_": "vcd_20260328_200812",
                  "pima__binary_": "vcd_20260329_001412"}
MARCH = {**{ds: "vcd_20260310_140932" for ds in REGRESSION}, **CLASSIFICATION}
SHORT = {"Algerian_forest_fires__binary_": "Algerian", "breast__binary_": "Breast (original)",
         "Diagnostic_Breast_Cancer__binary_": "Breast (diagnostic)",
         "heart_failure_clinical_records_dataset__binary_": "Heart Failure", "pima__binary_": "Pima"}
DATA_FIXED = {"heart_failure_clinical_records_dataset__binary_", "pima__binary_"}  # CSVs corrected in e63e54c
NOTES = {"Diagnostic_Breast_Cancer__binary_": "--max-neurons 5 (como em março)"}
N_MAX = {"Diagnostic_Breast_Cancer__binary_": 5}  # other datasets: the M2 default of run_dynamic_cv.py
N_MAX_DEFAULT, PATIENCE = 50, 15
CURVES = ["Algerian_forest_fires__binary_", "breast__binary_", "computer_hardware", "aquatic_toxicity"]

INK, INK_MUTED, GRID, SURFACE = "#1f1f1e", "#6b6b68", "#e6e6e3", "#fcfcfb"
SERIES = {"mar": ("#2a78d6", "março (sem reaproveitamento)"), "out": ("#eb6834", "outubro (com reaproveitamento)")}


def load(run_dir, ds):
    d = Path(run_dir) / ds / "method_2"
    if not (d / "vcd_summary.json").exists():
        return None
    hist = pd.read_csv(d / "vcd_history.csv", float_precision="round_trip")
    return {"summary": json.loads((d / "vcd_summary.json").read_text()), "history": hist, "run": Path(run_dir).name}


def describe(r, is_cls, n_max):
    s, h = r["summary"], r["history"]
    jt = h["J_T"].to_numpy()
    steps = int(s["max_neurons_evaluated"])  # last accepted n
    n_star = int(s["optimal_neurons"])
    # The loop stops early only after PATIENCE accepted steps without improvement, i.e. after n* (discarded steps
    # do not count); otherwise it runs up to n_max, possibly ending in discarded steps
    reached = int((h["n_neurons"] > n_star).sum()) < PATIENCE or steps == n_max
    row = {
        "run": r["run"], "n_star": n_star, "J_T_star": s["best_J_T"],
        "steps_evaluated": steps, "steps_discarded": steps - len(h), "n_max": n_max, "reached_n_max": reached,
        "J_T_1": float(jt[0]) if len(jt) else np.nan,
        "frac_decreasing": float(np.mean(np.diff(jt) < 0)) if len(jt) > 1 else np.nan,
        "scr_pct": 100 * s["cv_mean_scr"], "vcd_time_min": s["vcd_time"] / 60,
    }
    if is_cls:
        row.update(acc_mean=s["cv_mean_accuracy"], acc_std=s["cv_std_accuracy"], f1=s["cv_mean_f1"],
                   mcc=s["cv_mean_mcc"], final_n=int(s["final_n_neurons"]), final_scr_pct=100 * s["final_scr"])
    else:
        row.update(r2_mean=s["cv_mean_r2_test"], r2_std=s["cv_std_r2_test"])
    return row


def curves_png(pairs, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=SURFACE, layout="constrained")
    for ax, (ds, old, new) in zip(axes.flat, pairs):
        ax.set_facecolor(SURFACE)
        top = 0
        for band, (key, r) in enumerate((("mar", old), ("out", new))):
            if r is None:
                continue
            color, label = SERIES[key]
            h = r["history"]
            steps = int(r["summary"]["max_neurons_evaluated"])
            full = pd.Series(np.nan, index=range(1, steps + 1))
            full[h["n_neurons"].to_numpy()] = h["J_T"].to_numpy()
            ax.plot(full.index, full.values, color=color, lw=2, marker="o", ms=4.5, label=label, zorder=3)
            n_star = int(r["summary"]["optimal_neurons"])
            ax.plot([n_star], [full[n_star]], "o", ms=10, mfc="none", mec=color, mew=2, zorder=4)
            skipped = full.index[full.isna()]
            ax.vlines(skipped, 0.01 + 0.05 * band, 0.05 + 0.05 * band, color=color, lw=2,
                      transform=ax.get_xaxis_transform())
            top = max(top, steps)
        ax.set_title(SHORT.get(ds, ds), color=INK, fontsize=11, loc="left")
        ax.set_xlabel("n (neurônios)", color=INK_MUTED, fontsize=9)
        ax.set_ylabel("J_T(n)", color=INK_MUTED, fontsize=9)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        ax.set_xlim(0.5, top + 0.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False, fontsize=10, labelcolor=INK)
    fig.text(0.5, 0, "Anel: n*. Traços na base: passos n descartados (algum fold sem 100% de SCR).",
             ha="center", va="bottom", color=INK_MUTED, fontsize=8.5)
    fig.get_layout_engine().set(h_pad=0.15, rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-root", default="../thesis/results")
    ap.add_argument("--new-dirs", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    threads = {}  # run folder name -> BLAS threads of the job (from the manifest)
    if args.new_dirs is None:
        manifest = json.loads(Path("logs/rerun_2026-10/manifest.json").read_text())
        new_dirs = [j["results_dir"] for j in manifest["jobs"]
                    if j["label"].startswith("dcvm2_") and j["results_dir"] and j["exit_code"] == 0
                    and not j.get("superseded_by")]
        threads = {Path(j["results_dir"]).name: j.get("blas_threads", "") for j in manifest["jobs"] if j["results_dir"]}
    else:
        new_dirs = args.new_dirs

    rows, pairs = [], {}
    for ds in REGRESSION + list(CLASSIFICATION):
        is_cls = ds in CLASSIFICATION
        old = load(Path(args.old_root) / "dynamic_cv" / MARCH[ds], ds)
        new = next((r for d in new_dirs if (r := load(d, ds)) is not None), None)
        pairs[ds] = (old, new)
        for version, r in (("março", old), ("outubro", new)):
            row = {"dataset": SHORT.get(ds, ds), "task": "classificação" if is_cls else "regressão", "versao": version,
                   "nota": NOTES.get(ds, "") + (" CSV corrigido em out." if ds in DATA_FIXED else "")}
            row.update(describe(r, is_cls, N_MAX.get(ds, N_MAX_DEFAULT)) if r is not None else {"run": "pendente"})
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "dcv_wilcarc_reuse.csv", index=False)

    curve_rows = []
    for ds in CURVES:
        for version, r in zip(("março", "outubro"), pairs[ds]):
            if r is not None:
                for _, h in r["history"].iterrows():
                    curve_rows.append({"dataset": SHORT.get(ds, ds), "versao": version, "n": int(h["n_neurons"]),
                                       "J_T": h["J_T"]})
    pd.DataFrame(curve_rows).to_csv(out / "jt_curves.csv", index=False)
    curves_png([(ds, *pairs[ds]) for ds in CURVES], out / "jt_curves.png")

    md = ["# DCV do WILCAR-C: março (sem reaproveitamento) vs. outubro (com reaproveitamento)", "",
          "Em março, o worker paralelo do DCV recriava o modelo a cada n e não restaurava `_vcd_previous_param` "
          "(o atributo só era criado dentro de `constructive_step`). O WILCAR-C, portanto, reinicializou a rede "
          "inteira a cada n nos 16 DCVs. A correção (`a30890a`) restaura o reaproveitamento e faz o estado dos "
          "folds avançar só em passos aceitos (todos os K folds com 100% de SCR). Mesmas flags de março e 4 threads "
          "do BLAS por worker, como em março (ver `README.md`).", "",
          "`J_T(1) idêntico`: em n = 1 não há reaproveitamento, então o valor tem de coincidir com março quando os "
          "dados são os mesmos. Isso confirma que a única mudança está nos passos n ≥ 2. `frac. decresc.`: fração dos "
          "passos aceitos consecutivos em que J_T diminui. `passos`: último n aceito.", "",
          f"`atingiu n_max`: o laço chegou a n = n_max ({N_MAX_DEFAULT}; 5 no BCD) sem parar pela paciência "
          f"({PATIENCE} passos aceitos sem melhora; passos descartados não contam). Com `sim`, a busca foi truncada "
          "pelo limite, e n* pode estar subestimado.", ""]
    for task, cols in (("regressão", ["R² (CV)", "SCR %"]), ("classificação", ["Acc (CV)", "F1", "MCC", "SCR %", "final_n"])):
        md += [f"## {task.capitalize()}", "",
               "| Dataset | versão | n* | J_T* | " + " | ".join(cols) +
               " | passos | descartados | atingiu n_max | frac. decresc. | tempo (min) | J_T(1) idêntico |",
               "|---|---|---|---|" + "---|" * len(cols) + "---|---|---|---|---|---|"]
        for ds in (REGRESSION if task == "regressão" else list(CLASSIFICATION)):
            old, new = pairs[ds]
            for version, r in (("mar", old), ("out", new)):
                if r is None:
                    md.append(f"| {SHORT.get(ds, ds)} | {version} | pendente |")
                    continue
                d = describe(r, task == "classificação", N_MAX.get(ds, N_MAX_DEFAULT))
                vals = ([f"{d['r2_mean']:.4f} ± {d['r2_std']:.4f}", f"{d['scr_pct']:.1f}"] if task == "regressão" else
                        [f"{d['acc_mean']:.3f} ± {d['acc_std']:.3f}", f"{d['f1']:.3f}", f"{d['mcc']:.3f}",
                         f"{d['scr_pct']:.1f}", str(d["final_n"])])
                same = ""
                if version == "out" and old is not None:
                    if ds in DATA_FIXED:
                        same = "n/a (dados corrigidos)"
                    else:
                        same = "sim" if d["J_T_1"] == float(old["history"]["J_T"].iloc[0]) else "**não**"
                md.append(f"| {SHORT.get(ds, ds)} | {version} | {d['n_star']} | {d['J_T_star']:.6f} | " +
                          " | ".join(vals) + f" | {d['steps_evaluated']} | {d['steps_discarded']} | "
                          f"{'**sim**' if d['reached_n_max'] else 'não'} | "
                          f"{d['frac_decreasing']:.2f} | {d['vcd_time_min']:.1f} | {same} |")
        md.append("")
    md += ["## Tempo do DCV do WILCAR-C por dataset", "",
           "| Dataset | março (min) | outubro (min) | passos mar/out |", "|---|---|---|---|"]
    for ds in REGRESSION + list(CLASSIFICATION):
        old, new = pairs[ds]
        t = [f"{r['summary']['vcd_time'] / 60:.1f}" if r else "pendente" for r in (old, new)]
        st = "/".join(str(r["summary"]["max_neurons_evaluated"]) if r else "–" for r in (old, new))
        md.append(f"| {SHORT.get(ds, ds)} | {t[0]} | {t[1]} | {st} |")
    md += ["", "Notas:", "",
           "- Breast (diagnostic): `--max-neurons 5`, como em março (`vcd_20260328_200827`); com n_max = 5 o laço "
           "sempre chega ao limite.",
           "- Heart Failure e Pima: a versão de outubro usa os CSVs corrigidos, então a comparação mistura a correção "
           "dos dados e o reaproveitamento.",
           "- Tempos: as runs de outubro rodaram com `nice -n 10` e dividiram a máquina com outras re-rodadas; não são "
           "comparáveis com março."]
    pending = [SHORT.get(ds, ds) for ds, (_, new) in pairs.items()
               if new is not None and threads.get(new["run"], "").startswith("default")]
    if pending:
        md.append(f"- **Ainda com 3 threads por worker** (default; versão com 4 threads pendente): {', '.join(pending)}.")
    md += ["",
           "## Curvas J_T(n)", "", "![J_T(n)](jt_curves.png)", "", "Dados em `jt_curves.csv`.", ""]
    note = Path(__file__).parent / "sync_machine_note.md"
    if note.exists():
        md += [note.read_text().strip(), ""]
    (out / "dcv_wilcarc_reuse.md").write_text("\n".join(md))
    print(f"written to {out}")


if __name__ == "__main__":
    main()

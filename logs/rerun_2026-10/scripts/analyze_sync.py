"""Analysis of the synchronous_machine WILCAR-C DCV diagnostics (diag_sync.py outputs). Prints the findings and
writes sync_folds.csv and sync_folds.png to <report_dir>.  Usage: python analyze_sync.py <diag_dir> <report_dir>"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
NEW = Path("/home/ghabr/projects/WILCAR-dissertacao/results/dynamic_cv/vcd_20261027_073947_reg/synchronous_machine/method_2")
MAR = Path("/home/ghabr/projects/thesis/results/dynamic_cv/vcd_20260310_140932/synchronous_machine/method_2")
INK, INK_MUTED, GRID, SURFACE = "#1f1f1e", "#6b6b68", "#e6e6e3", "#fcfcfb"
FOLD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

steps = {m: pd.read_csv(D / f"steps_{m}.csv") for m in ("parallel", "noparallel", "march") if (D / f"steps_{m}.csv").exists()}
hist = {m: json.loads((D / f"history_{m}.json").read_text()) for m in steps}

# 1. replication check against the real runs (bit for bit)
real_new = pd.read_csv(NEW / "vcd_history.csv", float_precision="round_trip")["J_T"].tolist()
real_mar = pd.read_csv(MAR / "vcd_history.csv", float_precision="round_trip")["J_T"].tolist()
print("REPLICA parallel == October run:", hist.get("parallel", {}).get("J_T") == real_new)
print("REPLICA march    == March run:  ", hist.get("march", {}).get("J_T") == real_mar)

# 2. per fold and n
for mode, s in steps.items():
    acc = s[s["success"]]
    piv = acc.pivot(index="n", columns="fold", values="r2_test")
    print(f"\n[{mode}] R2 test per fold (rows n): first 4 n and n* region")
    print(piv.head(4).round(4).to_string())
    bad = acc[(acc["r2_train"] < 0.5)]
    print(f"[{mode}] accepted steps with R2_train < 0.5: folds {sorted(bad['fold'].unique().tolist())}, "
          f"n = {sorted(bad['n'].unique().tolist())[:10]}{'...' if bad['n'].nunique() > 10 else ''}")
    print(f"[{mode}] discarded (fold, n): {s[~s['success']][['fold', 'n']].values.tolist()}")

# 3. the degenerate fold at the step where it degenerates
p = steps["parallel"]
first_bad = p[(p["success"]) & (p["r2_train"] < 0.5)].sort_values("n").head(1)
if len(first_bad):
    f0, n0 = int(first_bad["fold"].iloc[0]), int(first_bad["n"].iloc[0])
    row = p[(p["fold"] == f0) & (p["n"] == n0)].iloc[0]
    print(f"\nDEGENERATE fold {f0} from n = {n0}: r2_train={row.r2_train:.4f} r2_test={row.r2_test:.4f} "
          f"mse_train={row.mse_train:.5f} mse_test={row.mse_test:.5f} scr={row.scr}")
    print(f"  reinitializations (attempts) = {row.n_attempts}; SLSQP calls = {row.slsqp_calls}, of which "
          f"success = {row.slsqp_success_count}")
    print(f"  accepted attempt: success={row.accepted_success} status={row.accepted_status} nit={row.accepted_nit} "
          f"fun={row.accepted_fun:.6f} message='{row.accepted_message}'")
    nxt = p[(p["fold"] == f0) & (p["n"] > n0) & p["success"]]
    print(f"  later steps of fold {f0}: r2_train range {nxt.r2_train.min():.4f}..{nxt.r2_train.max():.4f}; "
          f"accepted SLSQP success in {int(nxt.accepted_success.sum())}/{len(nxt)}; "
          f"messages: {nxt.accepted_message.value_counts().to_dict()}")
    if "march" in steps:
        m = steps["march"]
        mrow = m[(m["fold"] == f0) & (m["n"] == n0)]
        if len(mrow):
            print(f"  March (no reuse) same fold, same n: r2_train={mrow.r2_train.iloc[0]:.4f}; next n: "
                  f"{m[(m['fold'] == f0) & (m['n'] == n0 + 1)].r2_train.round(4).tolist()}")

# 4. state flow: each fold grows from its own last accepted state
leaks, wrong = 0, 0
acc = p[p["success"]].sort_values(["fold", "n"])
for fold, g in p.sort_values("n").groupby("fold"):
    last_out = None
    for r in g.itertuples():
        if r.n > 1 and r.prev_hash != last_out:
            wrong += 1
        if r.prev_hash is not None and r.n > 1:
            others = acc[(acc["fold"] != fold) & (acc["out_hash"] == r.prev_hash)]
            leaks += len(others)
        accepted_all = p[(p["n"] == r.n)]["success"].all()
        if accepted_all:
            last_out = r.out_hash
print(f"\nSTATE fold k receives its own last accepted state: violations = {wrong}; states from another fold = {leaks}")

# per-fold table and figures
VARIANTS = [("march", "março: paralelo, sem reaproveitamento"), ("parallel", "outubro: paralelo, com reaproveitamento"),
            ("noparallel", "outubro: --no-parallel")]
rows = []
for mode, title in VARIANTS:
    if mode in steps:
        s = steps[mode]
        rows.append(s[s["success"]].assign(versao=title)[["versao", "fold", "n", "mse_train", "mse_test", "r2_train",
                                                           "r2_test", "n_attempts", "accepted_success",
                                                           "accepted_nit", "accepted_message"]])
OUT.mkdir(parents=True, exist_ok=True)
pd.concat(rows).to_csv(OUT / "sync_folds.csv", index=False)


def figure(cols, log, path):
    present = [(m, t) for m, t in VARIANTS if m in steps]
    fig, axes = plt.subplots(len(present), 2, figsize=(10, 3.2 * len(present)), facecolor=SURFACE,
                             layout="constrained", sharey=True, squeeze=False)
    for i, (mode, title) in enumerate(present):
        s = steps[mode][steps[mode]["success"]]
        for j, (col, label) in enumerate(cols):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            last = s.sort_values("n").groupby("fold")[col].last()
            scale = np.log10(last) if log else last
            spread = scale.max() - scale.min()
            for fold, g in s.groupby("fold"):
                g = g.sort_values("n")
                ax.plot(g["n"], g[col], color=FOLD_COLORS[fold], lw=2, marker="o", ms=3, label=f"fold {fold}")
                # direct label only for a fold that stands apart from the others (the legend covers the rest)
                if spread > (1.0 if log else 0.5) and abs(scale[fold] - scale.drop(fold).median()) > 0.5 * spread:
                    ax.annotate(f"fold {fold}", (g["n"].iloc[-1], g[col].iloc[-1]), xytext=(4, 0),
                                textcoords="offset points", va="center", fontsize=7.5, color=INK_MUTED)
            if log:
                ax.set_yscale("log")
            ax.set_title(f"{title} | {label}", color=INK, fontsize=9.5, loc="left")
            ax.set_xlabel("n (neurônios)", color=INK_MUTED, fontsize=8.5)
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_MUTED, labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False, fontsize=9, labelcolor=INK)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


figure([("r2_train", "R² de treino"), ("r2_test", "R² de teste")], False, OUT / "sync_folds_r2.png")
figure([("mse_train", "MSE de treino"), ("mse_test", "MSE de teste")], True, OUT / "sync_folds_mse.png")
print(f"\nwritten sync_folds.csv, sync_folds_r2.png, sync_folds_mse.png to {OUT}")

"""
Figures of the manuscript.   python paper/make_figures.py
  fig_surfaces.pdf : true P(Y=1|x) of the synthetic 'and' target and the fits of a free, a sign-constrained and a
                     cegis-certified SLFN with the same number of hidden units; hatched = region where a gain has
                     the wrong sign.
  fig_syn.pdf      : error against the true probability (synthetic data) by method, from results_syn*.jsonl.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "figure.dpi": 150})


def surfaces(n=8, seed=8):
    from revision.data import SYNTHETIC, load
    from revision.methods import Model, Cfg
    from revision import sfnn, certify
    X, y, s = load("syn_and"); f = SYNTHETIC["syn_and"][2]
    g = np.linspace(0, 1, 201); XX, YY = np.meshgrid(g, g); G = np.column_stack([XX.ravel(), YY.ravel()])
    panels = [("True $P(Y=1\\mid x)$", None)]
    for title, meth, mode in [("Unconstrained", "RIXM", "mean"), ("Sign constraints", "RIXM-C", "global"),
                              ("Certified (cegis)", "RIXM-C", "cegis")]:
        m = Model(meth, s, Cfg(constraint_mode=mode, max_reinit=5), seed=seed)
        An = m.anchors(X) if m.constrained else None
        th, _ = m.step(n, X, y, None, An)
        if mode == "cegis":
            th, _ = m.certify_and_repair(th, n, X, y, An)
        st = certify.certify(th, n, 2, s)[0]
        viol = ((sfnn.gains(th, G, n) * np.array(s)) < 0).any(axis=1)
        mae = np.abs(m.predict(th, G, n) - f(G)).mean()
        panels.append((f"{title}\nMAE {mae:.3f}, {st}", (m.predict(th, G, n), viol)))
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1), sharey=True)
    for ax, (title, res) in zip(axs, panels):
        P = f(G) if res is None else res[0]
        cs = ax.contourf(XX, YY, P.reshape(XX.shape), levels=np.linspace(0, 1, 11), cmap="Greys", vmin=0, vmax=1.15)
        ax.contour(XX, YY, P.reshape(XX.shape), levels=[0.5], colors="k", linewidths=1.0)
        if res is not None and res[1].any():
            ax.contourf(XX, YY, res[1].reshape(XX.shape).astype(float), levels=[0.5, 1.5], colors="none",
                        hatches=["////"], alpha=0)
            ax.contour(XX, YY, res[1].reshape(XX.shape).astype(float), levels=[0.5], colors="tab:red", linewidths=0.8)
        ax.set_title(title); ax.set_xlabel("$x_1$"); ax.set_aspect("equal")
    axs[0].set_ylabel("$x_2$")
    fig.tight_layout()
    fig.savefig(OUT / "fig_surfaces.pdf"); fig.savefig(OUT / "fig_surfaces.png", dpi=300)
    print("wrote fig_surfaces")


def syn_errors():
    import pandas as pd
    rows = []
    for tag, mode in [("syn", None), ("syn_cegis", "cegis"), ("syn_global", "sign")]:
        f = ROOT / "results" / "revision" / f"results_{tag}.jsonl"
        if not f.exists():
            continue
        for l in open(f):
            r = json.loads(l)
            if "error" in r or "mae_true" not in r:
                continue
            lab = r["method"] + (" (DCV)" if r["selection"] == "dcv" else "")
            if r["method"].endswith("-C") and r["method"] != "ELM-C" and r["method"] not in ("LR-C", "XGB-C", "LGBM-C"):
                lab += " [" + (mode or "mean") + "]"
            rows.append({"dataset": r["dataset"].replace("syn_", ""), "label": lab, "mae": r["mae_true"]})
    if not rows:
        print("no synthetic results yet"); return
    df = pd.DataFrame(rows)
    keep = ["WILCAR-C (DCV) [cegis]", "RIXM-C [cegis]", "WILCAR-C (DCV) [sign]", "RIXM-C [sign]", "WILCAR-C (DCV) [mean]",
            "WILCAR (DCV)", "RIXM", "MLP", "LGBM-C", "XGB-C", "LR-C", "MINMAX", "CMNN", "LMN"]
    df = df[df.label.isin(keep)]
    piv = df.groupby(["label", "dataset"]).mae.mean().unstack()
    piv = piv.reindex([k for k in keep if k in piv.index])
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ds = list(piv.columns); w = 0.8 / len(piv.index)
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(piv.index):
        ax.bar(np.arange(len(ds)) + (i - len(piv.index) / 2) * w, piv.loc[lab].values, w, label=lab, color=cmap(i % 20))
    ax.set_xticks(np.arange(len(ds))); ax.set_xticklabels(ds); ax.set_ylabel("MAE vs. true $P(Y=1\\mid x)$")
    ax.legend(ncol=4, fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig_syn.pdf"); fig.savefig(OUT / "fig_syn.png", dpi=300)
    print("wrote fig_syn")


def capacity(sizes=(1, 2, 3, 4, 6, 8, 12, 16), seed=0):
    from revision.data import SYNTHETIC, load
    from revision.methods import Model, Cfg
    from revision import certify
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
    styles = {"unconstrained": ("RIXM", "mean", "tab:gray", "o"), "sign": ("RIXM-C", "global", "tab:blue", "s"),
              "cegis": ("RIXM-C", "cegis", "tab:red", "^")}
    for ax, key in zip(axs, ["syn_and", "syn_wave"]):
        X, y, s = load(key); f = SYNTHETIC[key][2]
        G = np.random.default_rng(1).random((20000, 2)); pt = f(G)
        for name, (meth, mode, col, mk) in styles.items():
            err, cert = [], []
            for n in sizes:
                best = None
                for r in range(3):          # best of three initialisations by the TRAINING objective
                    m = Model(meth, s, Cfg(constraint_mode=mode, max_reinit=5), seed=seed + 100 * r + n)
                    An = m.anchors(X) if m.constrained else None
                    th, _ = m.step(n, X, y, None, An)
                    if mode == "cegis":
                        th, _ = m.certify_and_repair(th, n, X, y, An)
                    J = m._obj(X, y, n)(th)[0]
                    if best is None or J < best[0]:
                        best = (J, m, th)
                _, m, th = best
                err.append(np.abs(m.predict(th, G, n) - pt).mean())
                cert.append(certify.certify(th, n, 2, s)[0] == "certified")
            ax.plot(sizes, err, "-", color=col, lw=1, label=name)
            for n_, e_, c_ in zip(sizes, err, cert):
                ax.plot(n_, e_, mk, color=col, ms=4, mfc=col if c_ else "white")
            print(key, name, np.round(err, 3), cert, flush=True)
        ax.set_title(key.replace("syn_", "")); ax.set_xlabel("hidden units $n$"); ax.set_xscale("log", base=2)
        ax.set_xticks(sizes); ax.set_xticklabels([str(v) for v in sizes])
    axs[0].set_ylabel("MAE vs. true $P(Y=1\\mid x)$"); axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_capacity.pdf"); fig.savefig(OUT / "fig_capacity.png", dpi=300)
    print("wrote fig_capacity")


if __name__ == "__main__":
    capacity()
    surfaces()

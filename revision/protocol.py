"""
Nested evaluation protocol, baselines and metrics for the Neural Networks revision.

Outer loop : stratified K_out-fold CV repeated over seeds; the outer test fold is
             used once, for the final metrics.
Inner loop : model selection inside the outer training set only
             - 'holdout': stratified 20 % validation split (constructive stopping by MCC)
             - 'dcv'    : Dynamic Cross-Validation over K_in inner folds (sum of validation BCE)
Final model: refitted on the whole outer training set with the selected n*.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (matthews_corrcoef, accuracy_score, f1_score, roc_auc_score,
                             brier_score_loss, log_loss)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from .methods import Model, Cfg
from .mono_baselines import MONO_BASELINES, fit as mono_fit


# ============================================================================ metrics
def cls_metrics(y, prob):
    prob = np.clip(np.asarray(prob, float), 0, 1)
    yhat = (prob >= 0.5).astype(int)
    out = {"mcc": matthews_corrcoef(y, yhat), "acc": accuracy_score(y, yhat),
           "f1": f1_score(y, yhat, zero_division=0), "brier": brier_score_loss(y, prob)}
    try:
        out["auc"] = roc_auc_score(y, prob)
    except ValueError:
        out["auc"] = float("nan")
    return out


def fd_gains(f, X, h=1e-3):
    """Central finite-difference gains of any probability function f: (M, p) -> (M,)."""
    G = np.empty_like(X, dtype=float)
    for i in range(X.shape[1]):
        E = np.zeros(X.shape[1]); E[i] = h
        G[:, i] = (f(X + E) - f(X - E)) / (2 * h)
    return G


def monotonicity(f, signals, X_test, n_uniform=10_000, seed=0):
    """
    Violation rates of the expected gain signs.
    - viol_test / viol_domain: share of (point, constrained variable) pairs with s_i * g_i < 0
      on the test points / on uniform points of [0, 1]^p (training box after min-max scaling);
    - vars_violated_domain: share of constrained variables violated at least once in the domain;
    - scr_legacy: SCR as in the dissertation (sign of the average finite-difference gain, delta = 0.1).
    """
    s = np.asarray(signals, float); idx = np.flatnonzero(s)
    if not len(idx):
        return {"viol_test": 0.0, "viol_domain": 0.0, "vars_violated_domain": 0.0, "scr_legacy": 1.0}
    rng = np.random.default_rng(seed)
    U = rng.random((n_uniform, X_test.shape[1]))
    tol = -1e-9
    gt = fd_gains(f, X_test)[:, idx] * s[idx]
    gu = fd_gains(f, U)[:, idx] * s[idx]
    base = f(X_test); legacy = []
    for i in idx:
        Xp = X_test.copy(); Xp[:, i] += 0.1
        legacy.append(np.sign(np.mean((f(Xp) - base) / 0.1)) == s[i])
    return {"viol_test": float(np.mean(gt < tol)), "viol_domain": float(np.mean(gu < tol)),
            "vars_violated_domain": float(np.mean((gu < tol).any(axis=0))),
            "scr_legacy": float(np.mean(legacy))}


# ============================================================================ constructive selection
def _bce(y, p):
    return log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])


def select_holdout(make, Xtr, ytr, Xva, yva):
    """Constructive loop on (Xtr, ytr); stopping and n* by validation MCC."""
    m = make(); m.start(Xtr, ytr); An = m.anchors(Xtr) if m.constrained else None
    best, best_n, wait, prev, curve = -np.inf, None, 0, None, []
    t0 = time.time()
    for n in range(1, m.max_neurons + 1):
        if m.cfg.time_budget and time.time() - t0 > m.cfg.time_budget:
            curve.append((n, "budget")); break
        state, info = m.step(n, Xtr, ytr, prev, An)
        prev = state                        # keeps shapes consistent for weight reuse
        if not info["feasible"]:
            curve.append((n, None)); wait += 1
            if wait >= m.patience:
                break
            continue
        score = matthews_corrcoef(yva, (m.predict(state, Xva, n) >= 0.5).astype(int))
        curve.append((n, score))
        if score > best:
            best, best_n, wait = score, n, 0
        else:
            wait += 1
            if wait >= m.patience:
                break
    return best_n or 1, curve


def select_dcv(make, X, y, k_in, seed):
    """Dynamic Cross-Validation: one model per inner fold, advanced together; n* = argmin sum of validation BCE."""
    folds = list(StratifiedKFold(k_in, shuffle=True, random_state=seed).split(X, y))
    models, prevs, anchors = [], [None] * k_in, []
    for tr, _ in folds:
        m = make(); m.start(X[tr], y[tr]); models.append(m)
        anchors.append(m.anchors(X[tr]) if m.constrained else None)
    best, best_n, wait, curve = np.inf, None, 0, []
    t0 = time.time()
    for n in range(1, models[0].max_neurons + 1):
        if models[0].cfg.time_budget and time.time() - t0 > models[0].cfg.time_budget:
            curve.append((n, "budget")); break
        J, ok = 0.0, True
        for f, (tr, va) in enumerate(folds):
            state, info = models[f].step(n, X[tr], y[tr], prevs[f], anchors[f])
            prevs[f] = state
            ok = ok and info["feasible"]
            J += _bce(y[va], models[f].predict(state, X[va], n))
        curve.append((n, J if ok else None))
        if ok and J < best:
            best, best_n, wait = J, n, 0
        else:
            wait += 1
            if wait >= models[0].patience:
                break
    return best_n or 1, curve


def refit(make, n_star, X, y):
    """Train the final model with n* neurons on the whole outer training set."""
    m = make(); m.start(X, y); An = m.anchors(X) if m.constrained else None
    prev, info = None, {"feasible": True}
    sizes = range(1, n_star + 1) if m.reuse else [n_star]
    for n in sizes:
        state, info = m.step(n, X, y, prev, An)
        prev = state
    cert = {}
    if m.base != "ELM":                     # sound certificate on [0,1]^p (cegis: with repair)
        if len(m.idx):
            prev, cert = m.certify_and_repair(prev, n_star, X, y, An)
    elif m.constrained and m._aligned:
        cert = {"cert_status": "by_construction"}
    return m, prev, n_star, info["feasible"], cert


# ============================================================================ baselines
def _logreg_fit(X, y, lam, bounds):
    p = X.shape[1]
    def obj(w):
        z = X @ w[:-1] + w[-1]
        o = 1 / (1 + np.exp(-np.clip(z, -500, 500))); oc = np.clip(o, 1e-12, 1 - 1e-12)
        loss = -np.mean(y * np.log(oc) + (1 - y) * np.log(1 - oc)) + lam * np.sum(w[:-1] ** 2)
        g = np.append(X.T @ (o - y) / len(y) + 2 * lam * w[:-1], np.mean(o - y))
        return loss, g
    r = minimize(obj, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=bounds + [(None, None)])
    w = r.x
    return lambda Z: 1 / (1 + np.exp(-np.clip(Z @ w[:-1] + w[-1], -500, 500)))


def baseline(name, signals, Xtr, ytr, Xva, yva, seed):
    """Fit a baseline with hyper-parameters chosen on (Xva, yva); return a predictor refitted on train+val."""
    s = np.asarray(signals)
    Xall, yall = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])
    score = lambda f: matthews_corrcoef(yva, (f(Xva) >= 0.5).astype(int))
    if name in ("LR", "LR-C"):
        bounds = [((0, None) if si > 0 else (None, 0) if si < 0 else (None, None)) if name == "LR-C" else (None, None) for si in s]
        grid = [1e-4, 1e-3, 1e-2, 1e-1]
        lam = max(grid, key=lambda l: score(_logreg_fit(Xtr, ytr, l, bounds)))
        return _logreg_fit(Xall, yall, lam, bounds), {"lambda": lam}
    if name in ("XGB", "XGB-C"):
        from xgboost import XGBClassifier
        mono = "(" + ",".join(str(int(v)) for v in s) + ")" if name == "XGB-C" else None
        grid = [(d, ne) for d in (2, 3, 4) for ne in (100, 300)]
        def fit(Xa, ya, d, ne):
            kw = dict(max_depth=d, n_estimators=ne, learning_rate=0.05, subsample=0.9, random_state=seed,
                      n_jobs=1, verbosity=0)
            if mono:
                kw["monotone_constraints"] = mono
            clf = XGBClassifier(**kw).fit(Xa, ya)
            return lambda Z: clf.predict_proba(Z)[:, 1]
        d, ne = max(grid, key=lambda g: score(fit(Xtr, ytr, *g)))
        return fit(Xall, yall, d, ne), {"max_depth": d, "n_estimators": ne}
    if name in ("LGBM", "LGBM-C"):
        from lightgbm import LGBMClassifier
        grid = [(nl, ne) for nl in (4, 8, 16) for ne in (100, 300)]
        def fit(Xa, ya, nl, ne):
            kw = dict(num_leaves=nl, n_estimators=ne, learning_rate=0.05, subsample=0.9, subsample_freq=1,
                      random_state=seed, n_jobs=1, verbose=-1, min_child_samples=5)
            if name == "LGBM-C":
                kw["monotone_constraints"] = [int(v) for v in s]
            clf = LGBMClassifier(**kw).fit(Xa, ya)
            return lambda Z: clf.predict_proba(Z)[:, 1]
        nl, ne = max(grid, key=lambda g: score(fit(Xtr, ytr, *g)))
        return fit(Xall, yall, nl, ne), {"num_leaves": nl, "n_estimators": ne}
    if name in MONO_BASELINES:
        return mono_fit(name, s, Xtr, ytr, Xva, yva, seed)
    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        grid = [1, 2, 4, 8, 16, 32, 64]
        def fit(Xa, ya, h):
            clf = MLPClassifier((h,), activation="logistic", max_iter=3000, early_stopping=False,
                                random_state=seed).fit(Xa, ya)
            return lambda Z: clf.predict_proba(Z)[:, 1]
        h = max(grid, key=lambda g: score(fit(Xtr, ytr, g)))
        return fit(Xall, yall, h), {"hidden": h}
    raise ValueError(name)


BASELINES = ("LR", "LR-C", "XGB", "XGB-C", "LGBM", "LGBM-C", "MLP") + MONO_BASELINES


# ============================================================================ evaluation
CERTIFIED_BY_CONSTRUCTION = {"LR-C", "XGB-C", "LGBM-C", "MINMAX", "CMNN", "LMN"}


def evaluate(Xtr_raw, ytr, Xte_raw, yte, signals, method, selection, seed, k_in=4, cfg: Cfg | None = None):
    """Scale on the training part, select, refit, evaluate once on the test part."""
    cfg = cfg or Cfg()
    sc = MinMaxScaler().fit(Xtr_raw)                   # scaler fitted on the training part only
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)
    t0 = time.time()
    extra, cert = {}, {}
    if method in BASELINES:
        itr, iva = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed).split(Xtr, ytr))
        f, extra = baseline(method, signals, Xtr[itr], ytr[itr], Xtr[iva], ytr[iva], seed)
        n_star, feasible = None, True
        if method in CERTIFIED_BY_CONSTRUCTION and any(signals):
            cert = {"cert_status": "by_construction"}
    else:
        make = lambda: Model(method, signals, cfg, seed)
        if selection == "dcv":
            n_star, curve = select_dcv(make, Xtr, ytr, k_in, seed)
        else:
            itr, iva = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed).split(Xtr, ytr))
            n_star, curve = select_holdout(make, Xtr[itr], ytr[itr], Xtr[iva], ytr[iva])
        m, state, n_star, feasible, cert = refit(make, n_star, Xtr, ytr)
        f = lambda Z: m.predict(state, Z, n_star)
        extra = {"curve_len": len(curve), "budget_hit": any(c[1] == "budget" for c in curve)}
    prob = f(Xte)
    st = cert.get("cert_status", "none")
    row = {"method": method, "selection": selection if method not in BASELINES else "holdout",
           "n_star": n_star, "feasible": feasible, "time_s": time.time() - t0,
           **cls_metrics(yte, prob), **monotonicity(f, signals, Xte, seed=seed),
           "certified": {"certified": 1.0, "by_construction": 1.0, "counterexample": 0.0}.get(st, float("nan")),
           **cert, **extra}
    row["cert_status"] = st
    return row


def run_fold(X, y, signals, method, selection, rep, fold, k_out=5, k_in=4, cfg: Cfg | None = None):
    skf = StratifiedKFold(k_out, shuffle=True, random_state=rep)
    tr, te = list(skf.split(X, y))[fold]
    row = evaluate(X[tr], y[tr], X[te], y[te], signals, method, selection, 1000 * rep + fold, k_in, cfg)
    return {**row, "rep": rep, "fold": fold}

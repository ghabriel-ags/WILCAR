"""
Verification of partial monotonicity for a sigmoid SLFN  p(x) = sigma(W2 . sigma(W1 x + b1) + b2)
on the box B = [0, 1]^p (the training box after min-max scaling).

For a constrained input i with expected sign s_i,

    s_i * dp/dx_i = sigma'(z2) * h_i(x),    h_i(x) = sum_j c_ij sigma'(w_j . x + b_j),   c_ij = s_i W2_j W1_ji,

and sigma'(z2) > 0, so monotonicity in x_i on B  <=>  min_{x in B} h_i(x) >= 0.

* `adversarial`: multi-start projected L-BFGS on h_i (analytic gradient) - cheap counterexample search.
* `certify`   : sound branch-and-bound. On a sub-box, z_j lies in [zl_j, zu_j] (exact interval of an affine
                map), and sigma' is unimodal with its maximum at 0, so on that interval
                    min sigma' = min(sigma'(zl), sigma'(zu)),   max sigma' = sigma'(clip(0, zl, zu)).
                A lower bound of h_i is  LB = sum_{c>=0} c * min sigma' + sum_{c<0} c * max sigma', tightened by
                the mean-value form  h(mid) - sum_d G_d r_d  with  G_d = sum_j |c_j| max|sigma''(z_j)| |W1_jd|
                (quadratic convergence as boxes shrink).
                LB >= 0 certifies the sub-box; a sub-box whose centre has h_i < 0 is a counterexample;
                otherwise the box is split along the coordinate with the largest width x sensitivity.
                Returns 'certified', 'counterexample' (with the point) or 'unknown' (node budget exhausted).
If c_ij >= 0 for every j the root box is certified immediately (this is the 'global' mode).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from . import sfnn


def _dsig(z):
    s = sfnn.sigmoid(z)
    return s * (1 - s)


_ZPK = np.log(2 + np.sqrt(3))            # |sigma''| peaks at z = +-1.3170 ...
_D2MAX = np.sqrt(3) / 18                 # ... with value 0.0962


def _d2abs_max(zl, zu):
    """max |sigma''(z)| on [zl, zu] (|sigma''| is even, increasing on [0, z*], decreasing after)."""
    def d2(z):
        s = sfnn.sigmoid(z); return np.abs(s * (1 - s) * (1 - 2 * s))
    m = np.maximum(d2(zl), d2(zu))
    hit = ((zl <= _ZPK) & (zu >= _ZPK)) | ((zl <= -_ZPK) & (zu >= -_ZPK))
    return np.where(hit, _D2MAX, m)


def coefficients(theta, n, p, signals):
    W1, b1, W2, _ = sfnn.unpack(theta, n, p)
    s = np.asarray(signals, float); idx = np.flatnonzero(s)
    C = (s[idx][:, None] * W2[None, :] * W1[:, idx].T)          # (|idx|, n)
    return W1, b1, idx, C


def h_value(x, W1, b1, c):
    return _dsig(W1 @ x + b1) @ c


def h_grad(x, W1, b1, c):
    z = W1 @ x + b1; s = sfnn.sigmoid(z); d = s * (1 - s)
    return float(d @ c), (c * d * (1 - 2 * s)) @ W1


def adversarial(theta, n, p, signals, rng, starts=16, X0=None, tol=0.0):
    """Return an array of points in [0,1]^p where some constrained gain has the wrong sign (possibly empty)."""
    W1, b1, idx, C = coefficients(theta, n, p, signals)
    out = []
    for a in range(len(idx)):
        c = C[a]
        if np.all(c >= 0):
            continue
        S = rng.random((starts, p))
        if X0 is not None and len(X0):
            hv = _dsig(X0 @ W1.T + b1) @ c
            S = np.vstack([S, X0[np.argsort(hv)[:starts // 2]]])
        best = None
        for x0 in S:
            r = minimize(h_grad, x0, args=(W1, b1, c), jac=True, method="L-BFGS-B", bounds=[(0, 1)] * p,
                         options={"maxiter": 200})
            if r.fun < tol and (best is None or r.fun < best[0]):
                best = (r.fun, r.x)
        if best is not None:
            out.append(best[1])
    return np.array(out).reshape(-1, p)


def _d2(z):
    sg = sfnn.sigmoid(z)
    return sg * (1 - sg) * (1 - 2 * sg)


def _d2_range(zl, zu):
    """Range [lo, hi] of sigma'' on [zl, zu] (sigma'' is odd, maximal at -t*, minimal at +t*)."""
    a, b = _d2(zl), _d2(zu)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    lo = np.where((zl <= _ZPK) & (zu >= _ZPK), -_D2MAX, lo)
    hi = np.where((zl <= -_ZPK) & (zu >= -_ZPK), _D2MAX, hi)
    return lo, hi


def _box_bounds(l, u, W1, b1, c, absW, K, mv=True):
    """Bounds of h on a batch of boxes: natural interval extension, gradient enclosure, mean-value form."""
    mid, rad = (l + u) / 2, (u - l) / 2
    zc = mid @ W1.T + b1; zr = rad @ absW.T
    zl, zu = zc - zr, zc + zr
    dl, du = _dsig(zl), _dsig(zu)
    LB = np.where(c >= 0, c * np.minimum(dl, du), c * _dsig(np.clip(0.0, zl, zu))).sum(axis=1)
    lo, hi = _d2_range(zl, zu)                                          # (B, n)
    t1, t2 = K[None] * lo[:, :, None], K[None] * hi[:, :, None]          # (B, n, p)
    Glo, Ghi = np.minimum(t1, t2).sum(axis=1), np.maximum(t1, t2).sum(axis=1)   # enclosure of dh/dx_d on the box
    hc = _dsig(zc) @ c
    if mv:
        LB = np.maximum(LB, hc - (np.maximum(np.abs(Glo), np.abs(Ghi)) * rad).sum(axis=1))
    return LB, hc, mid, Glo, Ghi


def _bb_one(W1, b1, c, p, budget, batch=4096, mv=True, mono=True):
    """
    Input-space branch and bound (interval global optimisation, cf. Hansen & Walster): natural interval extension and
    mean-value form with an enclosure of the gradient; monotonicity test -- if dh/dx_d has a fixed sign on a box, the
    minimum lies on the corresponding face and the box collapses along d.
    """
    if np.all(c >= 0):
        return "certified", None, 1
    absW = np.abs(W1)
    K = c[:, None] * W1                                                # (n, p): coefficients of sigma''_j in dh/dx_d
    L = np.zeros((1, p)); U = np.ones((1, p)); nodes = 0
    while len(L):
        take = min(batch, len(L))
        l, u = L[-take:].copy(), U[-take:].copy(); L, U = L[:-take], U[:-take]
        nodes += take
        # monotonicity test (applied twice: collapsing some coordinates tightens the enclosure of the others)
        for _ in range(2 if mono else 0):
            _, _, _, Glo, Ghi = _box_bounds(l, u, W1, b1, c, absW, K)
            inc, dec = Glo > 0, Ghi < 0                                # h increasing / decreasing in x_d on the box
            u = np.where(inc, l, u); l = np.where(dec, u, l)
        LB, hc, mid, Glo, Ghi = _box_bounds(l, u, W1, b1, c, absW, K, mv=mv)
        open_ = LB < 0
        if not open_.any():
            continue
        l, u, mid, hc = l[open_], u[open_], mid[open_], hc[open_]
        Glo, Ghi = Glo[open_], Ghi[open_]
        if (hc < 0).any():
            return "counterexample", mid[np.argmin(hc)], nodes
        if nodes > budget:
            return "unknown", None, nodes
        width = (u - l) * np.maximum(np.abs(Glo), np.abs(Ghi))
        d = np.argmax(width, axis=1)
        r = np.arange(len(l)); m = (l[r, d] + u[r, d]) / 2
        l2, u1 = l.copy(), u.copy(); u1[r, d] = m; l2[r, d] = m
        L = np.vstack([L, l, l2]); U = np.vstack([U, u1, u])
    return "certified", None, nodes


def _bb_z(W1, b1, c, p, max_lp=3000):
    """
    Branch-and-bound in pre-activation space (used when n < p). h(z) = sum_j c_j sigma'(z_j) is separable,
    so its minimum over a z-box is exact: sum_j min_{[zl_j, zu_j]} c_j sigma'. The reachable set is the zonotope
    Z = {W1 x + b1 : x in [0,1]^p}; a z-box is discarded if the minimum of h on it is >= 0 or if it does not
    meet Z (LP feasibility). A reachable point with h < 0 is a counterexample. Sound and, up to the budget,
    complete.
    """
    import heapq
    from scipy.optimize import linprog
    if np.all(c >= 0):
        return "certified", None, 1
    n = len(c)
    zl0 = b1 + np.minimum(W1, 0).sum(axis=1); zu0 = b1 + np.maximum(W1, 0).sum(axis=1)

    def lower(zl, zu):
        dl, du = _dsig(zl), _dsig(zu)
        return float(np.where(c >= 0, c * np.minimum(dl, du), c * _dsig(np.clip(0.0, zl, zu))).sum())

    A = np.vstack([W1, -W1]); bounds = [(0, 1)] * p
    heap = [(lower(zl0, zu0), 0, zl0, zu0)]; k = 1; lps = 0
    while heap:
        lb, _, zl, zu = heapq.heappop(heap)
        if lb >= 0:
            continue
        zc = (zl + zu) / 2
        sg = sfnn.sigmoid(zc)
        obj = W1.T @ (c * sg * (1 - sg) * (1 - 2 * sg))           # linearisation of h at the box centre
        r = linprog(obj, A_ub=A, b_ub=np.concatenate([zu - b1, b1 - zl]), bounds=bounds, method="highs")
        lps += 1
        if r.status == 2:                                          # infeasible: box outside the zonotope
            continue
        if r.status == 0:
            x = np.clip(r.x, 0, 1)
            if h_value(x, W1, b1, c) < 0:
                return "counterexample", x, lps
        if lps >= max_lp:
            return "unknown", None, lps
        j = int(np.argmax((zu - zl) * np.abs(c)))
        m = (zl[j] + zu[j]) / 2
        for a_, b_ in ((zl.copy(), np.where(np.arange(n) == j, m, zu)), (np.where(np.arange(n) == j, m, zl), zu.copy())):
            lb2 = lower(a_, b_)
            if lb2 < 0:
                heapq.heappush(heap, (lb2, k, a_, b_)); k += 1
    return "certified", None, lps


def certify(theta, n, p, signals, budget=200_000, mv=True, mono=True, zspace=True):
    """
    Sound certificate over [0,1]^p for every constrained input. Returns (status, counterexamples, nodes).
    mv / mono / zspace switch off the mean-value form, the monotonicity test and the pre-activation-space search
    (used only by the ablation in revision/cert_bench.py).
    """
    W1, b1, idx, C = coefficients(theta, n, p, signals)
    status, cex, total = "certified", [], 0
    for a in range(len(idx)):
        st, x, nd = _bb_one(W1, b1, C[a], p, budget // 4 if (n < p and zspace) else budget, mv=mv, mono=mono)
        if st == "unknown" and n < p and zspace:
            st, x, nd2 = _bb_z(W1, b1, C[a], p); nd += nd2
        total += nd
        if st == "counterexample":
            status = "counterexample"; cex.append(x)
        elif st == "unknown" and status == "certified":
            status = "unknown"
    return status, np.array(cex).reshape(-1, p), total

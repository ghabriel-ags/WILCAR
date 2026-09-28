"""
Constructive methods (WILCAR, RIXM, ELM and their -C variants) re-implemented
for the nested protocol of the Neural Networks revision.

Differences from src/ (documented in revision/README.md):
* no method ever sees test data: constraints and the reinitialisation check use
  training anchors only;
* SLSQP receives analytic gradients of the loss and of the gain constraints;
* constraint mode 'mean' (average gain over the training data >= eps, as in the
  dissertation) or 'multi' (gain >= eps at every one of M k-means anchors);
* optional zero initialisation of the new neuron's output weight (WILCAR only),
  which makes the n+1 network start exactly at the n network;
* ELM-C uses a sigmoid output with BCE (logistic ELM); the dissertation version
  applied BCE to a clipped linear output.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np
from numpy.linalg import pinv
from scipy.optimize import minimize

from . import sfnn


@dataclass
class Cfg:
    max_neurons: int = 750
    patience: int = 150
    max_neurons_c: int = 50
    patience_c: int = 15
    lr: float = 0.1
    lr_decay: float = 0.999
    min_lr: float = 1e-3
    epochs: int = 2501
    es_patience: int = 250
    tol: float = 1e-6
    eps_mean: float = 0.05
    eps_point: float = 1e-3
    constraint_mode: str = "mean"      # 'mean' | 'multi' | 'global'
    n_anchors: int = 10
    max_reinit: int = 100
    slsqp_maxiter: int = 1000
    zero_out: bool = False             # WILCAR: new neuron's output weight starts at 0


METHODS = ("WILCAR", "WILCAR-C", "RIXM", "RIXM-C", "ELM", "ELM-C")


class Model:
    """A constructive method. `step` trains the n-neuron network, reusing `prev` when the method reuses weights."""

    def __init__(self, name: str, signals, cfg: Cfg, seed: int):
        assert name in METHODS, name
        self.name = name
        self.base = name.split("-")[0]
        self.constrained = name.endswith("-C")
        self.reuse = self.base == "WILCAR"
        self.s = np.asarray(signals, dtype=float)
        self.idx = np.flatnonzero(self.s)
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.first = None

    # ------------------------------------------------------------------ helpers
    @property
    def max_neurons(self):
        return self.cfg.max_neurons_c if self.constrained else self.cfg.max_neurons

    @property
    def patience(self):
        return self.cfg.patience_c if self.constrained else self.cfg.patience

    def anchors(self, X):
        if self.cfg.constraint_mode == "mean" or len(X) <= self.cfg.n_anchors:
            return X
        from sklearn.cluster import KMeans
        km = KMeans(self.cfg.n_anchors, n_init=4, random_state=int(self.rng.integers(1 << 31))).fit(X)
        return km.cluster_centers_

    # ------------------------------------------------------------------ WILCAR first neuron (linearisation)
    def start(self, X, y):
        if self.base != "WILCAR":
            return
        from scipy.optimize import fsolve
        p = X.shape[1]
        A = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        b, k = coef[0], coef[1:]
        xM = X.mean(axis=0)

        def eqs(v):
            w, beta = v[:-1], v[-1]
            sg = sfnn.sigmoid(w @ xM + beta); sp = sg * (1 - sg)
            return np.concatenate([[sg - sp * (w @ xM) - b], sp * w - k])

        best, best_res = None, np.inf
        if not self.constrained:
            for t in range(5):
                v0 = np.append(k, b) if t == 0 else self.rng.normal(size=p + 1) * 0.5
                v, info, ier, _ = fsolve(eqs, v0, full_output=True)
                r = np.linalg.norm(info["fvec"])
                if r < best_res:
                    best, best_res = v, r
        else:
            obj = lambda v: float(np.sum(eqs(v) ** 2))
            cons = []
            for i in self.idx:
                def c(v, i=i):
                    w, beta = v[:-1], v[-1]
                    sg = sfnn.sigmoid(w @ xM + beta)
                    return self.s[i] * sg * (1 - sg) * w[i] - self.cfg.eps_mean
                cons.append({"type": "ineq", "fun": c})
            for t in range(10):
                v0 = np.append(k, b) if t == 0 else self.rng.normal(size=p + 1) * 0.5
                r = minimize(obj, v0, method="SLSQP", constraints=cons, options={"maxiter": 5000, "ftol": 1e-9})
                ok = all(cc["fun"](r.x) >= -1e-6 for cc in cons)
                if ok and r.fun < best_res:
                    best, best_res = r.x, r.fun
                if best_res < 1e-8:
                    break
            if best is None:
                best = np.append(k, b)
        self.first = best

    # ------------------------------------------------------------------ initial parameters
    def _init_theta(self, n, p, prev):
        rng = self.rng
        if self.reuse and prev is not None and n > 1:
            W1, b1, W2, b2 = sfnn.unpack(prev, n - 1, p)
            w_new = rng.normal(size=(1, p)) * np.sqrt(1.0 / p)
            o_new = 0.0 if self.cfg.zero_out else rng.normal() * np.sqrt(1.0 / n)
            return sfnn.pack(np.vstack([W1, w_new]), np.append(b1, 0.0), np.append(W2, o_new), b2)
        W1 = rng.normal(size=(n, p)) * np.sqrt(1.0 / p)
        b1 = np.zeros(n)
        W2 = rng.normal(size=n) * np.sqrt(1.0 / n)
        if self.base == "WILCAR" and self.first is not None:
            W1[0] = self.first[:-1]
            b1[0] = self.first[-1]
        return sfnn.pack(W1, b1, W2, 0.0)

    # ------------------------------------------------------------------ unconstrained training
    def _backprop(self, theta, X, y, n):
        c = self.cfg
        lr, best, wait = c.lr, np.inf, 0
        for _ in range(c.epochs):
            loss, g = sfnn.bce_grad(theta, X, y, n)
            if loss < best - c.tol:
                best, wait = loss, 0
            else:
                wait += 1
                if wait >= c.es_patience:
                    break
            theta = theta - lr * g
            lr = max(lr * c.lr_decay, c.min_lr)
        return theta

    # ------------------------------------------------------------------ constrained training
    def _global_constraints(self, n, p, elm=None):
        """
        Certified monotonicity: s_i * W2_j * W1_ji >= 0 for every hidden unit j and constrained input i.
        With sigmoid (or ReLU) hidden units and a monotone output this fixes the sign of dp/dx_i everywhere.
        For ELM (fixed hidden weights) the constraint is linear in the output weights: s_i * Win_ij * beta_j >= 0.
        """
        s, idx = self.s, self.idx
        if elm is not None:
            Win = elm[0]
            A = (s[idx][:, None] * Win[idx, :])                      # (|idx|, n)
            rows = np.zeros((len(idx) * n, n + 1))
            for a in range(len(idx)):
                rows[a * n:(a + 1) * n, :n] = np.diag(A[a])
            fun = lambda v: rows @ v
            return {"type": "ineq", "fun": fun, "jac": lambda v: rows}, fun

        def fun(v):
            W1, _, W2, _ = sfnn.unpack(v, n, p)
            return (s[idx][:, None] * W1[:, idx].T * W2[None, :]).ravel()      # (|idx|*n,)

        def jac(v):
            W1, _, W2, _ = sfnn.unpack(v, n, p)
            J = np.zeros((len(idx) * n, sfnn.n_params(n, p)))
            for a, i in enumerate(idx):
                for j in range(n):
                    r = a * n + j
                    J[r, j * p + i] = s[i] * W2[j]                  # d/dW1[j, i]
                    J[r, n * p + n + j] = s[i] * W1[j, i]           # d/dW2[j]
            return J
        return {"type": "ineq", "fun": fun, "jac": jac}, fun

    def _constraints(self, An, n, elm=None):
        c, s, idx = self.cfg, self.s, self.idx
        if c.constraint_mode == "global":
            p = len(s)
            return self._global_constraints(n, p, elm)
        mode_mean = c.constraint_mode == "mean"
        eps = c.eps_mean if mode_mean else c.eps_point

        def gg(v):
            if elm is None:
                return sfnn.gains_and_grad(v, An, n)
            return sfnn.elm_gains_and_grad(v, An, *elm)

        def fun(v):
            g, _ = gg(v)
            g = g[:, idx] * s[idx]
            return (g.mean(axis=0) if mode_mean else g.ravel()) - eps

        def jac(v):
            _, G = gg(v)
            G = G[:, idx, :] * s[idx][None, :, None]
            return G.mean(axis=0) if mode_mean else G.reshape(-1, G.shape[-1])

        return {"type": "ineq", "fun": fun, "jac": jac}, fun

    def _slsqp(self, obj, v0, cons):
        r = minimize(obj, v0, jac=True, method="SLSQP", constraints=[cons],
                     options={"maxiter": self.cfg.slsqp_maxiter, "ftol": self.cfg.tol})
        return r.x

    # ------------------------------------------------------------------ one constructive step
    def step(self, n, X, y, prev=None, An=None):
        """Return (state, info). state is theta (MLP methods) or dict (ELM)."""
        p = X.shape[1]
        if self.base == "ELM":
            return self._step_elm(n, X, y, An)
        if not self.constrained:
            theta = self._backprop(self._init_theta(n, p, prev), X, y, n)
            return theta, {"feasible": True, "attempts": 1}
        if not len(self.idx):
            theta = self._backprop(self._init_theta(n, p, prev), X, y, n)
            return theta, {"feasible": True, "attempts": 1}
        cons, cfun = self._constraints(An, n)
        obj = lambda v: sfnn.bce_grad(v, X, y, n)
        best, best_viol = None, np.inf
        for a in range(self.cfg.max_reinit):
            v = self._slsqp(obj, self._init_theta(n, p, prev), cons)
            viol = max(0.0, -cfun(v).min())
            if viol < best_viol:
                best, best_viol = v, viol
            if viol <= 1e-8:
                return v, {"feasible": True, "attempts": a + 1}
        return best, {"feasible": False, "attempts": self.cfg.max_reinit}

    def _step_elm(self, n, X, y, An):
        p = X.shape[1]
        for a in range(self.cfg.max_reinit if self.constrained else 1):
            Win = self.rng.uniform(-1, 1, (p, n)); b = np.zeros(n)
            H = sfnn.elm_hidden(X, Win, b)
            if not self.constrained:
                beta = pinv(H) @ y                       # original ELM: linear output, MSE
                return {"Win": Win, "b": b, "beta": np.append(beta, 0.0), "linear": True}, {"feasible": True, "attempts": 1}
            Hb = np.column_stack([H, np.ones(len(H))])
            beta0 = pinv(Hb) @ (4.0 * (y - 0.5))         # logit linearisation of the target
            def obj(bt):
                z = Hb @ bt; o = sfnn.sigmoid(z); oc = np.clip(o, 1e-12, 1 - 1e-12)
                loss = -np.mean(y * np.log(oc) + (1 - y) * np.log(1 - oc))
                return loss, Hb.T @ (o - y) / len(y)
            if not len(self.idx):
                beta = self._slsqp(obj, beta0, {"type": "ineq", "fun": lambda v: np.array([1.0]), "jac": lambda v: np.zeros((1, len(v)))})
                return {"Win": Win, "b": b, "beta": beta, "linear": False}, {"feasible": True, "attempts": 1}
            cons, cfun = self._constraints(An, n, elm=(Win, b))
            beta = self._slsqp(obj, beta0, cons)
            if -cfun(beta).min() <= 1e-8:
                return {"Win": Win, "b": b, "beta": beta, "linear": False}, {"feasible": True, "attempts": a + 1}
        return {"Win": Win, "b": b, "beta": beta, "linear": False}, {"feasible": False, "attempts": self.cfg.max_reinit}

    # ------------------------------------------------------------------ inference
    def predict(self, state, X, n):
        if self.base == "ELM":
            H = sfnn.elm_hidden(X, state["Win"], state["b"])
            z = H @ state["beta"][:-1] + state["beta"][-1]
            return np.clip(z, 0.0, 1.0) if state["linear"] else sfnn.sigmoid(z)
        return sfnn.predict(state, X, n)

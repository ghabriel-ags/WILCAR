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
  applied BCE to a clipped linear output;
* unconstrained networks are trained with L-BFGS on the analytic gradient (same
  quasi-Newton family as the SLSQP used for the constrained ones), so that
  constrained vs. unconstrained comparisons are not confounded by the optimiser;
  the dissertation's full-batch gradient descent remains available (optimizer='gd');
* ELM uses random hidden biases and an output bias (standard ELM);
* in 'global' mode, hidden weights of constrained inputs are drawn sign-aligned
  (ELM-C: monotone ELM, beta >= 0; WILCAR-C / RIXM-C: feasible starting point).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np
from numpy.linalg import pinv
from scipy.optimize import minimize

from . import sfnn


@dataclass
class Cfg:
    max_neurons: int = 300
    patience: int = 30
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
    constraint_mode: str = "mean"      # 'mean' | 'multi' | 'global' | 'cegis'
    n_anchors: int = 10
    max_anchors_mean: int = 1000       # 'mean' mode: random subset of the training data when larger
    cegis_rounds: int = 10             # counterexample rounds per constructive step
    cegis_starts: int = 16             # multi-start adversarial search per constrained input
    bb_budget: int = 200_000           # branch-and-bound nodes per constrained input
    time_budget: float = 3600.0        # seconds per constructive selection loop (0 = none)
    max_reinit: int = 100
    slsqp_maxiter: int = 1000
    feas_tol: float = 1e-6             # constraint violation accepted as feasible (SLSQP ftol is 1e-6)
    zero_out: bool = False             # WILCAR: new neuron's output weight starts at 0
    optimizer: str = "lbfgs"           # unconstrained training: 'lbfgs' (quasi-Newton, like SLSQP) | 'gd' (dissertation)
    lbfgs_maxiter: int = 1000
    l2: float = 0.3                    # Gaussian-prior precision lambda0 on W1, W2 (MAP): mean BCE + (lambda0 / N) ||W||^2,
                                       # same in constrained and unconstrained objectives (= 1e-3 at N = 300)
    n_starts: int = 1                  # independent initialisations per constructive step (best training objective)
    constrained_init: str = "free"     # first SLSQP start: 'free' = unconstrained L-BFGS optimum from the usual init
                                       # (then projected by SLSQP); 'plain' = the usual init (reuse / random)
    elm_bias: bool = True              # random hidden biases U(-1,1) + output bias (standard ELM); False = dissertation


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
        self.extra = None                  # counterexample anchors ('cegis')

    # ------------------------------------------------------------------ helpers
    @property
    def max_neurons(self):
        return self.cfg.max_neurons_c if self.constrained else self.cfg.max_neurons

    @property
    def patience(self):
        return self.cfg.patience_c if self.constrained else self.cfg.patience

    def anchors(self, X):
        c = self.cfg
        if c.constraint_mode == "mean":
            if len(X) > c.max_anchors_mean:
                return X[self.rng.choice(len(X), c.max_anchors_mean, replace=False)]
            return X
        if len(X) <= c.n_anchors:
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
    @property
    def _aligned(self):
        m = self.cfg.constraint_mode
        return self.constrained and len(self.idx) > 0 and (m == "global" or (m == "cegis" and self.base == "ELM"))

    def _align(self, W1, W2):
        """Sign-align hidden weights of constrained inputs and make output weights >= 0 (feasible for 'global')."""
        W1 = np.array(W1, dtype=float, ndmin=2); W2 = np.abs(np.atleast_1d(W2).astype(float))
        W1[:, self.idx] = np.abs(W1[:, self.idx]) * self.s[self.idx]
        return W1, W2

    def _init_theta(self, n, p, prev):
        rng = self.rng
        if self.reuse and prev is not None and n > 1:
            W1, b1, W2, b2 = sfnn.unpack(prev, n - 1, p)
            w_new = rng.normal(size=(1, p)) * np.sqrt(1.0 / p)
            o_new = 0.0 if self.cfg.zero_out else rng.normal() * np.sqrt(1.0 / n)
            if self._aligned:
                w_new, o_new = self._align(w_new, o_new); o_new = o_new[0]
            return sfnn.pack(np.vstack([W1, w_new]), np.append(b1, 0.0), np.append(W2, o_new), b2)
        W1 = rng.normal(size=(n, p)) * np.sqrt(1.0 / p)
        b1 = np.zeros(n)
        W2 = rng.normal(size=n) * np.sqrt(1.0 / n)
        if self.base == "WILCAR" and self.first is not None:
            W1[0] = self.first[:-1]
            b1[0] = self.first[-1]
        if self._aligned:
            W1, W2 = self._align(W1, W2)
        return sfnn.pack(W1, b1, W2, 0.0)

    # ------------------------------------------------------------------ unconstrained training
    def _obj(self, X, y, n):
        """Mean BCE + (lambda0 / N) * ||[W1, W2]||^2 (MAP with a Gaussian prior) and its gradient."""
        lam = self.cfg.l2 / len(y)
        if lam <= 0:
            return lambda v: sfnn.bce_grad(v, X, y, n)
        p = X.shape[1]
        mask = np.zeros(sfnn.n_params(n, p)); mask[:n * p] = 1; mask[n * p + n:n * p + 2 * n] = 1
        def f(v):
            l, g = sfnn.bce_grad(v, X, y, n)
            w = v * mask
            return l + lam * float(w @ w), g + 2 * lam * w
        return f

    def _fit_free(self, theta, X, y, n):
        if self.cfg.optimizer == "gd":
            return self._backprop(theta, X, y, n)
        r = minimize(self._obj(X, y, n), theta, jac=True, method="L-BFGS-B",
                     options={"maxiter": self.cfg.lbfgs_maxiter})
        return r.x

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

    @property
    def cegis(self):
        return self.constrained and self.cfg.constraint_mode == "cegis" and self.base != "ELM" and len(self.idx) > 0

    def _anchor_set(self, An):
        return An if self.extra is None else np.vstack([An, self.extra])

    def _add_anchors(self, P):
        if len(P):
            self.extra = P if self.extra is None else np.vstack([self.extra, P])

    def _constraints(self, An, n, elm=None):
        c, s, idx = self.cfg, self.s, self.idx
        if c.constraint_mode == "global" or (c.constraint_mode == "cegis" and elm is not None):
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
        """
        Return (state, info). state is theta (MLP methods) or dict (ELM).
        With n_starts > 1 the whole step (free fit, projection, counterexample rounds) is repeated from independent
        initialisations of the new weights and the feasible result with the lowest training objective is kept
        (identical for constrained and unconstrained methods; anchors found in any start are kept, they are valid).
        """
        k = max(1, int(self.cfg.n_starts))
        if k == 1 or self.base == "ELM":
            return self._step_once(n, X, y, prev, An)
        obj, best = self._obj(X, y, n), None
        for _ in range(k):
            v, info = self._step_once(n, X, y, prev, An)
            key = (not info["feasible"], obj(v)[0])
            if best is None or key < best[0]:
                best = (key, v, info)
        return best[1], {**best[2], "starts": k}

    def _step_once(self, n, X, y, prev=None, An=None):
        p = X.shape[1]
        if self.base == "ELM":
            return self._step_elm(n, X, y, An)
        if not self.constrained:
            theta = self._fit_free(self._init_theta(n, p, prev), X, y, n)
            return theta, {"feasible": True, "attempts": 1}
        if not len(self.idx):
            theta = self._fit_free(self._init_theta(n, p, prev), X, y, n)
            return theta, {"feasible": True, "attempts": 1}
        obj = self._obj(X, y, n)
        warm = self._fit_free(self._init_theta(n, p, prev), X, y, n) if self.cfg.constrained_init == "free" else None
        v, feasible, attempts = self._solve(n, p, obj, self._anchor_set(An), prev, warm=warm)
        info = {"feasible": feasible, "attempts": attempts}
        if self.cegis and feasible:
            v, info = self._cegis_loop(v, n, p, obj, An, prev, info, X)
        return v, info

    def _solve(self, n, p, obj, An, prev, warm=None):
        """SLSQP with random re-initialisation until the constraints on the anchors hold."""
        cons, cfun = self._constraints(An, n)
        best, best_viol = None, np.inf
        for a in range(self.cfg.max_reinit):
            v0 = warm if (a == 0 and warm is not None) else self._init_theta(n, p, prev)
            v = self._slsqp(obj, v0, cons)
            viol = max(0.0, -cfun(v).min())
            if viol < best_viol:
                best, best_viol = v, viol
            if viol <= self.cfg.feas_tol:
                return v, True, a + 1
        return best, False, self.cfg.max_reinit

    def _cegis_loop(self, v, n, p, obj, An, prev, info, X=None):
        """Counterexample-guided refinement: adversarial search on [0,1]^p, add violators as anchors, re-solve."""
        from . import certify
        rounds = 0
        for rounds in range(1, self.cfg.cegis_rounds + 1):
            cex = certify.adversarial(v, n, p, self.s, self.rng, starts=self.cfg.cegis_starts, X0=X)
            if not len(cex):
                rounds -= 1
                break
            self._add_anchors(cex)
            v2, ok, _ = self._solve(n, p, obj, self._anchor_set(An), prev, warm=v)
            if not ok:
                return v2, {**info, "feasible": False, "cegis_rounds": rounds}
            v = v2
        return v, {**info, "cegis_rounds": rounds, "n_anchors": len(self._anchor_set(An))}

    def certify_and_repair(self, v, n, X, y, An, rounds=20):
        """
        Final sound check (branch-and-bound). In the cegis mode, counterexamples are added as anchors and the model is
        re-solved (repair); if the network is still not certified, the method falls back to the sign-constrained
        problem at the same size, warm-started at the projection of the current weights onto the sign constraints
        (the offending products c_ij < 0 are zeroed). The returned network is therefore always certified.
        """
        from . import certify
        p = X.shape[1]
        status, cex, nodes = certify.certify(v, n, p, self.s, budget=self.cfg.bb_budget)
        r = 0
        if self.cegis:
            obj = self._obj(X, y, n)
            while status == "counterexample" and r < rounds:
                r += 1
                self._add_anchors(cex)
                v, ok, _ = self._solve(n, p, obj, self._anchor_set(An), None, warm=v)
                if ok:
                    v, _ = self._cegis_loop(v, n, p, obj, An, None, {"feasible": True}, X)
                status, cex, nd = certify.certify(v, n, p, self.s, budget=self.cfg.bb_budget)
                nodes += nd
            if status != "certified":
                v = self._sign_fallback(v, n, p, obj)
                status2, _, nd = certify.certify(v, n, p, self.s, budget=self.cfg.bb_budget)
                nodes += nd
                return v, {"cert_status": status2, "cert_nodes": nodes, "cert_repairs": r, "cert_fallback": status}
        return v, {"cert_status": status, "cert_nodes": nodes, "cert_repairs": r, "cert_fallback": ""}

    def _sign_fallback(self, v, n, p, obj):
        """Sign-constrained re-solve warm-started at the projection of v onto {c_ij >= 0}."""
        W1, b1, W2, b2 = sfnn.unpack(v.copy(), n, p)
        bad = (self.s[None, :] * W2[:, None] * W1) < 0          # (n, p); only constrained inputs can be negative
        W1 = np.where(bad, 0.0, W1)
        warm = sfnn.pack(W1, b1, W2, b2)
        cons, cfun = self._global_constraints(n, p)
        best, best_viol = None, np.inf
        for a in range(self.cfg.max_reinit):
            v0 = warm if a == 0 else sfnn.pack(*self._aligned_init(n, p))
            v2 = self._slsqp(obj, v0, cons)
            viol = max(0.0, -cfun(v2).min())
            if viol < best_viol:
                best, best_viol = v2, viol
            if viol <= 1e-10:
                break
        # exact feasibility: zero any residual (numerically) negative products
        W1, b1, W2, b2 = sfnn.unpack(best.copy(), n, p)
        W1 = np.where((self.s[None, :] * W2[:, None] * W1) < 0, 0.0, W1)
        return sfnn.pack(W1, b1, W2, b2)

    def _aligned_init(self, n, p):
        W1 = self.rng.normal(size=(n, p)) * np.sqrt(1.0 / p)
        W2 = self.rng.normal(size=n) * np.sqrt(1.0 / n)
        W1, W2 = self._align(W1, W2)
        return W1, np.zeros(n), W2, 0.0

    def _step_elm(self, n, X, y, An):
        p = X.shape[1]
        for a in range(self.cfg.max_reinit if self.constrained else 1):
            Win = self.rng.uniform(-1, 1, (p, n))
            b = self.rng.uniform(-1, 1, n) if self.cfg.elm_bias else np.zeros(n)
            if self._aligned:                            # monotone ELM: sign-aligned hidden weights, beta >= 0
                Win[self.idx] = np.abs(Win[self.idx]) * self.s[self.idx][:, None]
            H = sfnn.elm_hidden(X, Win, b)
            Hb = np.column_stack([H, np.ones(len(H))])
            if not self.constrained:                     # original ELM: linear output, least squares
                beta = pinv(Hb) @ y if self.cfg.elm_bias else np.append(pinv(H) @ y, 0.0)
                return {"Win": Win, "b": b, "beta": beta, "linear": True}, {"feasible": True, "attempts": 1}
            beta0 = pinv(Hb) @ (4.0 * (y - 0.5))         # logit linearisation of the target
            if self._aligned:
                beta0[:-1] = np.abs(beta0[:-1])
            def obj(bt):
                z = Hb @ bt; o = sfnn.sigmoid(z); oc = np.clip(o, 1e-12, 1 - 1e-12)
                loss = -np.mean(y * np.log(oc) + (1 - y) * np.log(1 - oc))
                return loss, Hb.T @ (o - y) / len(y)
            if not len(self.idx):
                beta = self._slsqp(obj, beta0, {"type": "ineq", "fun": lambda v: np.array([1.0]), "jac": lambda v: np.zeros((1, len(v)))})
                return {"Win": Win, "b": b, "beta": beta, "linear": False}, {"feasible": True, "attempts": 1}
            cons, cfun = self._constraints(An, n, elm=(Win, b))
            beta = self._slsqp(obj, beta0, cons)
            if -cfun(beta).min() <= self.cfg.feas_tol:
                return {"Win": Win, "b": b, "beta": beta, "linear": False}, {"feasible": True, "attempts": a + 1}
        return {"Win": Win, "b": b, "beta": beta, "linear": False}, {"feasible": False, "attempts": self.cfg.max_reinit}

    # ------------------------------------------------------------------ inference
    def predict(self, state, X, n):
        if self.base == "ELM":
            H = sfnn.elm_hidden(X, state["Win"], state["b"])
            z = H @ state["beta"][:-1] + state["beta"][-1]
            return np.clip(z, 0.0, 1.0) if state["linear"] else sfnn.sigmoid(z)
        return sfnn.predict(state, X, n)

"""
Monotone neural-network baselines from the literature (PyTorch), all with guaranteed partial monotonicity:

  MINMAX : Min-Max network (Sill, 1997) - max over K groups of min over J hyperplanes; weights of
           constrained inputs are s_i * exp(v) (sign-fixed by construction).
  CMNN   : Constrained Monotonic Neural Network (Runje & Shankaranarayana, ICML 2023) built from the
           authors' maintained implementation `mononet` (MonoLinear, absolute mode, convex / concave /
           saturated activation split). Constrained inputs go through MonoInput (sign flip) + MonoLinear,
           free inputs through an ordinary Linear layer; both branches are merged by MonoLinear layers,
           so the network is non-decreasing in every flipped constrained input and free in the others.
  LMN    : Lipschitz Monotonic Network (Nolte et al., ICLR 2023) from the authors' package
           `monotonicnetworks`: Lipschitz-constrained GroupSort network + lambda * sum_i s_i x_i.

Training: Adam, mini-batches, BCE on logits, early stopping on validation BCE; hyper-parameters chosen by
validation MCC (same inner split as every other method); final model refitted on train+validation for the
selected number of epochs.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


# ----------------------------------------------------------------------------- architectures
def _minmax(p, s, K, J, torch):
    nn = torch.nn

    class MinMax(nn.Module):
        def __init__(self):
            super().__init__()
            self.K, self.J = K, J
            self.v = nn.Parameter(torch.randn(K * J, p) * 0.5)
            self.b = nn.Parameter(torch.randn(K * J) * 0.1)
            st = torch.tensor(s, dtype=torch.float32)
            self.register_buffer("sign", st)
            self.register_buffer("mono", (st != 0).float())

        def forward(self, x):
            W = self.mono * self.sign * torch.exp(self.v) + (1 - self.mono) * self.v
            z = (x @ W.T + self.b).view(-1, self.K, self.J)
            return z.min(dim=2).values.max(dim=1).values

    return MinMax()


def _cmnn(p, s, h, torch):
    nn = torch.nn
    from mononet.torch import MonoInput, MonoLinear
    from mononet import MonotonicityMask
    s = np.asarray(s)
    mono, free = np.flatnonzero(s), np.flatnonzero(s == 0)

    class CMNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.mono = torch.tensor(mono, dtype=torch.long)
            self.free = torch.tensor(free, dtype=torch.long)
            width = 0
            if len(mono):
                self.flip = MonoInput(MonotonicityMask(s[mono].astype(np.int8)))
                self.m1 = MonoLinear(len(mono), h, activation="elu"); width += h
            if len(free):
                self.f1 = nn.Sequential(nn.Linear(len(free), h), nn.ELU()); width += h
            self.m2 = MonoLinear(width, h, activation="elu")
            self.out = MonoLinear(h, 1)

        def forward(self, x):
            parts = []
            if len(mono):
                parts.append(self.m1(self.flip(x[:, self.mono])))
            if len(free):
                parts.append(self.f1(x[:, self.free]))
            return self.out(self.m2(torch.cat(parts, dim=1))).squeeze(-1)

    return CMNN()


def _lmn(p, s, h, lam, torch):
    import monotonicnetworks as lmn
    nn = torch.nn
    depth = 3
    c = lam ** (1.0 / depth)
    body = nn.Sequential(
        lmn.LipschitzLinear(p, h, kind="one-inf", lipschitz_const=c), lmn.GroupSort(h // 2),
        lmn.LipschitzLinear(h, h, kind="inf", lipschitz_const=c), lmn.GroupSort(h // 2),
        lmn.LipschitzLinear(h, 1, kind="inf", lipschitz_const=c),
    )
    net = lmn.MonotonicWrapper(body, lipschitz_const=lam, monotonic_constraints=list(np.asarray(s, float)))

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__(); self.net = net

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return Wrap()


def _build(name, p, s, hp, torch):
    if name == "MINMAX":
        return _minmax(p, s, hp["K"], hp["K"], torch)
    if name == "CMNN":
        return _cmnn(p, s, hp["h"], torch)
    if name == "LMN":
        return _lmn(p, s, hp["h"], hp["lam"], torch)
    raise ValueError(name)


GRIDS = {
    "MINMAX": [{"K": k} for k in (2, 4, 8)],
    "CMNN": [{"h": h} for h in (8, 16, 32)],
    "LMN": [{"h": h, "lam": lam} for h in (16, 32) for lam in (2.0, 8.0, 32.0)],
}


# ----------------------------------------------------------------------------- training
def _train(name, hp, s, Xtr, ytr, seed, Xva=None, yva=None, epochs=None,
           max_epochs=1000, patience=50, lr=5e-3, batch=256):
    torch = _torch()
    torch.manual_seed(seed); np.random.seed(seed)
    model = _build(name, Xtr.shape[1], s, hp, torch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32) if Xva is not None else None
    yv = torch.tensor(yva, dtype=torch.float32) if yva is not None else None
    n = len(Xt); g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, wait = np.inf, 0, None, 0
    total = epochs if epochs is not None else max_epochs
    for ep in range(1, total + 1):
        model.train()
        for idx in torch.randperm(n, generator=g).split(batch):
            opt.zero_grad(); lossf(model(Xt[idx]), yt[idx]).backward(); opt.step()
        if Xv is None:
            continue
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(Xv), yv))
        if vl < best - 1e-5:
            best, best_ep, wait = vl, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(Z):
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(np.asarray(Z), dtype=torch.float32))).numpy().astype(float)
    return predict, max(best_ep, 1)


def fit(name, signals, Xtr, ytr, Xva, yva, seed):
    """Choose hyper-parameters on (Xva, yva) by MCC; refit on train+val for the selected epoch count."""
    s = np.asarray(signals, float)
    best = (-np.inf, None, None)
    for hp in GRIDS[name]:
        f, ep = _train(name, hp, s, Xtr, ytr, seed, Xva, yva)
        sc = matthews_corrcoef(yva, (f(Xva) >= 0.5).astype(int))
        if sc > best[0]:
            best = (sc, hp, ep)
    _, hp, ep = best
    Xall, yall = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])
    f, _ = _train(name, hp, s, Xall, yall, seed, epochs=ep)
    return f, {**hp, "epochs": ep}


MONO_BASELINES = tuple(GRIDS)

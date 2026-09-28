import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from revision import sfnn

rng = np.random.default_rng(0)

def num_grad(f, th, h=1e-6):
    g = np.zeros_like(th)
    for k in range(th.size):
        e = np.zeros_like(th); e[k] = h
        g[k] = (f(th + e) - f(th - e)) / (2 * h)
    return g

def test_bce_grad():
    n, p = 4, 5
    X = rng.random((30, p)); y = (rng.random(30) > .5).astype(float)
    th = rng.normal(size=sfnn.n_params(n, p))
    l, g = sfnn.bce_grad(th, X, y, n)
    assert abs(l - sfnn.bce(th, X, y, n)) < 1e-12
    assert np.allclose(g, num_grad(lambda t: sfnn.bce(t, X, y, n), th), atol=1e-7)

def test_gains_match_fd_input():
    n, p = 3, 4
    X = rng.random((7, p)); th = rng.normal(size=sfnn.n_params(n, p))
    g = sfnn.gains(th, X, n)
    h = 1e-6
    for i in range(p):
        E = np.zeros(p); E[i] = h
        fd = (sfnn.predict(th, X + E, n) - sfnn.predict(th, X - E, n)) / (2 * h)
        assert np.allclose(g[:, i], fd, atol=1e-8)

def test_gain_param_grad():
    n, p = 3, 4
    X = rng.random((5, p)); th = rng.normal(size=sfnn.n_params(n, p))
    g, G = sfnn.gains_and_grad(th, X, n)
    assert np.allclose(g, sfnn.gains(th, X, n))
    for m in range(5):
        for i in range(p):
            ng = num_grad(lambda t: sfnn.gains(t, X, n)[m, i], th)
            assert np.allclose(G[m, i], ng, atol=1e-7), (m, i, np.abs(G[m, i] - ng).max())

def test_elm_gain_grad():
    n, p = 6, 3
    X = rng.random((5, p)); Win = rng.uniform(-1, 1, (p, n)); b = rng.uniform(-1, 1, n)
    beta = rng.normal(size=n + 1)
    def gfun(bt):
        return sfnn.elm_gains_and_grad(bt, X, Win, b)[0]
    g, G = sfnn.elm_gains_and_grad(beta, X, Win, b)
    for m in range(5):
        for i in range(p):
            ng = num_grad(lambda t: gfun(t)[m, i], beta)
            assert np.allclose(G[m, i], ng, atol=1e-7)
    # gains vs finite differences in x
    h = 1e-6
    for i in range(p):
        E = np.zeros(p); E[i] = h
        f = lambda Z: sfnn.sigmoid(sfnn.elm_hidden(Z, Win, b) @ beta[:-1] + beta[-1])
        assert np.allclose(g[:, i], (f(X + E) - f(X - E)) / (2 * h), atol=1e-6)

def test_global_mode_certifies_monotonicity():
    from revision.methods import Model, Cfg
    from revision.protocol import monotonicity
    X = rng.random((120, 4)); y = ((X[:, 0] - X[:, 1] + 0.3 * rng.normal(size=120)) > 0).astype(float)
    s = [1, -1, 0, 1]
    m = Model("RIXM-C", s, Cfg(constraint_mode="global", max_reinit=5), seed=1)
    th, info = m.step(3, X, y, None, X)
    assert info["feasible"]
    mono = monotonicity(lambda Z: m.predict(th, Z, 3), s, X, n_uniform=2000)
    assert mono["viol_domain"] == 0.0 and mono["viol_test"] == 0.0

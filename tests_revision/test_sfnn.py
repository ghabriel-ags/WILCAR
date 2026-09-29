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

def test_global_elm_c_is_feasible_and_monotone():
    from revision.methods import Model, Cfg
    from revision.protocol import monotonicity
    X = rng.random((150, 4)); y = ((X[:, 0] - X[:, 1] + 0.3 * rng.normal(size=150)) > 0).astype(float)
    s = [1, -1, 0, 1]
    m = Model("ELM-C", s, Cfg(constraint_mode="global", max_reinit=5), seed=2)
    st, info = m.step(20, X, y, None, X)
    assert info["feasible"]
    mono = monotonicity(lambda Z: m.predict(st, Z, 20), s, X, n_uniform=2000)
    assert mono["viol_domain"] == 0.0
    acc = np.mean((m.predict(st, X, 20) >= 0.5) == y)
    assert acc > 0.75

def test_lbfgs_unconstrained_fits():
    from revision.methods import Model, Cfg
    X = rng.random((200, 3)); y = ((X[:, 0] + X[:, 1] > 1.0)).astype(float)
    m = Model("RIXM", [0, 0, 0], Cfg(), seed=3)
    th, _ = m.step(2, X, y)
    assert np.mean((m.predict(th, X, 2) >= 0.5) == y) > 0.9

def test_certify_sound_and_complete_on_small_nets():
    from revision import certify
    n, p = 5, 3
    s = [1, 0, -1]
    for seed in range(20):
        th = np.random.default_rng(seed).normal(size=sfnn.n_params(n, p)) * 2
        st, cex, _ = certify.certify(th, n, p, s, budget=100_000)
        U = np.random.default_rng(99).random((20000, p))
        g = sfnn.gains(th, U, n)[:, [0, 2]] * np.array([1, -1])
        if st == "certified":
            assert (g >= -1e-12).all()
        if st == "counterexample":
            gc = sfnn.gains(th, cex, n)[:, [0, 2]] * np.array([1, -1])
            assert (gc < 0).any(axis=1).all()
        if (g < -1e-6).any():
            assert st != "certified"

def test_cegis_mode_certifies():
    from revision.methods import Model, Cfg
    X = rng.random((200, 4)); y = ((2 * X[:, 0] - X[:, 1] + np.sin(5 * X[:, 2]) + 0.3 * rng.normal(size=200)) > 0.5).astype(float)
    s = [1, -1, 0, 0]
    m = Model("RIXM-C", s, Cfg(constraint_mode="cegis", max_reinit=5), seed=4)
    An = m.anchors(X)
    th, info = m.step(4, X, y, None, An)
    assert info["feasible"]
    th, cert = m.certify_and_repair(th, 4, X, y, An)
    assert cert["cert_status"] in ("certified", "unknown")

def test_certify_zspace_sound():
    from revision import certify
    n, p = 3, 8
    s = [1, -1, 0, 0, 1, 0, 0, 0]
    seen = set()
    for seed in range(40):
        th = np.random.default_rng(seed).normal(size=sfnn.n_params(n, p)) * 2
        W1, b1, idx, C = certify.coefficients(th, n, p, s)
        U = np.random.default_rng(7).random((20000, p))
        for a in range(len(idx)):
            st, x, _ = certify._bb_z(W1, b1, C[a], p)
            seen.add(st)
            h = certify._dsig(U @ W1.T + b1) @ C[a]
            if st == "certified":
                assert (h >= -1e-12).all()
            if st == "counterexample":
                assert certify.h_value(x, W1, b1, C[a]) < 0 and (0 <= x).all() and (x <= 1).all()
    assert {"certified", "counterexample"} <= seen

def test_bb_certifies_mixed_sign_monotone_net():
    """Two identical units with output weights 1 and -0.5: h = 0.5 sigma' > 0 although one c_j < 0."""
    from revision import certify
    p = 4; n = 2
    w = np.array([1.5, -0.7, 0.3, 2.0]); b = 0.2
    W1 = np.vstack([w, w]); b1 = np.array([b, b]); W2 = np.array([1.0, -0.5])
    th = sfnn.pack(W1, b1, W2, 0.1)
    st, cex, _ = certify.certify(th, n, p, [1, -1, 0, 1])
    assert st == "certified"
    st, cex, _ = certify.certify(th, n, p, [-1, 0, 0, 0])          # wrong sign for x_1: must be refuted
    assert st == "counterexample"

def test_cegis_fallback_always_certified():
    """A non-monotone network handed to certify_and_repair with no repair rounds must come back certified (fallback)."""
    from revision.methods import Model, Cfg
    from revision import certify
    X = rng.random((200, 2)); y = ((np.minimum(X[:, 0], X[:, 1]) + 0.1 * rng.normal(size=200)) > 0.45).astype(float)
    s = [1, 1]
    free = Model("RIXM", s, Cfg(), seed=5)
    th, _ = free.step(8, X, y)
    m = Model("RIXM-C", s, Cfg(constraint_mode="cegis", max_reinit=3), seed=5)
    An = m.anchors(X)
    th2, info = m.certify_and_repair(th, 8, X, y, An, rounds=0)
    assert info["cert_status"] == "certified"
    assert certify.certify(th2, 8, 2, s)[0] == "certified"

def test_dcv_one_se_rule():
    from revision.methods import Model, Cfg
    from revision.protocol import select_dcv
    X = rng.random((240, 3)); y = ((X[:, 0] + X[:, 1] + 0.3 * rng.normal(size=240)) > 1).astype(float)
    make = lambda: Model("WILCAR", [1, 1, 0], Cfg(max_neurons=12, patience=4), seed=0)
    n_min, n_1se, curve = select_dcv(make, X, y, 3, 0, one_se=True)
    J = dict((n, v) for n, v in curve if isinstance(v, float))
    assert 1 <= n_1se <= n_min and J[n_1se] >= J[n_min]

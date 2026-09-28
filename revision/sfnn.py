"""
Single-hidden-layer network with analytic gains (input Jacobian) for the
Neural Networks revision.

Conventions
-----------
X : (N, p) inputs, rows are samples;  y : (N,) targets in {0, 1}
theta = [W1 (n*p, row-major), b1 (n), W2 (n), b2 (1)]
Sigmoid hidden layer, sigmoid output p(x) = P(Y=1 | x).

Everything that SLSQP needs is analytic: loss, loss gradient, gains
g_i(x) = dp/dx_i and the gradient of the gains with respect to theta.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
                    np.exp(np.clip(z, -500, 500)) / (1.0 + np.exp(np.clip(z, -500, 500))))


# ----------------------------------------------------------------------------- packing
def n_params(n, p):
    return n * p + 2 * n + 1


def pack(W1, b1, W2, b2):
    return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), np.atleast_1d(b2).ravel()])


def unpack(theta, n, p):
    i = 0
    W1 = theta[i:i + n * p].reshape(n, p); i += n * p
    b1 = theta[i:i + n]; i += n
    W2 = theta[i:i + n]; i += n
    b2 = theta[i]
    return W1, b1, W2, b2


# ----------------------------------------------------------------------------- forward
def forward(theta, X, n):
    p_ = X.shape[1]
    W1, b1, W2, b2 = unpack(theta, n, p_)
    A = sigmoid(X @ W1.T + b1)          # (N, n)
    out = sigmoid(A @ W2 + b2)          # (N,)
    return out, A


def predict(theta, X, n):
    return forward(theta, X, n)[0]


# ----------------------------------------------------------------------------- loss
def bce(theta, X, y, n):
    out, _ = forward(theta, X, n)
    o = np.clip(out, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(o) + (1 - y) * np.log(1 - o)))


def bce_grad(theta, X, y, n):
    """Loss and gradient (sigmoid output + BCE -> dz2 = out - y)."""
    N, p_ = X.shape
    W1, b1, W2, b2 = unpack(theta, n, p_)
    A = sigmoid(X @ W1.T + b1)
    out = sigmoid(A @ W2 + b2)
    o = np.clip(out, EPS, 1 - EPS)
    loss = float(-np.mean(y * np.log(o) + (1 - y) * np.log(1 - o)))
    dz2 = (out - y) / N                           # (N,)
    gW2 = A.T @ dz2
    gb2 = dz2.sum()
    dZ1 = np.outer(dz2, W2) * A * (1 - A)         # (N, n)
    gW1 = dZ1.T @ X
    gb1 = dZ1.sum(axis=0)
    return loss, pack(gW1, gb1, gW2, gb2)


# ----------------------------------------------------------------------------- gains
def gains(theta, X, n):
    """g[m, i] = dp/dx_i at anchor m.  Shape (M, p)."""
    p_ = X.shape[1]
    W1, b1, W2, b2 = unpack(theta, n, p_)
    A = sigmoid(X @ W1.T + b1)                    # (M, n)
    out = sigmoid(A @ W2 + b2)                    # (M,)
    s1 = A * (1 - A)
    s2 = out * (1 - out)
    return s2[:, None] * ((s1 * W2) @ W1)         # (M, p)


def gains_and_grad(theta, X, n):
    """
    Gains g (M, p) and their gradient dg/dtheta (M, p, P) in closed form.

    g_i = s2 * u_i,  u_i = sum_j W2_j s1_j W1_ji
    with s1 = a(1-a), s2 = o(1-o), t1 = s1(1-2a), t2 = s2(1-2o).
    """
    M, p_ = X.shape
    W1, b1, W2, b2 = unpack(theta, n, p_)
    Z1 = X @ W1.T + b1
    A = sigmoid(Z1)
    out = sigmoid(A @ W2 + b2)
    s1 = A * (1 - A)                  # (M, n)
    t1 = s1 * (1 - 2 * A)
    s2 = out * (1 - out)              # (M,)
    t2 = s2 * (1 - 2 * out)
    V = s1 * W2                       # (M, n)   W2_j s1_j
    U = V @ W1                        # (M, p)   u_i
    g = s2[:, None] * U

    P = n_params(n, p_)
    G = np.zeros((M, p_, P))
    # --- d/dW1[k, l]
    #   t2 * V_k x_l * u_i  +  s2 * ( W2_k t1_k x_l W1_ki + W2_k s1_k [i == l] )
    Wt = W2 * t1                                       # (M, n)
    term_a = (t2[:, None, None] * U[:, :, None]) * 1.0  # (M, p, 1)
    dW1 = term_a[:, :, :, None] * (V[:, None, :, None] * X[:, None, None, :])      # (M,p,n,p)
    dW1 += s2[:, None, None, None] * (Wt[:, None, :, None] * X[:, None, None, :]) * W1.T[None, :, :, None]
    # identity term (i == l): add s2 * V_k to dW1[m, i, k, i]
    for i in range(p_):
        dW1[:, i, :, i] += s2[:, None] * V
    G[:, :, :n * p_] = dW1.reshape(M, p_, n * p_)
    # --- d/db1[k]:  t2 V_k u_i + s2 W2_k t1_k W1_ki
    db1 = t2[:, None, None] * U[:, :, None] * V[:, None, :] + s2[:, None, None] * Wt[:, None, :] * W1.T[None, :, :]
    G[:, :, n * p_:n * p_ + n] = db1
    # --- d/dW2[k]:  t2 a_k u_i + s2 s1_k W1_ki
    dW2 = t2[:, None, None] * U[:, :, None] * A[:, None, :] + s2[:, None, None] * s1[:, None, :] * W1.T[None, :, :]
    G[:, :, n * p_ + n:n * p_ + 2 * n] = dW2
    # --- d/db2:  t2 u_i
    G[:, :, -1] = t2[:, None] * U
    return g, G


# ----------------------------------------------------------------------------- ELM (fixed hidden layer)
def elm_hidden(X, Win, bin_):
    return np.maximum(X @ Win + bin_, 0.0)            # ReLU, as in the original ELM


def elm_gains_and_grad(beta, X, Win, bin_, sigmoid_out=True):
    """ELM with trainable output beta (n + 1, last = bias); gains and d gains / d beta."""
    H = elm_hidden(X, Win, bin_)                      # (M, n)
    D = (X @ Win + bin_ > 0).astype(float)            # ReLU derivative
    # dH/dx_i = D_j * Win_ij ;  lin = H beta[:n] + beta[n]
    J = (D[:, None, :] * Win[None, :, :])             # (M, p, n)
    Ulin = J @ beta[:-1]                              # (M, p)  d lin / dx_i
    if not sigmoid_out:
        g = Ulin
        G = np.concatenate([J, np.zeros(J.shape[:2] + (1,))], axis=2)
        return g, G
    out = sigmoid(H @ beta[:-1] + beta[-1])
    s2 = out * (1 - out)
    t2 = s2 * (1 - 2 * out)
    g = s2[:, None] * Ulin
    Hb = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)          # (M, n+1)
    G = t2[:, None, None] * Ulin[:, :, None] * Hb[:, None, :]
    G[:, :, :-1] += s2[:, None, None] * J
    return g, G

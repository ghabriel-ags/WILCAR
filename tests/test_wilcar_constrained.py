"""WILCAR-C (M2): constructive steps after a network size with no feasible (100% SCR) configuration.

When no reinitialization reaches 100% SCR at size n, train() skips n and keeps the last feasible model
(k < n neurons) for weight reuse. The next size must then be built with n + 1 neurons; before the fix
_expand_network added a single neuron (k + 1) and _vector_to_params failed with
"cannot reshape array of size ... into shape (1, n + 1)" (Algerian, fold 1, March 2026).
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from base import TrainingConfig  # noqa: E402
from wilcar.constrained import WILCARConstrainedNumpy  # noqa: E402


def make_model(max_neurons=4):
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(80, 3))
    y = (x[:, 0] + x[:, 1] + 0.2 * rng.normal(size=80) > 1).astype(float)
    config = TrainingConfig(max_neurons=max_neurons, patience_constructive=10, seed=0, verbose=0,
                            task="classification")
    return WILCARConstrainedNumpy(x[:60], y[:60], x[60:], y[60:], [+1, +1, 0], config, "synthetic")


def expand_network_march_2026(model, previous_param, n_neurons):
    """_expand_network as used for the dissertation runs (commit 0e54bfe)."""
    new_param = copy.deepcopy(previous_param)
    new_w1 = np.random.randn(1, model.n_inputs) * np.sqrt(1. / model.n_inputs)
    new_param['W1'] = np.vstack([new_param['W1'], new_w1])
    new_param['b1'] = np.vstack([new_param['b1'], np.zeros((1, 1))])
    new_w2 = np.random.randn(1, 1) * np.sqrt(1. / n_neurons)
    new_param['W2'] = np.hstack([new_param['W2'], new_w2])
    return new_param


@pytest.mark.parametrize("n", [2, 3, 7])
def test_expand_network_by_one_neuron_is_unchanged(n):
    model = make_model()
    np.random.seed(123)
    previous = model._initialize_parameters(n - 1)

    np.random.seed(7)
    expected = expand_network_march_2026(model, previous, n)
    after_expected = np.random.randn()
    np.random.seed(7)
    got = model._expand_network(previous, n)
    after_got = np.random.randn()

    for key in ("W1", "b1", "W2", "b2"):
        np.testing.assert_array_equal(got[key], expected[key])
    assert after_got == after_expected  # same number of draws from the global RNG


def test_expand_network_fills_skipped_sizes():
    model = make_model()
    previous = model._initialize_parameters(1)
    param = model._expand_network(previous, 3)

    assert param["W1"].shape == (3, model.n_inputs)
    assert param["b1"].shape == (3, 1)
    assert param["W2"].shape == (1, 3)
    np.testing.assert_array_equal(param["W1"][:1], previous["W1"])
    np.testing.assert_array_equal(param["W2"][:, :1], previous["W2"])
    back = model._vector_to_params(model._params_to_vector(param), 3)
    for key in ("W1", "b1", "W2", "b2"):
        np.testing.assert_array_equal(back[key], param[key])


def test_train_continues_after_infeasible_size(monkeypatch):
    model = make_model(max_neurons=4)
    check = model._check_conformity_on_data
    # Size 2 never reaches 100% SCR, so it is skipped and size 3 follows the 1-neuron model
    monkeypatch.setattr(model, "_check_conformity_on_data",
                        lambda x: 0.5 if model.n_neurons == 2 else check(x))
    monkeypatch.setattr(model, "MAX_REINIT_ATTEMPTS", 2)

    results = model.train()

    assert len(results.mcc_test) == 3  # sizes 1, 3 and 4
    assert results.best_neurons in (1, 3, 4)
    assert model.param["W1"].shape[0] == results.best_neurons

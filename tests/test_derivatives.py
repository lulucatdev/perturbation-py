"""Tests for automatic differentiation / finite-difference Jacobian computation.

The ``compute_jacobians`` function numerically differentiates the model's
transition and arbitrage equations around the deterministic steady state.
For the linear scalar test model the Jacobians are known analytically:

    g_s = rho,      g_x = 0,       g_e = sigma
    f_s = -kappa,   f_x = 1,       f_S = 0,        f_X = -beta

This test verifies that finite-difference approximations match these values
to machine-level accuracy (given a suitably small step size).
"""

import numpy as np

from perturbation_py.derivatives import compute_jacobians
from helpers import make_scalar_model


def test_finite_difference_jacobians_match_known_linear_model():
    """Verify that all eight Jacobian blocks match the analytic values for the linear model.

    The transition Jacobians (g_s, g_x, g_e) and the arbitrage Jacobians
    (f_s, f_x, f_S, f_X) are each checked to within 1e-7.
    """
    model = make_scalar_model()
    jac = compute_jacobians(model, epsilon=1e-7)

    assert np.allclose(jac.g_s, [[0.9]], atol=1e-7)
    assert np.allclose(jac.g_x, [[0.0]], atol=1e-7)
    assert np.allclose(jac.g_e, [[0.01]], atol=1e-7)

    assert np.allclose(jac.f_s, [[-0.1]], atol=1e-7)
    assert np.allclose(jac.f_x, [[1.0]], atol=1e-7)
    assert np.allclose(jac.f_S, [[0.0]], atol=1e-7)
    assert np.allclose(jac.f_X, [[-0.95]], atol=1e-7)

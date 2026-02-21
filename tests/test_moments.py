"""Tests for unconditional moments computation.

The ``compute_unconditional_moments`` function solves the discrete Lyapunov
equation for state (and control) covariances and derives means, standard
deviations, and autocorrelations from a given ``Policy`` object.

For the scalar AR(1) test model  s' = rho * s + sigma * eps:

- The unconditional variance of s is  sigma^2 / (1 - rho^2).
- The lag-k autocorrelation of s is  rho^k.
- At first order the unconditional mean equals the deterministic steady state.
- At second order the mean includes the risk-adjustment  0.5 * ghs2.

Each test below verifies one of these analytic results.
"""

import numpy as np
from numpy.testing import assert_allclose

from helpers import make_scalar_model
from perturbation_py.moments import compute_unconditional_moments
from perturbation_py.policy import Policy
from perturbation_py.solver import solve_first_order
from perturbation_py.solver_second_order import solve_second_order


def test_scalar_ar1_variance_matches_analytic():
    """Verify state variance equals sigma^2 / (1 - rho^2) for the scalar AR(1) model."""
    rho = 0.9
    sigma = 0.01
    model = make_scalar_model(rho=rho, sigma=sigma)
    fo = solve_first_order(model)
    policy = Policy.from_first_order(fo)

    moments = compute_unconditional_moments(policy)

    # State variance: sigma^2 / (1 - rho^2) since transition is s' = rho*s + sigma*e
    expected_var_s = sigma**2 / (1.0 - rho**2)
    assert_allclose(moments.variance_states[0, 0], expected_var_s, rtol=1e-10)

    # State std
    assert_allclose(moments.std_states[0], np.sqrt(expected_var_s), rtol=1e-10)


def test_first_order_mean_at_steady_state():
    """At order 1 the unconditional mean of controls equals the deterministic steady state.

    Because the first-order solution is linear and shocks are mean-zero,
    E[x] = x_ss exactly, with no risk-adjustment term.
    """
    model = make_scalar_model()
    fo = solve_first_order(model)
    policy = Policy.from_first_order(fo)

    moments = compute_unconditional_moments(policy)
    assert_allclose(moments.mean_controls, policy.steady_state_controls, atol=1e-14)
    assert_allclose(moments.mean_states, 0.0, atol=1e-14)


def test_second_order_mean_includes_ghs2_correction():
    """At order 2 the unconditional mean of controls includes the risk correction 0.5 * ghs2.

    The second-order perturbation adds a constant shift ghs2 that captures
    the effect of uncertainty on the mean level of decision variables (the
    "precautionary" or "Jensen's inequality" term).  The unconditional mean
    is therefore  x_ss + 0.5 * ghs2.
    """
    model = make_scalar_model()
    so = solve_second_order(model, method="sylvester")
    policy = Policy.from_second_order(so)

    moments = compute_unconditional_moments(policy)
    expected_mean = policy.steady_state_controls + 0.5 * so.ghs2
    assert_allclose(moments.mean_controls, expected_mean, atol=1e-14)


def test_autocorrelation_shape_and_decay():
    """Autocorrelations have the expected shape and match rho^k for the AR(1) state.

    For the scalar AR(1) model the theoretical lag-k autocorrelation of the
    state is exactly rho^k.  This test also checks that the returned array
    has the correct shape (max_lag, n_states + n_controls) and that
    autocorrelations decay monotonically for a stable system.
    """
    model = make_scalar_model(rho=0.9)
    fo = solve_first_order(model)
    policy = Policy.from_first_order(fo)

    max_lag = 5
    moments = compute_unconditional_moments(policy, max_lag=max_lag)

    assert moments.autocorrelations is not None
    assert moments.autocorrelations.shape == (max_lag, 2)  # 1 state + 1 control

    # State autocorrelation at lag k should be rho^k
    for k in range(1, max_lag + 1):
        expected = 0.9**k
        assert_allclose(moments.autocorrelations[k - 1, 0], expected, rtol=1e-10)

    # Autocorrelations should decay
    assert moments.autocorrelations[0, 0] > moments.autocorrelations[-1, 0]

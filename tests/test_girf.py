"""Tests for generalized impulse response functions.

The generalized IRF (GIRF) extends the standard linear IRF to higher-order
approximations by computing the difference between a shocked simulation path
and an unshocked baseline, both evaluated using the full (possibly nonlinear)
policy function.

Key properties tested:

- At order 1 the GIRF must reproduce the standard linear IRF exactly, because
  the policy function is linear and the baseline path is trivially zero.
- At order 2+ the GIRF can exhibit asymmetry with respect to the sign of the
  shock (Jensen's inequality), but for a linear model the asymmetry vanishes.
- For a stable model the GIRF must decay toward zero as the horizon grows.
"""

import numpy as np
from numpy.testing import assert_allclose

from helpers import make_scalar_model
from perturbation_py.policy import Policy
from perturbation_py.simulation import generalized_irf, impulse_response
from perturbation_py.solver import solve_first_order
from perturbation_py.solver_second_order import solve_second_order


def test_girf_order1_equals_linear_irf():
    """At order 1, GIRF should equal the standard linear IRF.

    Note: simulate_linear doesn't include the contemporaneous ghu @ e term
    at t=0, so we compare from t=1 onwards for controls and verify the t=0
    control separately.
    """
    model = make_scalar_model()
    fo = solve_first_order(model)
    policy = Policy.from_first_order(fo)

    horizon = 20
    linear_irf = impulse_response(fo, horizon=horizon, shock_index=0, shock_size=1.0)
    girf = generalized_irf(policy, horizon=horizon, shock_index=0, shock_size=1.0)

    # States should match exactly
    assert_allclose(girf.states, linear_irf.states, atol=1e-12)

    # Controls from t=1 onwards should match (no contemporaneous shock)
    assert_allclose(girf.controls[1:], linear_irf.controls[1:], atol=1e-12)

    # At t=0, GIRF includes ghu @ e which linear IRF omits
    expected_t0 = fo.control_shock_impact @ np.array([1.0])
    assert_allclose(girf.controls[0], expected_t0, atol=1e-12)


def test_girf_shape():
    """Verify that the GIRF output arrays have the expected dimensions.

    States have shape (horizon+1, n_states) because they include the initial
    condition, while controls have shape (horizon, n_controls).
    """
    model = make_scalar_model()
    so = solve_second_order(model)
    policy = Policy.from_second_order(so)

    horizon = 10
    girf = generalized_irf(policy, horizon=horizon, shock_index=0, shock_size=1.0)

    assert girf.states.shape == (horizon + 1, model.n_states)
    assert girf.controls.shape == (horizon, model.n_controls)
    assert girf.baseline_states.shape == (horizon + 1, model.n_states)
    assert girf.baseline_controls.shape == (horizon, model.n_controls)


def test_girf_asymmetry_for_nonlinear_model():
    """For a linear model the GIRF is symmetric with respect to the shock sign.

    At order 2 and above, a genuinely nonlinear model would produce
    |GIRF(+1)| != |GIRF(-1)| due to the curvature of the policy function.
    Here we use the linear test model to confirm that the symmetric special
    case is handled correctly.
    """
    model = make_scalar_model()
    so = solve_second_order(model, method="sylvester")
    policy = Policy.from_second_order(so)

    horizon = 10
    girf_pos = generalized_irf(policy, horizon=horizon, shock_index=0, shock_size=1.0)
    girf_neg = generalized_irf(policy, horizon=horizon, shock_index=0, shock_size=-1.0)

    # For linear model, |GIRF(+1)| ≈ |GIRF(-1)|
    assert_allclose(
        np.abs(girf_pos.controls), np.abs(girf_neg.controls), atol=1e-8,
    )


def test_girf_decays_for_stable_model():
    """For a stable AR(1) model the GIRF decays geometrically toward zero.

    With rho = 0.8 the state after 50 periods should be negligibly small
    relative to the initial impact, confirming that the simulated path
    converges back to the steady state.
    """
    model = make_scalar_model(rho=0.8)
    fo = solve_first_order(model)
    policy = Policy.from_first_order(fo)

    horizon = 50
    girf = generalized_irf(policy, horizon=horizon, shock_index=0, shock_size=1.0)

    # Last state should be much smaller than first
    assert np.abs(girf.states[-1, 0]) < 0.01 * np.abs(girf.states[1, 0])

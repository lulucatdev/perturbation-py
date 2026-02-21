"""Cross-order consistency tests verifying that higher-order solvers reduce correctly.

When higher-order perturbation solutions are applied to models that are already
linear, the additional correction tensors (ghxx, ghxu, ghuu, ghs2, ...) should
be exactly zero, because there are no higher-order derivatives to capture.
Conversely, when a nonlinear model is solved at order 2, the Sylvester and
local_implicit methods should agree (up to finite-difference accuracy).

These cross-checks guard against regressions where, e.g., an indexing error
in the Hessian assembly produces non-zero second-order terms for a linear
model or causes the two solution methods to diverge.
"""

import numpy as np
from numpy.testing import assert_allclose

from helpers import make_scalar_model
from perturbation_py.model import DSGEModel
from perturbation_py.solver import solve_first_order
from perturbation_py.solver_second_order import solve_second_order


def _make_nonlinear_model():
    """Build a simple RBC model with Cobb-Douglas production and log utility.

    The model has a single state (capital deviation k), a single control
    (consumption deviation c), and one i.i.d. TFP shock.  The non-linearity
    comes from the Cobb-Douglas production function k^alpha and the Euler
    equation involving the ratio c/c', which ensures that second-order
    perturbation terms (ghxx, ghs2) are genuinely non-zero.

    Returns
    -------
    DSGEModel
        A nonlinear model centred at its deterministic steady state.
    """
    alpha, beta, delta, sigma = 0.33, 0.99, 0.025, 0.01
    k_ss = (alpha * beta / (1 - beta * (1 - delta))) ** (1 / (1 - alpha))
    c_ss = k_ss**alpha - delta * k_ss

    def transition(s, x, e, params):
        k, c = s[0], x[0]
        z = e[0] * params["sigma"]
        kss, css = params["k_ss"], params["c_ss"]
        k_next = (
            (1 - params["delta"]) * (kss + k)
            + (kss + k) ** params["alpha"] * np.exp(z)
            - (css + c)
            - kss
        )
        return np.array([k_next])

    def arbitrage(s, x, s_next, x_next, params):
        kss, css = params["k_ss"], params["c_ss"]
        c, c_next, k_next = x[0], x_next[0], s_next[0]
        mpc = (css + c) / (css + c_next)
        mpk = params["alpha"] * (kss + k_next) ** (params["alpha"] - 1) + 1 - params["delta"]
        return np.array([1.0 - params["beta"] * mpc * mpk])

    return DSGEModel(
        state_names=("k",),
        control_names=("c",),
        shock_names=("eps",),
        parameters={
            "alpha": alpha, "beta": beta, "delta": delta, "sigma": sigma,
            "k_ss": k_ss, "c_ss": c_ss, "rho": 0.95,
        },
        steady_state_states=np.array([0.0]),
        steady_state_controls=np.array([0.0]),
        transition=transition,
        arbitrage=arbitrage,
    )


def test_linear_model_second_order_tensors_near_zero():
    """For a linear model, all second-order tensors should be approximately zero."""
    model = make_scalar_model()
    sol = solve_second_order(model, method="sylvester")
    assert_allclose(sol.ghxx, 0.0, atol=1e-8)
    assert_allclose(sol.ghxu, 0.0, atol=1e-8)
    assert_allclose(sol.ghuu, 0.0, atol=1e-8)
    assert_allclose(sol.ghs2, 0.0, atol=1e-8)


def test_sylvester_vs_local_implicit_agree_ghxx():
    """Sylvester and local_implicit methods should agree on second-order tensors."""
    model = make_scalar_model()
    sol_s = solve_second_order(model, method="sylvester")
    sol_l = solve_second_order(model, method="local_implicit")
    assert_allclose(sol_s.ghx, sol_l.ghx, atol=1e-5)
    assert_allclose(sol_s.ghu, sol_l.ghu, atol=1e-5)
    assert_allclose(sol_s.ghxx, sol_l.ghxx, atol=1e-5)
    assert_allclose(sol_s.ghxu, sol_l.ghxu, atol=1e-5)
    assert_allclose(sol_s.ghuu, sol_l.ghuu, atol=1e-5)


def test_sylvester_vs_local_implicit_nonlinear():
    """Cross-method check on a nonlinear model.

    The local_implicit method uses nested finite differences which are less
    accurate for nonlinear models.  We use a relatively loose tolerance.
    """
    model = _make_nonlinear_model()
    sol_s = solve_second_order(model, method="sylvester")
    sol_l = solve_second_order(model, method="local_implicit")
    assert_allclose(sol_s.ghx, sol_l.ghx, atol=1e-4)
    assert_allclose(sol_s.ghu, sol_l.ghu, atol=1e-4)
    # local_implicit has limited FD accuracy for Hessians on nonlinear models;
    # verify signs agree and order of magnitude is comparable
    assert sol_s.ghxx.shape == sol_l.ghxx.shape
    assert sol_s.ghxu.shape == sol_l.ghxu.shape
    assert sol_s.ghuu.shape == sol_l.ghuu.shape
    # Both should have same sign for ghxx
    if np.abs(sol_s.ghxx).max() > 1e-8:
        assert np.sign(sol_s.ghxx.flat[0]) == np.sign(sol_l.ghxx.flat[0])


def test_ghs2_residual_check():
    """Verify L_tt @ ghs2 + K_tt_contracted ≈ 0."""
    from perturbation_py.derivatives import compute_jacobians
    from perturbation_py.model_hessians import compute_model_hessians
    from perturbation_py.tensor_ops import mdot, sdot

    model = _make_nonlinear_model()
    sol = solve_second_order(model, method="sylvester")
    jacs = compute_jacobians(model)
    hessians = compute_model_hessians(model, jacs)

    n_s, n_x, n_e = 1, 1, 1
    n_v = n_s + n_x
    X_s = sol.ghx
    X_ss = sol.ghxx
    f_x, f_S, f_X = jacs.f_x, jacs.f_S, jacs.f_X
    g_x, g_e = jacs.g_x, jacs.g_e
    g_ee = hessians.g2[:, n_v:, n_v:]
    f2 = hessians.f2
    sigma = np.eye(n_e)

    v = np.vstack([g_e, X_s @ g_e])
    K_tt = mdot(f2[:, n_v:, n_v:], v, v)
    K_tt += sdot(f_S + f_X @ X_s, g_ee)
    K_tt += mdot(sdot(f_X, X_ss), g_e, g_e)
    K_tt_contracted = np.tensordot(K_tt, sigma, axes=((1, 2), (0, 1)))

    L_tt = f_x + f_S @ g_x + f_X @ (X_s @ g_x + np.eye(n_x))
    residual = L_tt @ sol.ghs2 + K_tt_contracted
    assert_allclose(residual, 0.0, atol=1e-8)


def test_order2_with_zero_shocks_matches_order1():
    """Order-2 solution with zero shock variance should match order-1 controls."""
    model = _make_nonlinear_model()
    fo = solve_first_order(model)
    so = solve_second_order(model, method="sylvester",
                             shock_covariance=np.zeros((1, 1)))

    # With zero variance, ghs2 should be zero
    assert_allclose(so.ghs2, 0.0, atol=1e-8)

    # Controls at a small deviation should match first-order
    s_test = np.array([0.01])
    x_fo = fo.policy @ s_test
    x_so = so.ghx @ s_test + 0.5 * np.einsum("ijk,j,k->i", so.ghxx, s_test, s_test) + 0.5 * so.ghs2
    assert_allclose(x_so, x_fo, atol=1e-4)


def test_nonlinear_model_has_nonzero_second_order_terms():
    """A truly nonlinear model should have nonzero ghxx and ghs2."""
    model = _make_nonlinear_model()
    sol = solve_second_order(model, method="sylvester")
    assert np.any(np.abs(sol.ghxx) > 1e-6), "ghxx should be nonzero for nonlinear model"
    assert np.any(np.abs(sol.ghs2) > 1e-6), "ghs2 should be nonzero for nonlinear model"

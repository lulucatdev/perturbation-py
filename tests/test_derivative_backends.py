"""Tests for alternative derivative backends (finite difference vs. complex step).

The ``compute_jacobians`` function supports multiple differentiation backends.
The default ``finite_difference`` backend uses real-valued central differences,
while the ``complex_step`` backend exploits the identity

    f'(x) = Im[f(x + i*h)] / h

which is exact to machine precision for analytic (holomorphic) functions and
avoids the subtractive cancellation inherent in finite differences.

This test verifies that both backends produce identical Jacobians on a model
whose equations are safe for complex arithmetic (no abs(), no branching on
values, etc.).
"""

import numpy as np

from perturbation_py.derivatives import compute_jacobians
from perturbation_py.model import DSGEModel


def make_complex_safe_scalar_model() -> DSGEModel:
    """Build a scalar AR(1) model whose equations are safe for complex-step differentiation.

    This is functionally identical to ``make_scalar_model`` in helpers.py but
    is defined locally to emphasise that the model's equations must not contain
    operations that break complex arithmetic (e.g., ``np.abs``, ``np.real``).

    Returns
    -------
    DSGEModel
        A linear scalar model suitable for both finite-difference and
        complex-step differentiation.
    """
    def transition(
        s: np.ndarray, x: np.ndarray, e: np.ndarray, params: dict[str, float]
    ) -> np.ndarray:
        return np.array([params["rho"] * s[0] + params["sigma"] * e[0]])

    def arbitrage(
        s: np.ndarray,
        x: np.ndarray,
        s_next: np.ndarray,
        x_next: np.ndarray,
        params: dict[str, float],
    ) -> np.ndarray:
        return np.array([x[0] - params["beta"] * x_next[0] - params["kappa"] * s[0]])

    return DSGEModel(
        state_names=("k",),
        control_names=("c",),
        shock_names=("eps",),
        parameters={"rho": 0.9, "beta": 0.95, "kappa": 0.1, "sigma": 0.01},
        steady_state_states=np.array([0.0]),
        steady_state_controls=np.array([0.0]),
        transition=transition,
        arbitrage=arbitrage,
    )


def test_complex_step_matches_finite_difference_on_linear_fixture():
    """All Jacobian blocks from the complex-step backend should match finite differences.

    Because the model is linear, both backends should agree to essentially
    machine precision (atol = 1e-10).
    """
    model = make_complex_safe_scalar_model()

    jac_fd = compute_jacobians(model, backend="finite_difference")
    jac_cs = compute_jacobians(model, backend="complex_step")

    assert np.allclose(jac_fd.g_s, jac_cs.g_s, atol=1e-10)
    assert np.allclose(jac_fd.g_e, jac_cs.g_e, atol=1e-10)
    assert np.allclose(jac_fd.f_s, jac_cs.f_s, atol=1e-10)
    assert np.allclose(jac_fd.f_x, jac_cs.f_x, atol=1e-10)
    assert np.allclose(jac_fd.f_X, jac_cs.f_X, atol=1e-10)

"""Tests for the second-order perturbation solver.

The second-order solver augments the first-order linear policy with quadratic
correction tensors (ghxx, ghxu, ghuu) and a constant risk-adjustment ghs2.
For the linear test model every second-order tensor should be zero, while the
first-order component ghx should still equal the analytic value.

These tests verify output shapes and numerical agreement with the known
solution.
"""

import numpy as np

from helpers import make_scalar_model
from perturbation_py.solver_second_order import solve_second_order


def test_second_order_solver_returns_expected_tensor_shapes():
    """Verify tensor shapes and that ghx matches the analytic first-order policy.

    Expected shapes for n_s=1, n_x=1, n_e=1:
      ghx:  (1, 1)
      ghu:  (1, 1)
      ghxx: (1, 1, 1)
      ghxu: (1, 1, 1)
      ghuu: (1, 1, 1)
      ghs2: (1,)
    """
    model = make_scalar_model()
    sol = solve_second_order(model)

    assert sol.ghx.shape == (model.n_controls, model.n_states)
    assert sol.ghu.shape == (model.n_controls, model.n_shocks)
    assert sol.ghxx.shape == (model.n_controls, model.n_states, model.n_states)
    assert sol.ghxu.shape == (model.n_controls, model.n_states, model.n_shocks)
    assert sol.ghuu.shape == (model.n_controls, model.n_shocks, model.n_shocks)
    assert sol.ghs2.shape == (model.n_controls,)

    expected = 0.1 / (1.0 - 0.95 * 0.9)
    assert np.allclose(sol.ghx, [[expected]], atol=1e-6)

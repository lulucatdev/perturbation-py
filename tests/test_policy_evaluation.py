"""Tests for the Policy evaluation interface.

The ``Policy`` object wraps the perturbation solution tensors and provides a
``controls(state, shock)`` method that evaluates the full polynomial decision
rule.  When constructed from a first-order solution (``Policy.from_first_order``),
only the linear term ghx is non-zero, so the policy reduces to  x = P * s.

These tests verify that the ``Policy`` evaluator returns the same result as
a direct matrix-vector product with the first-order policy matrix.
"""

import numpy as np

from helpers import make_scalar_model
from perturbation_py.policy import Policy
from perturbation_py.solver import solve_first_order


def test_policy_evaluator_matches_first_order_when_higher_terms_zero():
    """Policy.controls should equal P @ s when higher-order tensors are absent.

    With zero shocks and a purely first-order policy, the evaluator should
    reproduce the simple linear mapping  x = policy @ state.
    """
    sol = solve_first_order(make_scalar_model())
    policy = Policy.from_first_order(sol)
    state = np.array([0.2], dtype=float)
    shock = np.array([0.0], dtype=float)

    out = policy.controls(state=state, shock=shock)
    assert np.allclose(out, sol.policy @ state)

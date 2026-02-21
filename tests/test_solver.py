"""Tests for the first-order perturbation solver.

The first-order solver linearises the model around its deterministic steady
state and solves the resulting matrix quadratic (or generalised eigenvalue)
problem to obtain the policy function  x = P * s  and the state transition
s' = T * s + R * eps.

For the scalar AR(1) test model the closed-form policy coefficient is

    P = kappa / (1 - beta * rho)

which provides an exact check of the numerical solver.  The test also
verifies that the Blanchard-Kahn conditions are reported as satisfied.
"""

import numpy as np

from perturbation_py.solver import solve_first_order
from helpers import make_scalar_model


def test_first_order_solver_matches_analytic_policy():
    """Verify the first-order policy coefficient matches kappa / (1 - beta * rho).

    Also checks that the state transition matrix equals [[rho]] and that the
    Blanchard-Kahn rank conditions are satisfied.
    """
    rho = 0.9
    beta = 0.95
    kappa = 0.1
    model = make_scalar_model(rho=rho, beta=beta, kappa=kappa, sigma=0.01)

    solution = solve_first_order(model)

    expected_policy = kappa / (1.0 - beta * rho)
    assert np.allclose(solution.policy, [[expected_policy]], atol=1e-8)
    assert np.allclose(solution.transition, [[rho]], atol=1e-8)
    assert solution.blanchard_kahn_satisfied

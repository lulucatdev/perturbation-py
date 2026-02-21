"""Tests for linear simulation and impulse response functions.

The ``simulate_linear`` function propagates the first-order law of motion
s' = T s + R eps forward in time given a sequence of shocks, while
``impulse_response`` is a convenience wrapper that applies a single unit
shock at t=0 and records the resulting trajectory.

For the scalar AR(1) model with sigma = 0.01, a unit shock produces an
initial state impact of 0.01 which then decays geometrically at rate rho.
Both the simulation and the IRF should display this decay.
"""

import numpy as np

from perturbation_py.simulation import impulse_response, simulate_linear
from perturbation_py.solver import solve_first_order
from helpers import make_scalar_model


def test_simulation_and_irf_decay_for_stable_system():
    """A unit shock should produce an initial state of sigma=0.01, decaying at rate rho.

    The test verifies this for both ``simulate_linear`` (given an explicit shock
    sequence) and ``impulse_response`` (given a shock index and size).
    """
    model = make_scalar_model(rho=0.9, beta=0.95, kappa=0.1, sigma=0.01)
    solution = solve_first_order(model)

    shocks = np.zeros((20, 1), dtype=float)
    shocks[0, 0] = 1.0
    sim = simulate_linear(solution, initial_state=np.array([0.0]), shocks=shocks)

    assert np.isclose(sim.states[1, 0], 0.01, atol=1e-10)
    assert abs(sim.states[-1, 0]) < abs(sim.states[2, 0])

    irf = impulse_response(solution, horizon=20, shock_index=0, shock_size=1.0)
    assert np.isclose(irf.states[1, 0], 0.01, atol=1e-10)
    assert abs(irf.states[-1, 0]) < abs(irf.states[2, 0])

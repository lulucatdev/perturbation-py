"""Tests for the numerical steady-state solver.

When a model's deterministic steady state cannot be obtained analytically,
``solve_steady_state`` uses Newton's method to find values (s_ss, x_ss)
such that the static residual -- the model equations evaluated at
s = s' = s_ss, x = x' = x_ss, e = 0 -- is zero.

The test model here has a nonlinear transition that includes a quadratic
control term (gamma * x^2), which shifts the steady state away from the
origin and requires a numerical solve.
"""

import numpy as np

from perturbation_py.model import DSGEModel
from perturbation_py.steady_state import solve_steady_state


def make_nonlinear_fixture_model() -> DSGEModel:
    """Build a scalar model with a quadratic nonlinearity in the transition.

    Transition:  k' = 0.5 + rho * k + gamma * c^2 + sigma * eps
    Arbitrage:   c  = a + b * k'

    The quadratic term gamma * c^2 prevents the steady state from being at the
    origin, making this fixture suitable for testing the Newton solver.

    Returns
    -------
    DSGEModel
        A nonlinear model whose steady state must be found numerically.
    """
    def transition(
        s: np.ndarray, x: np.ndarray, e: np.ndarray, params: dict[str, float]
    ) -> np.ndarray:
        return np.array(
            [
                0.5
                + params["rho"] * s[0]
                + params["gamma"] * x[0] * x[0]
                + params["sigma"] * e[0]
            ],
            dtype=float,
        )

    def arbitrage(
        s: np.ndarray,
        x: np.ndarray,
        s_next: np.ndarray,
        x_next: np.ndarray,
        params: dict[str, float],
    ) -> np.ndarray:
        return np.array(
            [x[0] - (params["a"] + params["b"] * s_next[0])],
            dtype=float,
        )

    return DSGEModel(
        state_names=("k",),
        control_names=("c",),
        shock_names=("eps",),
        parameters={"rho": 0.2, "gamma": 0.3, "sigma": 0.0, "a": 0.1, "b": 0.4},
        steady_state_states=np.array([0.0]),
        steady_state_controls=np.array([0.0]),
        transition=transition,
        arbitrage=arbitrage,
    )


def test_newton_solver_finds_consistent_steady_state():
    """The Newton solver should find (s, x) such that the static residual is zero.

    Starting from the initial guess (0.3, 0.3), the solver must converge to a
    fixed point where the transition and arbitrage equations are simultaneously
    satisfied with residual norm below 1e-10.
    """
    model = make_nonlinear_fixture_model()
    ss = solve_steady_state(model, guess_states=[0.3], guess_controls=[0.3])

    residual = model.static_residual(ss.states, ss.controls)
    assert np.linalg.norm(residual, ord=np.inf) < 1e-10
    assert ss.success

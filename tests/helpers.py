"""Shared test helpers for the perturbation_py test suite.

Provides factory functions that build small, analytically tractable DSGE models
used across many test modules. Keeping them in a single helpers module avoids
duplicating model definitions and makes it easy to adjust the canonical test
fixtures in one place.
"""

import numpy as np

from perturbation_py.model import DSGEModel


def make_scalar_model(
    rho: float = 0.9, beta: float = 0.95, kappa: float = 0.1, sigma: float = 0.01
) -> DSGEModel:
    """Build a scalar AR(1) model with a linear Euler equation.

    The model has one state (k), one control (c), and one shock (eps):

        Transition:  k' = rho * k + sigma * eps
        Arbitrage:   c  = beta * c' + kappa * k

    Because both equations are linear, the model admits an exact closed-form
    first-order policy:

        c = [kappa / (1 - beta * rho)] * k

    and all higher-order perturbation tensors (ghxx, ghuuu, etc.) are exactly
    zero. This makes the model ideal for testing that:
      - the first-order solver recovers the analytic policy coefficient,
      - higher-order solvers correctly produce zero higher-order terms, and
      - simulation / IRF / moments routines behave as expected on a simple
        known system.

    Parameters
    ----------
    rho : float
        Persistence of the AR(1) state process.
    beta : float
        Discount factor in the Euler equation.
    kappa : float
        Slope of the Euler equation with respect to the state.
    sigma : float
        Standard deviation of the shock entering the transition.

    Returns
    -------
    DSGEModel
        A fully specified model with steady state at the origin (s=0, x=0).
    """
    def transition(
        s: np.ndarray, x: np.ndarray, e: np.ndarray, params: dict[str, float]
    ) -> np.ndarray:
        return np.array([params["rho"] * s[0] + params["sigma"] * e[0]], dtype=float)

    def arbitrage(
        s: np.ndarray,
        x: np.ndarray,
        s_next: np.ndarray,
        x_next: np.ndarray,
        params: dict[str, float],
    ) -> np.ndarray:
        return np.array(
            [x[0] - params["beta"] * x_next[0] - params["kappa"] * s[0]], dtype=float
        )

    return DSGEModel(
        state_names=("k",),
        control_names=("c",),
        shock_names=("eps",),
        parameters={"rho": rho, "beta": beta, "kappa": kappa, "sigma": sigma},
        steady_state_states=np.array([0.0]),
        steady_state_controls=np.array([0.0]),
        transition=transition,
        arbitrage=arbitrage,
    )

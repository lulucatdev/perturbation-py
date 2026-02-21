from __future__ import annotations

import numpy as np

from perturbation_py.model import DSGEModel


def build_model() -> DSGEModel:
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
        parameters={"rho": 0.9, "beta": 0.95, "kappa": 0.1, "sigma": 0.01},
        steady_state_states=np.array([0.0]),
        steady_state_controls=np.array([0.0]),
        transition=transition,
        arbitrage=arbitrage,
    )

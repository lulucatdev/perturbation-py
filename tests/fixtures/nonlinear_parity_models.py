"""Nonlinear model fixtures for second-order Dynare parity testing.

Each fixture provides a matched pair: a Python ``DSGEModel`` (deviation form,
steady state at zero) and a Dynare ``.mod`` file (level form) for the same
economic model.  Both should produce identical second-order perturbation
coefficients (ghx, ghu, ghxx, ghxu, ghuu, ghs2).

Models
------
rbc_simple
    A one-sector neoclassical growth (RBC) model with Cobb-Douglas production,
    full depreciation of capital each period simplified to allow log utility,
    and an i.i.d. TFP shock.  The non-linearity comes from k^alpha in the
    production function and the 1/c Euler equation.
"""
from __future__ import annotations

import numpy as np

from perturbation_py.model import DSGEModel


# ---------------------------------------------------------------------------
# RBC model parameters and steady state
# ---------------------------------------------------------------------------
_ALPHA = 0.33
_BETA = 0.99
_DELTA = 0.025
_SIGMA = 0.01  # shock scaling: TFP = exp(sigma * eps)

# Deterministic steady state
_K_SS = (_ALPHA * _BETA / (1 - _BETA * (1 - _DELTA))) ** (1 / (1 - _ALPHA))
_Y_SS = _K_SS ** _ALPHA
_C_SS = _Y_SS - _DELTA * _K_SS


def make_rbc_model() -> DSGEModel:
    """Create the RBC model as a Python ``DSGEModel`` in deviation form.

    Variables are deviations from steady state: ``k = K - K_ss``,
    ``c = C - C_ss``.  The steady state is ``(k, c) = (0, 0)``.

    Returns
    -------
    DSGEModel
    """
    def transition(s, x, e, p):
        k, c = s[0], x[0]
        kss, css = p["k_ss"], p["c_ss"]
        z = p["sigma"] * e[0]
        K = kss + k
        C = css + c
        K_next = (1 - p["delta"]) * K + K ** p["alpha"] * np.exp(z) - C
        return np.array([K_next - kss])

    def arbitrage(s, x, s_next, x_next, p):
        kss, css = p["k_ss"], p["c_ss"]
        C = css + x[0]
        C_next = css + x_next[0]
        K_next = kss + s_next[0]
        mpk = p["alpha"] * K_next ** (p["alpha"] - 1) + 1 - p["delta"]
        return np.array([1.0 - p["beta"] * (C / C_next) * mpk])

    return DSGEModel(
        state_names=("k",),
        control_names=("c",),
        shock_names=("eps",),
        parameters={
            "alpha": _ALPHA,
            "beta": _BETA,
            "delta": _DELTA,
            "sigma": _SIGMA,
            "k_ss": _K_SS,
            "c_ss": _C_SS,
        },
        steady_state_states=np.array([0.0]),
        steady_state_controls=np.array([0.0]),
        transition=transition,
        arbitrage=arbitrage,
    )


def rbc_dynare_mod() -> str:
    """Return a Dynare ``.mod`` file for the same RBC model (level form).

    The ``.mod`` uses ``predetermined_variables`` and ``stoch_simul(order=2)``.
    Steady-state values are computed analytically and set in ``initval``.

    Returns
    -------
    str
        Complete ``.mod`` file text.
    """
    return f"""\
var K C;
varexo eps;
predetermined_variables K;

parameters alpha beta delta sigma_e;
alpha  = {_ALPHA};
beta   = {_BETA};
delta  = {_DELTA};
sigma_e = {_SIGMA};

model;
  K(+1) = (1 - delta) * K + K ^ alpha * exp(sigma_e * eps) - C;
  1 = beta * (C / C(+1)) * (alpha * K(+1) ^ (alpha - 1) + 1 - delta);
end;

initval;
  K   = {_K_SS:.16g};
  C   = {_C_SS:.16g};
  eps = 0;
end;

steady;
check;

shocks;
  var eps; stderr 1;
end;

stoch_simul(order=2, irf=0, noprint, nograph, nocorr, nomoments);
"""


# ---------------------------------------------------------------------------
# Catalogue for parametrized tests
# ---------------------------------------------------------------------------

NONLINEAR_PARITY_SUITE = [
    {
        "name": "rbc_simple",
        "model_fn": make_rbc_model,
        "mod_fn": rbc_dynare_mod,
    },
]

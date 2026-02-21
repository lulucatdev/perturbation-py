"""Numerical steady-state solver for DSGE models.

Computes the deterministic steady state of a DSGE model by finding the
fixed point of the system:

.. math::

    s = g(s, x, 0)  \\quad \\text{and} \\quad 0 = f(s, x, s, x)

where *g* is the transition function and *f* is the arbitrage
(Euler-equation) system.  The solver uses ``scipy.optimize.root`` with
configurable algorithm and tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import root

from .model import DSGEModel

Array = np.ndarray


@dataclass(frozen=True)
class SteadyStateResult:
    """Result of the steady-state solver.

    Attributes
    ----------
    states : Array, shape (n_s,)
        Steady-state values of the state variables.
    controls : Array, shape (n_x,)
        Steady-state values of the control variables.
    residual : Array, shape (n_s + n_x,)
        Residual of the static equilibrium conditions at the solution.
    residual_norm : float
        Infinity-norm of the residual.
    success : bool
        Whether the underlying root-finder converged.
    message : str
        Status message from the root-finder.
    nfev : int
        Number of function evaluations used.
    """

    states: Array
    controls: Array
    residual: Array
    residual_norm: float
    success: bool
    message: str
    nfev: int


def solve_steady_state(
    model: DSGEModel,
    guess_states: Sequence[float] | Array | None = None,
    guess_controls: Sequence[float] | Array | None = None,
    *,
    tol: float = 1e-10,
    method: str = "hybr",
    maxfev: int | None = None,
) -> SteadyStateResult:
    """Find the deterministic steady state of a DSGE model.

    Solves the nonlinear system :math:`s = g(s, x, 0)` and
    :math:`0 = f(s, x, s, x)` for the fixed point
    :math:`(\\bar{s}, \\bar{x})` using ``scipy.optimize.root``.

    Parameters
    ----------
    model : DSGEModel
        The model whose steady state is sought.
    guess_states : array_like or None
        Initial guess for states; defaults to ``model.steady_state_states``.
    guess_controls : array_like or None
        Initial guess for controls; defaults to
        ``model.steady_state_controls``.
    tol : float
        Convergence tolerance for the root-finder.
    method : str
        Root-finding algorithm passed to ``scipy.optimize.root``
        (default ``"hybr"``).
    maxfev : int or None
        Maximum number of function evaluations.

    Returns
    -------
    SteadyStateResult
        The solved steady state with convergence diagnostics.

    Raises
    ------
    RuntimeError
        If the root-finder fails to converge.
    """
    if guess_states is None:
        states0 = np.asarray(model.steady_state_states, dtype=float)
    else:
        states0 = np.asarray(guess_states, dtype=float).reshape(-1)

    if guess_controls is None:
        controls0 = np.asarray(model.steady_state_controls, dtype=float)
    else:
        controls0 = np.asarray(guess_controls, dtype=float).reshape(-1)

    if states0.size != model.n_states:
        raise ValueError("guess_states has wrong size")
    if controls0.size != model.n_controls:
        raise ValueError("guess_controls has wrong size")

    z0 = np.concatenate([states0, controls0])

    options: dict[str, int] = {}
    if maxfev is not None:
        options["maxfev"] = int(maxfev)

    result = root(
        lambda z: model.static_residual_from_vector(z),
        z0,
        method=method,
        tol=tol,
        options=options,
    )

    residual = np.asarray(model.static_residual_from_vector(result.x), dtype=float)
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))

    states, controls = model.split_state_control_vector(result.x)
    output = SteadyStateResult(
        states=states,
        controls=controls,
        residual=residual,
        residual_norm=residual_norm,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(getattr(result, "nfev", 0)),
    )

    if not output.success:
        raise RuntimeError(f"Steady-state solver failed: {output.message}")

    return output

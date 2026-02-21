"""Numerical computation of first-order derivatives (Jacobians) for DSGE models.

Evaluates the Jacobian matrices of the transition function *g* and the
arbitrage function *f* at the deterministic steady state using numerical
differentiation.  These Jacobians form the linear system that underpins
the first-order perturbation solution.

Notation
--------
Let *s* denote states, *x* controls, *e* shocks, *S* = s' (next-period
states), and *X* = x' (next-period controls).  The transition is
``s' = g(s, x, e)`` and the arbitrage condition is ``0 = f(s, x, S, X)``.
The Jacobians are:

* ``g_s``, ``g_x``, ``g_e`` -- partial derivatives of *g* w.r.t. *s*,
  *x*, *e*.
* ``f_s``, ``f_x``, ``f_S``, ``f_X`` -- partial derivatives of *f*
  w.r.t. *s*, *x*, *S*, *X*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .derivative_backends import complex_step_jacobian, finite_difference_jacobian
from .model import DSGEModel

Array = np.ndarray


@dataclass(frozen=True)
class Jacobians:
    """First-order partial derivatives of the transition and arbitrage functions.

    All Jacobians are evaluated at the deterministic steady state.

    Attributes
    ----------
    g_s : Array, shape (n_s, n_s)
        :math:`\\partial g / \\partial s` -- transition Jacobian w.r.t.
        current states.
    g_x : Array, shape (n_s, n_x)
        :math:`\\partial g / \\partial x` -- transition Jacobian w.r.t.
        current controls.
    g_e : Array, shape (n_s, n_e)
        :math:`\\partial g / \\partial e` -- transition Jacobian w.r.t.
        shocks.
    f_s : Array, shape (n_x, n_s)
        :math:`\\partial f / \\partial s` -- arbitrage Jacobian w.r.t.
        current states.
    f_x : Array, shape (n_x, n_x)
        :math:`\\partial f / \\partial x` -- arbitrage Jacobian w.r.t.
        current controls.
    f_S : Array, shape (n_x, n_s)
        :math:`\\partial f / \\partial s'` -- arbitrage Jacobian w.r.t.
        next-period states.
    f_X : Array, shape (n_x, n_x)
        :math:`\\partial f / \\partial x'` -- arbitrage Jacobian w.r.t.
        next-period controls.
    """

    g_s: Array
    g_x: Array
    g_e: Array
    f_s: Array
    f_x: Array
    f_S: Array
    f_X: Array


def compute_jacobians(
    model: DSGEModel, epsilon: float = 1e-6, backend: str = "finite_difference"
) -> Jacobians:
    """Compute all first-order Jacobians of the model at the steady state.

    Differentiates the transition function *g* with respect to ``(s, x, e)``
    and the arbitrage function *f* with respect to ``(s, x, s', x')``,
    all evaluated at the deterministic steady state where
    ``s' = s = s_bar`` and ``x' = x = x_bar``.

    Parameters
    ----------
    model : DSGEModel
        The model whose derivatives are to be computed.
    epsilon : float
        Perturbation step size for the numerical differentiation backend.
    backend : {"finite_difference", "complex_step"}
        Numerical differentiation method.  ``"finite_difference"`` uses
        central differences (O(epsilon^2)); ``"complex_step"`` uses the
        Squire-Trapp (1998) method for near machine-precision accuracy.

    Returns
    -------
    Jacobians
        Dataclass containing all seven Jacobian matrices.
    """
    s = np.asarray(model.steady_state_states, dtype=float)
    x = np.asarray(model.steady_state_controls, dtype=float)
    e = np.asarray(model.steady_state_shocks, dtype=float)

    def g(ss: Array, xx: Array, ee: Array) -> Array:
        return np.asarray(model.transition(ss, xx, ee, model.parameters))

    def f(ss: Array, xx: Array, s_next: Array, x_next: Array) -> Array:
        return np.asarray(model.arbitrage(ss, xx, s_next, x_next, model.parameters))

    if backend == "finite_difference":
        jacobian = finite_difference_jacobian
    elif backend == "complex_step":
        jacobian = complex_step_jacobian
    else:
        raise ValueError(f"Unknown derivative backend: {backend}")

    g_s = jacobian(g, (s, x, e), 0, epsilon)
    g_x = jacobian(g, (s, x, e), 1, epsilon)
    g_e = jacobian(g, (s, x, e), 2, epsilon)

    f_s = jacobian(f, (s, x, s, x), 0, epsilon)
    f_x = jacobian(f, (s, x, s, x), 1, epsilon)
    f_S = jacobian(f, (s, x, s, x), 2, epsilon)
    f_X = jacobian(f, (s, x, s, x), 3, epsilon)

    return Jacobians(g_s=g_s, g_x=g_x, g_e=g_e, f_s=f_s, f_x=f_x, f_S=f_S, f_X=f_X)

"""Compute model Hessians (and optionally third derivatives) by finite differences of Jacobians.

These higher-order derivatives of the transition ``g`` and arbitrage ``f``
equations are needed by the closed-form Sylvester-equation solvers at orders 2
and 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .derivatives import Jacobians, compute_jacobians
from .model import DSGEModel

Array = np.ndarray


@dataclass(frozen=True)
class ModelHessians:
    """Second-order derivatives of *f* and *g* at steady state.

    Shapes (using ``n_v = n_s + n_x``, ``n_w = n_v + n_e``):

    * ``f2``: ``(n_x, 2*n_v, 2*n_v)`` — Hessian of arbitrage w.r.t.
      ``(s, x, s', x')``
    * ``g2``: ``(n_s, n_w, n_w)`` — Hessian of transition w.r.t.
      ``(s, x, e)``
    """
    f2: Array
    g2: Array


@dataclass(frozen=True)
class ModelThirdDerivatives:
    """Third-order derivatives of *f* and *g* at steady state.

    * ``f3``: ``(n_x, 2*n_v, 2*n_v, 2*n_v)``
    * ``g3``: ``(n_s, n_w, n_w, n_w)``
    """
    f3: Array
    g3: Array


def compute_model_hessians(
    model: DSGEModel,
    jacobians: Jacobians | None = None,
    epsilon: float = 1e-5,
    jac_epsilon: float = 1e-6,
    backend: str = "finite_difference",
) -> ModelHessians:
    """Compute second derivatives of *f* and *g* by finite-differencing the Jacobians.

    For each input dimension *i*, the steady state is perturbed by
    ``+/- epsilon`` and the full Jacobian is recomputed.  A central
    finite-difference formula then yields the Hessian column::

        H[:, :, i] = (J(+h) - J(-h)) / (2h)

    The resulting tensor is symmetrised over the last two indices to
    enforce Schwarz's theorem (equality of mixed partials).

    Parameters
    ----------
    model : DSGEModel
        Model specification including steady state and equation callables.
    jacobians : Jacobians or None, optional
        Pre-computed Jacobians (currently unused but reserved for
        analytic-Hessian backends).
    epsilon : float, optional
        Step size for the outer (Hessian-level) finite difference.
    jac_epsilon : float, optional
        Step size passed to :func:`compute_jacobians` for the inner
        Jacobian evaluation at each shifted point.
    backend : str, optional
        Jacobian computation backend (default ``"finite_difference"``).

    Returns
    -------
    ModelHessians
        Dataclass with fields ``f2`` of shape ``(n_x, 2*n_v, 2*n_v)`` and
        ``g2`` of shape ``(n_s, n_w, n_w)`` where ``n_v = n_s + n_x`` and
        ``n_w = n_v + n_e``.

    Notes
    -----
    The Hessian of the transition function *g* is differentiated with
    respect to the stacked vector ``(s, x, e)`` of dimension ``n_w``.
    The Hessian of the arbitrage function *f* is differentiated with
    respect to ``(s, x, s', x')`` of dimension ``2 * n_v``.
    """
    n_s = model.n_states
    n_x = model.n_controls
    n_e = model.n_shocks
    n_v = n_s + n_x
    n_w = n_v + n_e  # total inputs to g

    s0 = model.steady_state_states.copy()
    x0 = model.steady_state_controls.copy()
    e0 = model.steady_state_shocks.copy()

    # --- Hessian of g: shape (n_s, n_w, n_w) ---
    # g depends on (s, x, e).  We perturb each input dimension and
    # re-compute the Jacobian of g, then take central differences.
    g2 = np.zeros((n_s, n_w, n_w), dtype=float)

    for i in range(n_w):
        s_p, x_p, e_p = s0.copy(), x0.copy(), e0.copy()
        s_m, x_m, e_m = s0.copy(), x0.copy(), e0.copy()
        h = epsilon
        if i < n_s:
            s_p[i] += h
            s_m[i] -= h
        elif i < n_v:
            x_p[i - n_s] += h
            x_m[i - n_s] -= h
        else:
            e_p[i - n_v] += h
            e_m[i - n_v] -= h

        model_p = _shifted_model(model, s_p, x_p, e_p)
        model_m = _shifted_model(model, s_m, x_m, e_m)

        jac_p = compute_jacobians(model_p, epsilon=jac_epsilon, backend=backend)
        jac_m = compute_jacobians(model_m, epsilon=jac_epsilon, backend=backend)

        # Stack full Jacobian rows: [g_s | g_x | g_e]
        full_p = np.hstack([jac_p.g_s, jac_p.g_x, jac_p.g_e])
        full_m = np.hstack([jac_m.g_s, jac_m.g_x, jac_m.g_e])
        g2[:, :, i] = (full_p - full_m) / (2.0 * h)

    # Symmetrise
    g2 = 0.5 * (g2 + g2.transpose(0, 2, 1))

    # --- Hessian of f: shape (n_x, 2*n_v, 2*n_v) ---
    # f depends on (s, x, s', x').  Dimension = 2*n_v.
    f2 = np.zeros((n_x, 2 * n_v, 2 * n_v), dtype=float)

    for i in range(2 * n_v):
        s_p, x_p = s0.copy(), x0.copy()
        s_m, x_m = s0.copy(), x0.copy()
        sn_p, xn_p = s0.copy(), x0.copy()
        sn_m, xn_m = s0.copy(), x0.copy()
        h = epsilon

        if i < n_s:
            s_p[i] += h
            s_m[i] -= h
        elif i < n_v:
            x_p[i - n_s] += h
            x_m[i - n_s] -= h
        elif i < n_v + n_s:
            sn_p[i - n_v] += h
            sn_m[i - n_v] -= h
        else:
            xn_p[i - n_v - n_s] += h
            xn_m[i - n_v - n_s] -= h

        model_p = _shifted_model_f(model, s_p, x_p, sn_p, xn_p)
        model_m = _shifted_model_f(model, s_m, x_m, sn_m, xn_m)

        jac_p = compute_jacobians(model_p, epsilon=jac_epsilon, backend=backend)
        jac_m = compute_jacobians(model_m, epsilon=jac_epsilon, backend=backend)

        # Stack: [f_s | f_x | f_S | f_X]
        full_p = np.hstack([jac_p.f_s, jac_p.f_x, jac_p.f_S, jac_p.f_X])
        full_m = np.hstack([jac_m.f_s, jac_m.f_x, jac_m.f_S, jac_m.f_X])
        f2[:, :, i] = (full_p - full_m) / (2.0 * h)

    # Symmetrise
    f2 = 0.5 * (f2 + f2.transpose(0, 2, 1))

    return ModelHessians(f2=f2, g2=g2)


def compute_model_third_derivatives(
    model: DSGEModel,
    hessians: ModelHessians | None = None,
    epsilon: float = 1e-4,
    hess_epsilon: float = 1e-5,
    jac_epsilon: float = 1e-6,
    backend: str = "finite_difference",
) -> ModelThirdDerivatives:
    """Compute third derivatives of *f* and *g* by finite-differencing the Hessians.

    The approach mirrors :func:`compute_model_hessians` but operates one
    level higher: for each input dimension *i* the steady state is perturbed
    by ``+/- epsilon``, the full Hessian is recomputed at the shifted point,
    and a central finite-difference yields the third-derivative slice::

        T[:, :, :, i] = (H2(+h) - H2(-h)) / (2h)

    The result is symmetrised over all six permutations of the last three
    indices.

    Parameters
    ----------
    model : DSGEModel
        Model specification including steady state and equation callables.
    hessians : ModelHessians or None, optional
        Pre-computed Hessians (currently unused but reserved for future
        analytic backends).
    epsilon : float, optional
        Outer (third-derivative-level) finite-difference step size.
    hess_epsilon : float, optional
        Step size passed to :func:`compute_model_hessians` for the inner
        Hessian evaluation.
    jac_epsilon : float, optional
        Step size passed to :func:`compute_jacobians` within each Hessian
        evaluation.
    backend : str, optional
        Jacobian computation backend (default ``"finite_difference"``).

    Returns
    -------
    ModelThirdDerivatives
        Dataclass with fields ``f3`` of shape
        ``(n_x, 2*n_v, 2*n_v, 2*n_v)`` and ``g3`` of shape
        ``(n_s, n_w, n_w, n_w)``.

    Notes
    -----
    Because the third derivative is obtained by finite-differencing
    second derivatives that are themselves finite-differenced Jacobians,
    accuracy degrades quickly.  Use progressively larger step sizes at
    each level (``epsilon > hess_epsilon > jac_epsilon``) to balance
    truncation and round-off error.
    """
    n_s = model.n_states
    n_x = model.n_controls
    n_e = model.n_shocks
    n_v = n_s + n_x
    n_w = n_v + n_e

    s0 = model.steady_state_states.copy()
    x0 = model.steady_state_controls.copy()
    e0 = model.steady_state_shocks.copy()

    # --- g3: shape (n_s, n_w, n_w, n_w) ---
    g3 = np.zeros((n_s, n_w, n_w, n_w), dtype=float)
    for i in range(n_w):
        s_p, x_p, e_p = s0.copy(), x0.copy(), e0.copy()
        s_m, x_m, e_m = s0.copy(), x0.copy(), e0.copy()
        h = epsilon
        if i < n_s:
            s_p[i] += h; s_m[i] -= h
        elif i < n_v:
            x_p[i - n_s] += h; x_m[i - n_s] -= h
        else:
            e_p[i - n_v] += h; e_m[i - n_v] -= h

        model_p = _shifted_model(model, s_p, x_p, e_p)
        model_m = _shifted_model(model, s_m, x_m, e_m)

        h2_p = compute_model_hessians(model_p, epsilon=hess_epsilon, jac_epsilon=jac_epsilon, backend=backend)
        h2_m = compute_model_hessians(model_m, epsilon=hess_epsilon, jac_epsilon=jac_epsilon, backend=backend)

        g3[:, :, :, i] = (h2_p.g2 - h2_m.g2) / (2.0 * h)

    # Symmetrise over last 3 indices
    g3 = _symmetrise_3(g3)

    # --- f3: shape (n_x, 2*n_v, 2*n_v, 2*n_v) ---
    f3 = np.zeros((n_x, 2 * n_v, 2 * n_v, 2 * n_v), dtype=float)
    for i in range(2 * n_v):
        s_p, x_p = s0.copy(), x0.copy()
        s_m, x_m = s0.copy(), x0.copy()
        sn_p, xn_p = s0.copy(), x0.copy()
        sn_m, xn_m = s0.copy(), x0.copy()
        h = epsilon

        if i < n_s:
            s_p[i] += h; s_m[i] -= h
        elif i < n_v:
            x_p[i - n_s] += h; x_m[i - n_s] -= h
        elif i < n_v + n_s:
            sn_p[i - n_v] += h; sn_m[i - n_v] -= h
        else:
            xn_p[i - n_v - n_s] += h; xn_m[i - n_v - n_s] -= h

        model_p = _shifted_model_f(model, s_p, x_p, sn_p, xn_p)
        model_m = _shifted_model_f(model, s_m, x_m, sn_m, xn_m)

        h2_p = compute_model_hessians(model_p, epsilon=hess_epsilon, jac_epsilon=jac_epsilon, backend=backend)
        h2_m = compute_model_hessians(model_m, epsilon=hess_epsilon, jac_epsilon=jac_epsilon, backend=backend)

        f3[:, :, :, i] = (h2_p.f2 - h2_m.f2) / (2.0 * h)

    f3 = _symmetrise_3(f3)

    return ModelThirdDerivatives(f3=f3, g3=g3)


def _symmetrise_3(T: Array) -> Array:
    """Average a 4-D tensor over all 6 permutations of the last 3 indices."""
    return (
        T
        + T.transpose(0, 1, 3, 2)
        + T.transpose(0, 2, 1, 3)
        + T.transpose(0, 2, 3, 1)
        + T.transpose(0, 3, 1, 2)
        + T.transpose(0, 3, 2, 1)
    ) / 6.0


def _shifted_model(
    model: DSGEModel,
    ss_states: Array,
    ss_controls: Array,
    ss_shocks: Array,
) -> DSGEModel:
    """Return a copy of *model* with a shifted steady-state point.

    This is used when computing Hessians of the *transition* function ``g``
    by finite differences.  The Jacobian of ``g`` is evaluated at a
    perturbed steady state ``(s0 +/- h_i, x0 +/- h_j, e0 +/- h_k)`` by
    creating a new :class:`DSGEModel` whose nominal steady state is the
    perturbed point.  Because ``compute_jacobians`` always linearises
    around the model's stored steady state, this indirection achieves the
    desired off-centre evaluation without modifying the original model.

    Parameters
    ----------
    model : DSGEModel
        Original model.
    ss_states : ndarray, shape (n_s,)
        Perturbed steady-state states.
    ss_controls : ndarray, shape (n_x,)
        Perturbed steady-state controls.
    ss_shocks : ndarray, shape (n_e,)
        Perturbed steady-state shocks.

    Returns
    -------
    DSGEModel
        A new model instance identical to *model* except for the
        steady-state vectors.
    """
    return DSGEModel(
        state_names=model.state_names,
        control_names=model.control_names,
        shock_names=model.shock_names,
        parameters=model.parameters,
        steady_state_states=ss_states,
        steady_state_controls=ss_controls,
        transition=model.transition,
        arbitrage=model.arbitrage,
        steady_state_shocks=ss_shocks,
        equations=model.equations,
    )


def _shifted_model_f(
    model: DSGEModel,
    ss_states: Array,
    ss_controls: Array,
    ss_states_next: Array,
    ss_controls_next: Array,
) -> DSGEModel:
    """Return a model whose arbitrage linearisation point is shifted.

    The arbitrage (Euler) equation ``f(s, x, s', x')`` depends on both
    current and next-period variables.  To compute its Hessian by finite
    differences we need to perturb any of the four blocks independently.

    Unlike ``_shifted_model``, simply moving the steady state is not
    sufficient: the Jacobian routine obtains ``s'`` from the transition
    ``g(s, x, e0)``, so we must also modify ``g`` (and ``f`` itself) to
    ensure they evaluate to zero at the desired shifted point.

    Concretely, two offsets are applied:

    1. **Transition offset** -- a constant is added to ``g`` so that
       ``g_shifted(ss_states, ss_controls, e0) = ss_states_next``.
    2. **Arbitrage offset** -- a constant is subtracted from ``f`` so that
       ``f_shifted(ss_states, ss_controls, ss_states_next, ss_controls_next) = 0``.

    These offsets guarantee that the Jacobian computation (which relies on
    the function being zero at the linearisation point) remains valid.

    Parameters
    ----------
    model : DSGEModel
        Original model.
    ss_states : ndarray, shape (n_s,)
        Perturbed current-period states.
    ss_controls : ndarray, shape (n_x,)
        Perturbed current-period controls.
    ss_states_next : ndarray, shape (n_s,)
        Perturbed next-period states.
    ss_controls_next : ndarray, shape (n_x,)
        Perturbed next-period controls.

    Returns
    -------
    DSGEModel
        A new model with offset transition and arbitrage functions and
        the perturbed steady state.
    """
    # We need to create a model where the steady state for the arbitrage
    # evaluation is (ss_states, ss_controls, ss_states_next, ss_controls_next).
    # The Jacobians of f are computed at the SS, where s' = g(s, x, e0).
    # So we must set SS such that:
    #   - current: (ss_states, ss_controls)
    #   - next:    (ss_states_next, ss_controls_next)
    # The transition must satisfy: g(ss_states, ss_controls, e0) = ss_states_next
    # But we only want to shift the linearization point, not the actual dynamics.
    # Solution: shift the transition so g_shifted returns ss_states_next at the new SS.

    original_transition = model.transition
    original_arbitrage = model.arbitrage
    e0 = model.steady_state_shocks.copy()

    # Compute what the original transition returns at the shifted point
    g_at_shifted = np.asarray(
        original_transition(ss_states, ss_controls, e0, model.parameters), dtype=float
    ).reshape(-1)

    # Offset so that the transition returns ss_states_next at the shifted SS
    offset = ss_states_next - g_at_shifted

    def shifted_transition(s, x, e, params):
        return np.asarray(original_transition(s, x, e, params), dtype=float).reshape(-1) + offset

    # Similarly shift arbitrage so it evaluates to zero at the shifted SS
    a_at_shifted = np.asarray(
        original_arbitrage(ss_states, ss_controls, ss_states_next, ss_controls_next, model.parameters),
        dtype=float,
    ).reshape(-1)

    def shifted_arbitrage(s, x, sn, xn, params):
        return np.asarray(original_arbitrage(s, x, sn, xn, params), dtype=float).reshape(-1) - a_at_shifted

    return DSGEModel(
        state_names=model.state_names,
        control_names=model.control_names,
        shock_names=model.shock_names,
        parameters=model.parameters,
        steady_state_states=ss_states,
        steady_state_controls=ss_controls,
        transition=shifted_transition,
        arbitrage=shifted_arbitrage,
        steady_state_shocks=e0,
        equations=model.equations,
    )
